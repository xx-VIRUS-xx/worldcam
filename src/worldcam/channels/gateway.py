"""ChannelGateway: the single door between a Channel and the rest of the agent.

Polls one channel, authorizes every sender against the UserDirectory and hands
``(User, InboundMessage)`` to ``on_message``.

* Whitelisted + active user with the ``chat`` permission -> dispatched.
* Unknown sender sending ``/join CODE`` (or ``/start CODE``, the t.me deep link) ->
  the invite is redeemed (single use, expiring), a welcome is sent and users who
  hold ``manage_users`` are told who joined. Wrong codes are answered with one
  neutral line; after ``max_join_failures`` per hour the sender is silently ignored.
* Any other unknown sender (and deactivated users) -> at most ONE refusal line per
  sender per ``refusal_interval_s`` (default a day), otherwise silence. Their
  content is never echoed, stored or logged (only a count).
* One poller per channel: Telegram answers 409 Conflict to a second getUpdates
  consumer. Run exactly one gateway per bot token (and do not also poll with
  ``TelegramNotifier``); a 409 is logged and retried with backoff.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Optional

from ..users.api import User, can
from ..users.store import InviteError
from .api import Channel, InboundMessage, OutboundMessage

log = logging.getLogger("worldcam.channels.gateway")

REFUSAL_TEXT = "This is a private assistant. Ask its owner for an invite."
BAD_CODE_TEXT = "That invite code is not valid."
_JOIN_COMMANDS = ("/join", "/start")


def parse_join(text: str) -> Optional[str]:
    """'/join CODE', '/join@bot CODE', '/start CODE' -> CODE, else None."""
    parts = (text or "").strip().split()
    if len(parts) != 2:
        return None
    cmd = parts[0].lower().split("@", 1)[0]
    return parts[1] if cmd in _JOIN_COMMANDS else None


class ChannelGateway:
    def __init__(self, channel: Channel, users, on_message: Callable[[User, InboundMessage], None], *,
                 on_joined: Optional[Callable[[User], None]] = None,
                 poll_timeout_s: float = 25.0, refusal_interval_s: float = 24 * 3600,
                 max_join_failures: int = 5, join_window_s: float = 3600.0,
                 max_tracked_senders: int = 2000,
                 clock: Callable[[], float] = time.time):
        self.channel = channel
        self.users = users
        self.on_message = on_message
        self.on_joined = on_joined
        self.poll_timeout_s = poll_timeout_s
        self.refusal_interval_s = refusal_interval_s
        self.max_join_failures = max_join_failures
        self.join_window_s = join_window_s
        self._max_tracked = max_tracked_senders
        self._clock = clock
        self._refused: "OrderedDict[str, float]" = OrderedDict()
        self._join_fail: "OrderedDict[str, deque]" = OrderedDict()
        self.stats = {"dispatched": 0, "refused": 0, "ignored": 0, "joined": 0, "errors": 0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if hasattr(channel, "attachment_filter"):   # strangers never make us save files
            channel.attachment_filter = lambda sid: self.users.by_channel(
                channel.name, sid) is not None

    # ------------------------------------------------------------ dispatch ---
    def handle(self, msg: InboundMessage) -> None:
        user = self.users.by_channel(msg.channel, msg.sender_id)
        if user is not None:
            if not can(user, "chat"):
                self.stats["ignored"] += 1
                return
            self.stats["dispatched"] += 1
            try:
                self.on_message(user, msg)
            except Exception as e:   # a handler bug must not kill the poller
                self.stats["errors"] += 1
                log.error("on_message failed: %s", type(e).__name__)
            return
        self._unknown(msg)

    def _unknown(self, msg: InboundMessage) -> None:
        now = self._clock()
        key = f"{msg.channel}:{msg.sender_id}"
        code = parse_join(msg.text) if msg.modality == "text" and not msg.choice_value else None
        if code is not None and hasattr(self.users, "redeem"):
            fails = self._join_fail.setdefault(key, deque())
            while fails and now - fails[0] > self.join_window_s:
                fails.popleft()
            if len(fails) >= self.max_join_failures:
                self.stats["ignored"] += 1
                return
            name = str(msg.raw.get("sender_name") or "guest")
            try:
                user = self.users.redeem(code, msg.channel, msg.sender_id, name)
            except ValueError:   # InviteError or "already whitelisted"
                fails.append(now)
                self._trim(self._join_fail)
                self._reply(msg, BAD_CODE_TEXT)
                return
            self.stats["joined"] += 1
            log.info("user joined via invite (id=%s, role=%s)", user.id, user.role)
            self._reply(msg, f"Welcome, {user.name}! You are now connected as {user.role}. "
                             "Just write to me; I will tell you if I cannot help with something.")
            self._tell_admins(user)
            if self.on_joined:
                try:
                    self.on_joined(user)
                except Exception as e:
                    log.error("on_joined failed: %s", type(e).__name__)
            return
        last = self._refused.get(key)
        if last is not None and now - last < self.refusal_interval_s:
            self.stats["ignored"] += 1
            return
        self._refused[key] = now
        self._refused.move_to_end(key)
        self._trim(self._refused)
        self.stats["refused"] += 1
        self._reply(msg, REFUSAL_TEXT)

    def _trim(self, od: OrderedDict) -> None:
        while len(od) > self._max_tracked:
            od.popitem(last=False)

    def _reply(self, msg: InboundMessage, text: str) -> None:
        try:
            self.channel.send(msg.sender_id, OutboundMessage(text, silent=True))
        except Exception as e:
            log.warning("reply failed: %s", type(e).__name__)

    def _tell_admins(self, new: User) -> None:
        try:
            for admin in self.users.with_permission("manage_users"):
                rid = admin.channel_ids.get(self.channel.name)
                if rid and admin.id != new.id:
                    self.channel.send(rid, OutboundMessage(
                        f"{new.name} joined as {new.role} (user {new.id}). "
                        "Change or remove access with worldcam.cli.users_admin.", silent=True))
        except Exception as e:
            log.warning("admin notice failed: %s", type(e).__name__)

    # --------------------------------------------------------------- loop ---
    def poll_once(self, timeout_s: Optional[float] = None) -> int:
        msgs = self.channel.poll(self.poll_timeout_s if timeout_s is None else timeout_s)
        for m in msgs:
            self.handle(m)
        return len(msgs)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="channel-gateway", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 35.0) -> None:
        """Signal the loop and wait (a long poll may take up to poll_timeout_s)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout_s)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.poll_once()
                backoff = 1.0
            except Exception as e:
                self.stats["errors"] += 1
                code = getattr(e, "error_code", None)
                log.warning("poll failed (%s%s); retry in %.0fs", type(e).__name__,
                            ", 409: another poller uses this bot token" if code == 409 else "",
                            backoff)
                self._stop.wait(backoff)
                backoff = min(60.0, backoff * 2)
