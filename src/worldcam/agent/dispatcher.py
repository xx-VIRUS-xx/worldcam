"""Dispatcher: the only way the agent reaches users.

Applies, in order: user active -> attention budget (quiet hours, caps, ignore
back-off) -> channel routing -> `view_images` permission (images are stripped for
users without it) -> send -> record in the budget.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional, Sequence, Union

from ..channels.api import Channel, OutboundMessage
from ..users.api import User, UserDirectory, can

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SendResult:
    sent: bool
    user_id: str
    ref: Optional[str] = None
    channel: Optional[str] = None
    reason: str = ""            # when not sent: inactive | budget | no_channel | send_failed
    images_stripped: bool = False


class Dispatcher:
    def __init__(self, users: UserDirectory, channels: Union[dict, Sequence[Channel]], budget,
                 clock: Callable[[], float] = time.time, log_size: int = 200):
        self.users = users
        self.channels: dict = (dict(channels) if isinstance(channels, dict)
                               else {c.name: c for c in channels})
        self.budget = budget
        self.clock = clock
        self.sent_log: deque = deque(maxlen=log_size)    # bounded history of SendResult

    # -- single recipient ------------------------------------------------------
    def send(self, user: User, msg: OutboundMessage, *, kind: str = "question", priority: int = 1,
             profile: Optional[str] = None, now: Optional[float] = None) -> SendResult:
        now = self.clock() if now is None else now
        res = self._send(user, msg, kind, priority, profile, now)
        self.sent_log.append(res)
        return res

    def _send(self, user, msg, kind, priority, profile, now) -> SendResult:
        if not user.active:
            return SendResult(False, user.id, reason="inactive")
        if not self.budget.allow(user.id, priority=priority, profile=profile, now=now):
            return SendResult(False, user.id, reason="budget")
        route = self._route(user)
        if route is None:
            return SendResult(False, user.id, reason="no_channel")
        channel, rid = route
        stripped = False
        if msg.images and not can(user, "view_images"):
            msg, stripped = replace(msg, images=()), True
        try:
            ref = channel.send(rid, msg)
        except Exception as e:   # never leak message text/secrets into the log
            log.warning("send to %s via %s failed: %s", user.id, channel.name, type(e).__name__)
            return SendResult(False, user.id, reason="send_failed")
        self.budget.record(user.id, kind, now)
        return SendResult(True, user.id, str(ref), channel.name, images_stripped=stripped)

    def _route(self, user: User):
        for ch_name, rid in user.channel_ids.items():
            ch = self.channels.get(ch_name)
            if ch is not None:
                return ch, rid
        return None

    # -- many recipients ---------------------------------------------------------
    def recipients(self, permission: str, profile: Optional[str] = None) -> list[User]:
        """Holders of `permission` subscribed to `profile`, plus owners holding it (the owner is
        always responsible). If nobody subscribed, every holder."""
        holders = self.users.with_permission(permission)
        if not profile:
            return holders
        subs = {u.id for u in self.users.with_permission(permission, profile)}
        out = [u for u in holders if u.id in subs or u.role == "owner"]
        return out if subs else holders

    def send_many(self, users: Iterable[User], msg: OutboundMessage, *, kind: str, priority: int = 1,
                  profile: Optional[str] = None, now: Optional[float] = None) -> list[SendResult]:
        return [self.send(u, msg, kind=kind, priority=priority, profile=profile, now=now) for u in users]

    def alert(self, permission: str, text: str, *, priority: int = 2, profile: Optional[str] = None,
              images: Sequence[str] = (), now: Optional[float] = None, kind: str = "alert",
              silent: bool = False) -> list[SendResult]:
        msg = OutboundMessage(text=text, images=tuple(images)[:2], silent=silent)
        return self.send_many(self.recipients(permission, profile), msg, kind=kind,
                              priority=priority, profile=profile, now=now)
