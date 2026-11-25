"""ChannelNotifier: the old ``Notifier`` protocol on top of a Channel + UserDirectory,
so the AmbiguityEngine keeps working unchanged in the multi-user world.

* ``ask(q)`` sends the question to one recipient: ``q.recipient`` (a User) or
  ``q.recipient_user_id`` if the question object carries one (optional attribute),
  else ``recipient_resolver(q)`` if given, else the owner. Refs are remembered
  (bounded) together with the recipient and the topic (``q.topic``, else
  ``topic_resolver(q)``, default "identity").
* Replies reach the adapter via ``feed(user, msg)`` (call it from the gateway's
  ``on_message``). ``poll_replies()`` drains them:
  - a reply to a question we sent, from a user who ``can_confirm(user, topic)``
    -> ``OwnerReply`` with ``raw = {user_id, role, topic, ...}``;
  - a reply from a user who may NOT confirm that topic is never a confirmation:
    it goes to ``on_other`` (e.g. the chat handler) so it is not lost.
* A bare text/tap with no ``reply_to_ref`` is matched to the latest unanswered
  question sent to that same user (the "just type the answer" habit).
* ``poll_replies(timeout)`` waits up to ``timeout_s`` for fed replies. It never
  polls the channel itself, so there is exactly one poller (the gateway).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Optional, Sequence

from ..messenger.base import OutgoingQuestion, OwnerReply
from ..users.api import User, can_confirm
from .api import Channel, InboundMessage, OutboundMessage

MAX_OPEN_QUESTIONS = 500
MAX_QUEUE = 500


class ChannelNotifier:
    def __init__(self, channel: Channel, users, *,
                 recipient_resolver: Optional[Callable[[OutgoingQuestion], Optional[User]]] = None,
                 topic_resolver: Optional[Callable[[OutgoingQuestion], str]] = None,
                 on_other: Optional[Callable[[User, InboundMessage], None]] = None,
                 clock: Callable[[], float] = time.time):
        self.channel = channel
        self.users = users
        self._recipient_resolver = recipient_resolver
        self._topic_resolver = topic_resolver
        self.on_other = on_other
        self._clock = clock
        self._cv = threading.Condition()
        self._open: "OrderedDict[str, dict]" = OrderedDict()   # ref -> state
        self._inbox: deque = deque(maxlen=MAX_QUEUE)

    # --------------------------------------------------------- recipients ---
    def _owner(self) -> User:
        for u in self.users.list():
            if u.role == "owner" and self.channel.name in u.channel_ids:
                return u
        raise RuntimeError("no active owner reachable on this channel")

    def _recipient(self, q: OutgoingQuestion) -> User:
        u = getattr(q, "recipient", None)
        if isinstance(u, User):
            return u
        uid = getattr(q, "recipient_user_id", None)
        if uid:
            u = self.users.get(uid)
            if u is not None and u.active:
                return u
        if self._recipient_resolver:
            u = self._recipient_resolver(q)
            if u is not None:
                return u
        return self._owner()

    def _topic(self, q: OutgoingQuestion) -> str:
        t = getattr(q, "topic", None)
        if isinstance(t, str) and t:
            return t
        return self._topic_resolver(q) if self._topic_resolver else "identity"

    # ----------------------------------------------------------- Notifier ---
    def ask(self, q: OutgoingQuestion) -> str:
        user = self._recipient(q)
        rid = user.channel_ids.get(self.channel.name)
        if not rid:
            raise RuntimeError(f"user {user.id} has no {self.channel.name} address")
        text = q.text
        if q.choices and q.allow_free_text:
            text += "\n\nTap a button, or reply to this message with your own answer."
        ref = self.channel.send(rid, OutboundMessage(text, tuple(q.image_paths), tuple(q.choices)))
        with self._cv:
            self._open[ref] = {"user_id": user.id, "topic": self._topic(q),
                               "doubt_id": q.doubt_id, "answered": False}
            while len(self._open) > MAX_OPEN_QUESTIONS:
                self._open.popitem(last=False)
        return ref

    def notify(self, text: str) -> None:
        owner = self._owner()
        self.channel.send(owner.channel_ids[self.channel.name], OutboundMessage(text))

    def feed(self, user: User, msg: InboundMessage) -> bool:
        """Offer an inbound message. True = consumed (queued as a reply, or routed
        to ``on_other`` because the sender may not confirm). False = unrelated;
        the caller treats it as ordinary chat."""
        with self._cv:
            if msg.reply_to_ref:
                ref = msg.reply_to_ref if msg.reply_to_ref in self._open else None
            else:
                mine = [r for r, q in self._open.items()
                        if q["user_id"] == user.id and not q["answered"]]
                ref = mine[-1] if mine else None
            if ref is None or (not msg.text.strip() and msg.choice_value is None):
                return False
            q = self._open[ref]
            if can_confirm(user, q["topic"]):
                if q["answered"]:
                    return False
                q["answered"] = True
                self._inbox.append((user, msg, ref, dict(q)))
                self._cv.notify_all()
                return True
        if self.on_other is None:
            return False
        self.on_other(user, msg)
        return True

    def poll_replies(self, timeout_s: float = 25.0) -> Sequence[OwnerReply]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cv:
            while not self._inbox:
                left = deadline - time.monotonic()
                if left <= 0:
                    return []
                self._cv.wait(left)
            items = list(self._inbox)
            self._inbox.clear()
        out = []
        for user, msg, ref, q in items:
            out.append(OwnerReply(
                message_ref=ref,
                text=None if msg.choice_value is not None else (msg.text or None),
                choice_value=msg.choice_value, ts=msg.ts or self._clock(),
                raw={"user_id": user.id, "role": user.role, "topic": q["topic"],
                     "channel": msg.channel, "modality": msg.modality,
                     "doubt_id": q["doubt_id"]}))
        return out
