"""Channel contract: how users talk to the agent and how it reaches them.
Telegram is the first channel. Voice (v2) is another channel; nothing else changes
because chat only sees text plus a `modality` tag.

Distinct from messenger.base.Notifier (owner-only, one question at a time); a
`ChannelNotifier` adapter (channels package) lets the existing AmbiguityEngine keep
using the Notifier protocol on top of a Channel.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from ..messenger.base import Choice


@dataclass(frozen=True)
class InboundMessage:
    channel: str                          # "telegram"
    sender_id: str                        # channel-specific id (Telegram chat id)
    text: str = ""
    ts: float = 0.0
    ref: Optional[str] = None             # this message's id on the channel
    reply_to_ref: Optional[str] = None    # the message it answers (a question we sent)
    choice_value: Optional[str] = None    # tapped button payload
    modality: str = "text"                # "text" | "voice" (v2: text is the transcript)
    attachments: tuple[str, ...] = ()     # local paths of received images
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundMessage:
    text: str
    images: tuple[str, ...] = ()          # local paths (max 2 on Telegram)
    choices: tuple[Choice, ...] = ()
    reply_to_ref: Optional[str] = None
    modality_hint: str = "text"           # "voice": keep it short and speakable
    silent: bool = False                  # no notification sound (companion nudges)


class Channel(Protocol):
    name: str
    def send(self, recipient_id: str, msg: OutboundMessage) -> str:
        """Returns the message ref used to match replies."""
    def poll(self, timeout_s: float = 25.0) -> Sequence[InboundMessage]:
        """Long-poll. Returns messages from ANY sender; authorization (the
        whitelist) is the caller's job, so unknown senders can be handled
        (e.g. /join <code>) or politely refused."""


class FakeChannel:
    name = "fake"

    def __init__(self):
        self.sent: list[tuple[str, str, OutboundMessage]] = []   # (ref, recipient, msg)
        self._inbox: list[InboundMessage] = []
        self._n = 0

    def send(self, recipient_id, msg):
        self._n += 1
        ref = f"m{self._n}"
        self.sent.append((ref, str(recipient_id), msg))
        return ref

    def push(self, text="", sender_id="1", **kw) -> InboundMessage:
        m = InboundMessage(self.name, str(sender_id), text, kw.pop("ts", time.time()),
                           ref=kw.pop("ref", f"in{len(self._inbox) + 1}"), **kw)
        self._inbox.append(m)
        return m

    def poll(self, timeout_s=25.0):
        out, self._inbox = self._inbox, []
        return out
