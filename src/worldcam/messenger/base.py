"""Messenger contract. Telegram is the first adapter; a Cloudflare messenger can
implement the same Protocol later. Only the owner is ever messaged."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class Choice:
    label: str        # button text shown to the owner
    value: str        # payload returned when tapped (<= 64 bytes for Telegram)


@dataclass(frozen=True)
class OutgoingQuestion:
    doubt_id: str
    text: str
    image_paths: tuple[str, ...] = ()     # local crops; 0-2 images
    choices: tuple[Choice, ...] = ()      # optional quick-reply buttons
    allow_free_text: bool = True


@dataclass(frozen=True)
class OwnerReply:
    message_ref: str                       # id of the question message being answered
    text: Optional[str] = None             # free-text reply
    choice_value: Optional[str] = None     # tapped button payload
    ts: float = 0.0
    raw: dict = field(default_factory=dict)


class Notifier(Protocol):
    def ask(self, q: OutgoingQuestion) -> str:
        """Send to the owner; return message_ref used to match the reply."""
    def notify(self, text: str) -> None:
        """Plain informational message to the owner."""
    def poll_replies(self, timeout_s: float = 25.0) -> Sequence[OwnerReply]:
        """Long-poll for owner replies. Messages from anyone else are dropped."""


class FakeNotifier:
    """In-memory notifier for tests: records questions, replays queued replies."""

    def __init__(self):
        self.sent: list[tuple[str, OutgoingQuestion]] = []
        self.notes: list[str] = []
        self._pending: list[OwnerReply] = []
        self._n = 0

    def ask(self, q: OutgoingQuestion) -> str:
        self._n += 1
        ref = f"msg_{self._n}"
        self.sent.append((ref, q))
        return ref

    def notify(self, text: str) -> None:
        self.notes.append(text)

    def queue_reply(self, reply: OwnerReply) -> None:
        self._pending.append(reply)

    def poll_replies(self, timeout_s: float = 25.0) -> Sequence[OwnerReply]:
        out, self._pending = self._pending, []
        return out
