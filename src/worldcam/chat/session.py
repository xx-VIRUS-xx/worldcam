"""Bounded per-user conversation memory so follow-ups ("and yesterday?") work.

In-memory only. Bounded three ways: last `max_turns` messages per user, each
capped to `max_chars`, entries expire after `ttl_s`, and at most `max_users`
sessions (least recently used evicted). Stores text only, never images.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable


class SessionStore:
    def __init__(self, max_turns: int = 6, ttl_s: float = 1800.0, max_chars: int = 400,
                 max_users: int = 100, clock: Callable[[], float] = time.time):
        self.max_turns, self.ttl_s, self.max_chars, self.max_users = max_turns, ttl_s, max_chars, max_users
        self._clock = clock
        self._s: "OrderedDict[str, list[tuple[float, str, str]]]" = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, uid: str, now: float) -> list:
        rows = [r for r in self._s.get(uid, []) if now - r[0] <= self.ttl_s]
        if rows:
            self._s[uid] = rows
        else:
            self._s.pop(uid, None)
        return rows

    def add(self, user_id: str, role: str, text: str) -> None:
        """role: 'user' | 'assistant'."""
        if role not in ("user", "assistant"):
            raise ValueError("role must be user|assistant")
        now = self._clock()
        with self._lock:
            rows = self._prune(user_id, now)
            rows.append((now, role, str(text)[: self.max_chars]))
            self._s[user_id] = rows[-self.max_turns:]
            self._s.move_to_end(user_id)
            while len(self._s) > self.max_users:
                self._s.popitem(last=False)

    def history(self, user_id: str) -> list[tuple[str, str]]:
        with self._lock:
            return [(r, t) for _, r, t in self._prune(user_id, self._clock())]

    def reset(self, user_id: str) -> None:
        with self._lock:
            self._s.pop(user_id, None)

    def clear(self) -> None:
        with self._lock:
            self._s.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._s)
