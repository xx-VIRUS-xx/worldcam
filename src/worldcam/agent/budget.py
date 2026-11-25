"""Per-user attention budget: how often may the agent interrupt someone?

Policy (shared by the in-memory and SQLite stores):
  * per-hour and per-day caps by role (optionally tightened per profile);
  * quiet hours per user (default 23:00-08:00 local);
  * priority 3 bypasses quiet hours, the normal caps and any ignore cooldown,
    but never the hard daily ceiling (`hard_day_factor` x the daily cap);
  * ignore back-off: every consecutive "ignored" record (the user tapped Ignore, or
    never answered within N hours) doubles the cooldown; an "answered" resets it.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

DEFAULT_ROLE_LIMITS: dict = {   # role -> (per hour, per day)
    "owner": (6, 30),
    "household": (3, 12),
    "caregiver": (4, 16),
    "guest": (1, 3),
}


@dataclass
class BudgetConfig:
    role_limits: dict = field(default_factory=lambda: dict(DEFAULT_ROLE_LIMITS))
    profile_limits: dict = field(default_factory=dict)   # profile -> (per_hour, per_day), min()ed with the role cap
    default_limits: tuple = (3, 12)
    quiet_start_hour: int = 23
    quiet_end_hour: int = 8
    hard_day_factor: float = 2.0
    cooldown_base_s: float = 2 * 3600.0
    cooldown_max_s: float = 48 * 3600.0
    keep_s: float = 7 * 86400.0


def in_quiet(hour: int, start: Optional[int], end: Optional[int]) -> bool:
    if start is None or end is None or start == end:
        return False
    return (hour >= start or hour < end) if start > end else (start <= hour < end)


class _BudgetBase:
    """Policy on top of four storage primitives."""

    def __init__(self, config: Optional[BudgetConfig] = None,
                 role_of: Optional[Callable[[str], Optional[str]]] = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = config or BudgetConfig()
        self.role_of = role_of or (lambda uid: None)
        self.clock = clock

    # storage primitives (implemented by subclasses)
    def _count(self, user_id: str, since: float, until: float) -> int: raise NotImplementedError
    def _add(self, user_id: str, kind: str, ts: float) -> None: raise NotImplementedError
    def _get(self, user_id: str, key: str, default=None): raise NotImplementedError
    def _set(self, user_id: str, key: str, value) -> None: raise NotImplementedError

    # -- config -------------------------------------------------------------
    def set_quiet_hours(self, user_id: str, start: Optional[int], end: Optional[int]) -> None:
        """None disables quiet hours for this user."""
        self._set(user_id, "quiet", None if start is None or end is None else [int(start), int(end)])

    def quiet_hours(self, user_id: str) -> tuple:
        q = self._get(user_id, "quiet", "default")
        if q == "default":
            return (self.cfg.quiet_start_hour, self.cfg.quiet_end_hour)
        return (None, None) if q is None else (q[0], q[1])

    def limits(self, user_id: str, profile: Optional[str] = None) -> tuple:
        h, d = self.cfg.role_limits.get(self.role_of(user_id) or "", self.cfg.default_limits)
        if profile in self.cfg.profile_limits:
            ph, pd = self.cfg.profile_limits[profile]
            h, d = min(h, ph), min(d, pd)
        return h, d

    def hard_ceiling(self, user_id: str) -> int:
        _, d = self.limits(user_id)
        return max(d + 1, int(math.ceil(d * self.cfg.hard_day_factor)))

    # -- policy -------------------------------------------------------------
    def is_quiet(self, user_id: str, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        s, e = self.quiet_hours(user_id)
        return in_quiet(time.localtime(now).tm_hour, s, e)

    def cooldown_until(self, user_id: str) -> float:
        return float(self._get(user_id, "cooldown_until", 0.0) or 0.0)

    def allow(self, user_id: str, *, priority: int = 1, profile: Optional[str] = None,
              now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        day = self._count(user_id, now - 86400.0, now + 1)
        if day >= self.hard_ceiling(user_id):
            return False                                   # nobody bypasses the hard ceiling
        if priority >= 3:
            return True
        if self.is_quiet(user_id, now) or now < self.cooldown_until(user_id):
            return False
        per_hour, per_day = self.limits(user_id, profile)
        return self._count(user_id, now - 3600.0, now + 1) < per_hour and day < per_day

    def record(self, user_id: str, kind: str, now: Optional[float] = None) -> None:
        """kind: any send kind ("question", "alert", "checkin", "notify"...) counts against
        the caps. "ignored" starts/extends the back-off; "answered" resets it."""
        now = self.clock() if now is None else now
        if kind == "ignored":
            streak = int(self._get(user_id, "streak", 0) or 0) + 1
            self._set(user_id, "streak", streak)
            cd = min(self.cfg.cooldown_base_s * (2 ** (streak - 1)), self.cfg.cooldown_max_s)
            self._set(user_id, "cooldown_until", now + cd)
            return
        if kind == "answered":
            self._set(user_id, "streak", 0)
            self._set(user_id, "cooldown_until", 0.0)
            return
        self._add(user_id, kind, now)

    def remaining(self, user_id: str, now: Optional[float] = None) -> dict:
        now = self.clock() if now is None else now
        per_hour, per_day = self.limits(user_id)
        h = self._count(user_id, now - 3600.0, now + 1)
        d = self._count(user_id, now - 86400.0, now + 1)
        cu = self.cooldown_until(user_id)
        return {"hour": max(0, per_hour - h), "day": max(0, per_day - d),
                "hard_day": max(0, self.hard_ceiling(user_id) - d),
                "quiet": self.is_quiet(user_id, now),
                "cooldown_until": cu if cu > now else None,
                "ignore_streak": int(self._get(user_id, "streak", 0) or 0)}


class InMemoryAttentionBudget(_BudgetBase):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._events: dict = defaultdict(list)
        self._state: dict = {}
        self._lock = threading.Lock()

    def _count(self, user_id, since, until):
        with self._lock:
            return sum(1 for ts in self._events[user_id] if since <= ts < until)

    def _add(self, user_id, kind, ts):
        with self._lock:
            ev = self._events[user_id]
            ev.append(ts)
            if len(ev) > 500:
                self._events[user_id] = [t for t in ev if t >= ts - self.cfg.keep_s][-500:]

    def _get(self, user_id, key, default=None):
        with self._lock:
            return self._state.get((user_id, key), default)

    def _set(self, user_id, key, value):
        with self._lock:
            self._state[(user_id, key)] = value


class SqliteAttentionBudget(_BudgetBase):
    """Survives restarts. `path=":memory:"` works for tests."""

    def __init__(self, path: str = ":memory:", *a, **kw):
        super().__init__(*a, **kw)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS sends(user_id TEXT, kind TEXT, ts REAL)")
            self._db.execute("CREATE INDEX IF NOT EXISTS sends_u ON sends(user_id, ts)")
            self._db.execute("CREATE TABLE IF NOT EXISTS state(user_id TEXT, k TEXT, v TEXT, PRIMARY KEY(user_id, k))")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _count(self, user_id, since, until):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM sends WHERE user_id=? AND ts>=? AND ts<?",
                                    (user_id, since, until)).fetchone()[0]

    def _add(self, user_id, kind, ts):
        with self._lock, self._db:
            self._db.execute("INSERT INTO sends VALUES (?,?,?)", (user_id, kind, ts))
            self._db.execute("DELETE FROM sends WHERE user_id=? AND ts<?", (user_id, ts - self.cfg.keep_s))

    def _get(self, user_id, key, default=None):
        with self._lock:
            row = self._db.execute("SELECT v FROM state WHERE user_id=? AND k=?", (user_id, key)).fetchone()
        return default if row is None else json.loads(row[0])

    def _set(self, user_id, key, value):
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO state VALUES (?,?,?)", (user_id, key, json.dumps(value)))


def budget_from_users(users, path: Optional[str] = None, config: Optional[BudgetConfig] = None,
                      clock: Callable[[], float] = time.time):
    """Role lookup from a UserDirectory; SQLite when a path is given."""
    def role_of(uid):
        u = users.get(uid)
        return u.role if u else None
    if path:
        return SqliteAttentionBudget(path, config, role_of, clock)
    return InMemoryAttentionBudget(config, role_of, clock)
