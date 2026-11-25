"""The brain's time dimension: events (a timeline), routines (learned
regularities) and summaries. The graph (graph/api.py) holds WHO/WHAT exists;
this holds WHAT HAPPENED and WHEN. Both are keyed by the same entity ids
(Person.id covers people, animals and objects).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, Sequence

EVENT_KINDS = (
    "entered", "left", "zone_changed", "stationary", "seen",
    "activity_changed",       # eating, taking_medication, lying_down, fall_suspected ...
    "alert",                  # something a profile rule flagged
    "owner_statement",        # a user said it ("took my pills at 9")
    "observation",            # scene description / curiosity look-around
)
EVENT_SOURCES = ("camera", "user", "inferred")


@dataclass(frozen=True)
class BrainEvent:
    id: str
    ts: float
    kind: str
    subject_id: Optional[str] = None      # graph Person.id (person/animal/object); None = unknown
    subject_label: Optional[str] = None   # "unknown person", "dog", "Rahul"
    zone: Optional[str] = None
    camera: str = "default"
    activity: Optional[str] = None
    confidence: float = 0.0
    source: str = "camera"
    snapshot_paths: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()        # profiles this event matters to
    detail: dict = field(default_factory=dict)   # track ids, doubt ids, confirmed_by, ...


def new_event(kind: str, **kw) -> BrainEvent:
    return BrainEvent(id=kw.pop("id", "ev_" + uuid.uuid4().hex[:12]),
                      ts=kw.pop("ts", time.time()), kind=kind, **kw)


class EventStore(Protocol):
    def add_event(self, e: BrainEvent) -> str: ...
    def get_event(self, event_id: str) -> Optional[BrainEvent]: ...
    def query(self, *, since: Optional[float] = None, until: Optional[float] = None,
              subject_id: Optional[str] = None, kind: Optional[str] = None,
              zone: Optional[str] = None, profile: Optional[str] = None,
              limit: int = 100, newest_first: bool = True) -> list[BrainEvent]: ...
    def last_event(self, subject_id: str, kind: Optional[str] = None) -> Optional[BrainEvent]: ...
    def count(self, **filters) -> int: ...
    def purge_before(self, ts: float) -> list[str]:
        """Retention. Returns snapshot paths the caller must delete from disk."""
    def forget_subject(self, subject_id: str) -> list[str]:
        """Hard-delete a subject's events AND routines. Returns snapshot paths to delete."""
    def delete_event(self, event_id: str) -> list[str]:
        """Delete one event; returns its snapshot paths (retention, corrections)."""


ROUTINE_KINDS = ("daily_at", "weekday_at", "interval", "presence_window")


@dataclass(frozen=True)
class Routine:
    id: str
    subject_id: str
    kind: str                              # one of ROUTINE_KINDS
    description: str                       # "takes medication around 09:00"
    params: dict = field(default_factory=dict)   # {"event_kind": "activity_changed", "activity": "taking_medication", "minute_of_day": 540, "tolerance_min": 45, "weekdays": [..]}
    support: int = 0                       # how many days it was observed
    confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    source: str = "inferred"               # mined = inferred; owner-declared = user
    profiles: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutineStatus:
    routine: Routine
    state: str                             # "on_time" | "due" | "overdue" | "not_applicable"
    expected_at: Optional[float] = None
    last_done: Optional[float] = None


class RoutineStore(Protocol):
    def upsert_routine(self, r: Routine) -> str: ...
    def list_routines(self, subject_id: Optional[str] = None, profile: Optional[str] = None) -> list[Routine]: ...
    def status(self, now: float, subject_id: Optional[str] = None) -> list[RoutineStatus]: ...
    def delete_routine(self, routine_id: str) -> None: ...


@dataclass(frozen=True)
class DaySummary:
    day: str                               # "2026-11-25" (local date)
    text: str
    event_ids: tuple[str, ...] = ()
    stats: dict = field(default_factory=dict)
    profile: Optional[str] = None


class InMemoryEventStore:
    """Reference implementation for tests (also implements RoutineStore.upsert/list)."""

    def __init__(self):
        self.events: dict[str, BrainEvent] = {}
        self.routines: dict[str, Routine] = {}

    def add_event(self, e):
        if e.kind not in EVENT_KINDS or e.source not in EVENT_SOURCES:
            raise ValueError("bad event kind/source")
        self.events[e.id] = e
        return e.id

    def get_event(self, event_id):
        return self.events.get(event_id)

    def _filt(self, since=None, until=None, subject_id=None, kind=None, zone=None, profile=None):
        for e in self.events.values():
            if ((since is None or e.ts >= since) and (until is None or e.ts < until)
                    and (subject_id is None or e.subject_id == subject_id)
                    and (kind is None or e.kind == kind) and (zone is None or e.zone == zone)
                    and (profile is None or profile in e.profiles)):
                yield e

    def query(self, *, since=None, until=None, subject_id=None, kind=None, zone=None,
              profile=None, limit=100, newest_first=True):
        rows = sorted(self._filt(since, until, subject_id, kind, zone, profile),
                      key=lambda e: e.ts, reverse=newest_first)
        return rows[:limit]

    def last_event(self, subject_id, kind=None):
        r = self.query(subject_id=subject_id, kind=kind, limit=1)
        return r[0] if r else None

    def count(self, **filters):
        return sum(1 for _ in self._filt(**filters))

    def purge_before(self, ts):
        gone = [e for e in self.events.values() if e.ts < ts]
        for e in gone:
            del self.events[e.id]
        return sorted({p for e in gone for p in e.snapshot_paths})

    def forget_subject(self, subject_id):
        gone = [e for e in self.events.values() if e.subject_id == subject_id]
        for e in gone:
            del self.events[e.id]
        self.routines = {k: r for k, r in self.routines.items() if r.subject_id != subject_id}
        return sorted({p for e in gone for p in e.snapshot_paths})

    def delete_event(self, event_id):
        e = self.events.pop(event_id, None)
        return list(e.snapshot_paths) if e else []

    # minimal routine support so consumers can be tested against one object
    def upsert_routine(self, r):
        self.routines[r.id] = r
        return r.id

    def list_routines(self, subject_id=None, profile=None):
        return [r for r in self.routines.values()
                if (subject_id is None or r.subject_id == subject_id)
                and (profile is None or profile in r.profiles)]

    def delete_routine(self, routine_id):
        self.routines.pop(routine_id, None)

    def status(self, now, subject_id=None):
        return []   # real logic lives in brain/routines.py
