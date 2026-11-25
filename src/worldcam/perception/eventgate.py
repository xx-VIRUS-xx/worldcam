"""ActivityEventGate: decides when the slow, costly VLM is worth calling.

Fires only on STATE CHANGES per track, never on a timer per frame:
  entered        a watched track is seen for the first time
  new_zone       its zone set gained a zone
  became_active  STATIONARY -> ACTIVE
  dwell          it crossed a dwell threshold (once per threshold)
  best_snapshot  it gained its best snapshot (first time)
  possible_floor PRIORITY: person box is wide + low in frame (cheap pre-trigger,
                 NOT a diagnosis; only for profiles in `priority_profiles`)

Cost control: per-track cooldown, a global per-hour budget and a per-profile
per-hour budget. Priority triggers skip the cooldown/normal budget but have their
own hard hourly ceiling. Clock is injectable. Every decision is kept in
`last_decision` and counted in `stats` so the cost is observable.
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from ..geometry import Box
from ..pipeline import Identity, Track, TrackState, TrackUpdate

HOUR = 3600.0


def posture(box: Box, frame_aspect: float = 16 / 9) -> str:
    """Crude posture from box shape: 'horizontal' | 'upright' | 'unknown'.
    `frame_aspect` = frame width/height, since boxes are normalized. A heuristic
    on a 2D box: a person seated facing sideways or partly out of frame can fool it."""
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return "unknown"
    ratio = (w * frame_aspect) / h          # pixel width / pixel height
    if ratio >= 1.2:
        return "horizontal"
    if ratio <= 0.8:
        return "upright"
    return "unknown"


def looks_on_floor(box: Box, frame_aspect: float = 16 / 9, min_foot_y: float = 0.6) -> bool:
    """Wide box whose bottom edge is low in the frame. A CHEAP pre-trigger only."""
    return posture(box, frame_aspect) == "horizontal" and box[3] >= min_foot_y


@dataclass(frozen=True)
class FireDecision:
    reasons: tuple[str, ...]
    track_ids: tuple[int, ...]
    priority: bool = False
    profiles: tuple[str, ...] = ()


@dataclass
class _TrackMemory:
    zones: tuple[str, ...] = ()
    state: Optional[TrackState] = None
    had_best: bool = False
    dwell_done: set = field(default_factory=set)
    last_fire: Optional[float] = None
    last_floor: Optional[float] = None


class _Window:
    """Sliding one-hour counter."""

    def __init__(self):
        self.t: deque[float] = deque()

    def count(self, now: float) -> int:
        while self.t and now - self.t[0] >= HOUR:
            self.t.popleft()
        return len(self.t)

    def add(self, now: float) -> None:
        self.t.append(now)


class ActivityEventGate:
    def __init__(
        self,
        profiles: Sequence = (),                 # agent.api.Profile-like: .name
        watch_labels: Iterable[str] = ("person",),
        global_budget_per_hour: int = 60,
        profile_budget_per_hour: Optional[dict[str, int]] = None,
        default_profile_budget: int = 40,
        priority_profiles: Iterable[str] = ("medical",),
        priority_budget_per_hour: int = 20,
        track_cooldown_s: float = 20.0,
        floor_cooldown_s: float = 60.0,
        dwell_thresholds_s: Sequence[float] = (30.0, 120.0, 600.0),
        frame_aspect: float = 16 / 9,
        clock: Callable[[], float] = time.time,
        max_tracks: int = 256,
    ):
        self.profile_names = tuple(p.name for p in profiles)
        self.watch_labels = frozenset(watch_labels)
        self.global_budget = global_budget_per_hour
        self.profile_budgets = dict(profile_budget_per_hour or {})
        self.default_profile_budget = default_profile_budget
        self.priority_profiles = tuple(n for n in priority_profiles if n in self.profile_names)
        self.priority_budget = priority_budget_per_hour
        self.cooldown = track_cooldown_s
        self.floor_cooldown = floor_cooldown_s
        self.dwell_thresholds = tuple(sorted(dwell_thresholds_s))
        self.frame_aspect = frame_aspect
        self.clock = clock
        self.max_tracks = max_tracks
        self._mem: "OrderedDict[int, _TrackMemory]" = OrderedDict()
        self._global = _Window()
        self._priority = _Window()
        self._per_profile = {n: _Window() for n in self.profile_names}
        self.last_decision: Optional[FireDecision] = None
        self.stats = {"fired": 0, "priority_fired": 0, "suppressed_cooldown": 0,
                      "suppressed_budget": 0, "triggers": 0}

    # -- budget introspection ------------------------------------------------
    def remaining(self) -> dict:
        now = self.clock()
        return {
            "global": self.global_budget - self._global.count(now),
            "priority": self.priority_budget - self._priority.count(now),
            "profiles": {n: self._limit(n) - w.count(now) for n, w in self._per_profile.items()},
        }

    def _limit(self, name: str) -> int:
        return self.profile_budgets.get(name, self.default_profile_budget)

    # -- main entry ----------------------------------------------------------
    def should_fire(self, update: TrackUpdate, ids: Sequence[Identity] = (), world: Optional[dict] = None) -> bool:
        now = self.clock()
        self.last_decision = None
        normal: list[tuple[str, int]] = []
        urgent: list[tuple[str, int]] = []
        for t in update.ended:
            self._mem.pop(t.track_id, None)
        for t in update.active:
            if t.label not in self.watch_labels:
                continue
            m = self._mem.get(t.track_id)
            fresh = m is None
            if fresh:
                m = self._mem[t.track_id] = _TrackMemory()
                normal.append(("entered", t.track_id))
            self._mem.move_to_end(t.track_id)
            if not fresh:
                if set(t.zones) - set(m.zones):
                    normal.append(("new_zone", t.track_id))
                if m.state == TrackState.STATIONARY and t.state == TrackState.ACTIVE:
                    normal.append(("became_active", t.track_id))
            dwell = t.last_ts - t.first_ts
            for thr in self.dwell_thresholds:
                if dwell >= thr and thr not in m.dwell_done:
                    m.dwell_done.add(thr)
                    normal.append((f"dwell_{int(thr)}s", t.track_id))
            if t.best is not None and not m.had_best:
                normal.append(("best_snapshot", t.track_id))
            if (self.priority_profiles and t.label == "person"
                    and looks_on_floor(t.box, self.frame_aspect)
                    and (m.last_floor is None or now - m.last_floor >= self.floor_cooldown)):
                m.last_floor = now
                urgent.append(("possible_floor", t.track_id))
            m.zones, m.state, m.had_best = tuple(t.zones), t.state, m.had_best or t.best is not None
        while len(self._mem) > self.max_tracks:
            self._mem.popitem(last=False)

        self.stats["triggers"] += len(normal) + len(urgent)
        if urgent and self._priority.count(now) < self.priority_budget:
            self._priority.add(now)
            self.stats["fired"] += 1
            self.stats["priority_fired"] += 1
            tids = tuple(sorted({i for _, i in urgent + normal}))
            for i in tids:
                if i in self._mem:
                    self._mem[i].last_fire = now
            self.last_decision = FireDecision(tuple(r for r, _ in urgent + normal), tids, True,
                                              self.priority_profiles)
            return True
        if not normal:
            return False
        # cooldown: keep only triggers whose track is not cooling down
        ready = [(r, i) for r, i in normal
                 if self._mem.get(i) is None or self._mem[i].last_fire is None
                 or now - self._mem[i].last_fire >= self.cooldown]
        if not ready:
            self.stats["suppressed_cooldown"] += 1
            return False
        if self._global.count(now) >= self.global_budget or any(
                w.count(now) >= self._limit(n) for n, w in self._per_profile.items()):
            self.stats["suppressed_budget"] += 1
            return False
        self._global.add(now)
        for w in self._per_profile.values():
            w.add(now)
        tids = tuple(sorted({i for _, i in ready}))
        for i in tids:
            if i in self._mem:
                self._mem[i].last_fire = now
        self.stats["fired"] += 1
        self.last_decision = FireDecision(tuple(r for r, _ in ready), tids, False, self.profile_names)
        return True
