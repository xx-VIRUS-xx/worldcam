"""EventRecorder: perception output -> debounced BrainEvents. See docs/brain.md.

Feed it TrackUpdates (`on_track_update`), identity results (`on_resolve`) and
entity sightings (`on_sighting`). It emits to an EventStore and returns the
events it created.

Debouncing rules:
- `entered` only once a track is confirmed (not NEW, alive >= confirm_s) and its
  identity is known OR identity_wait_s has passed; and only when the subject has
  no live presence (another track of the same person, or a `left` still in its
  grace period, means no new `entered`).
- `left` is delayed by leave_grace_s after the last track ends / is unseen for
  absence_s; a return inside the grace cancels it. Event ts = last time seen.
- `zone_changed` needs the new zone to persist zone_min_s.
- `stationary` once per stationary episode (after stationary_after_s).
- `seen` at most every seen_every_s while present; also on late identification.
- Hard cap: max_events_per_min per subject (sliding 60 s). Dropped events are counted.
Memory is bounded: tracks/presences are evicted on end/idle and capped at max_subjects.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from ..pipeline import Track, TrackState, TrackUpdate
from .api import BrainEvent, new_event

DEFAULT_PROFILE_MAP: Mapping[str, tuple[str, ...]] = {
    "person": ("cctv", "companion", "medical"),
    "dog": ("companion", "cctv"),
    "cat": ("companion", "cctv"),
}


@dataclass
class RecorderConfig:
    profile_map: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_PROFILE_MAP))
    confirm_s: float = 1.0
    identity_wait_s: float = 3.0
    absence_s: float = 30.0          # unseen this long => treated as ended
    leave_grace_s: float = 10.0
    zone_min_s: float = 2.0
    stationary_after_s: float = 60.0
    seen_every_s: float = 300.0
    max_events_per_min: int = 6
    max_subjects: int = 256
    max_tracks: int = 512
    unknown_label: str = "unknown person"


@dataclass
class _TrackRec:
    key: tuple                         # (camera, track_id)
    camera: str
    label: str
    first_ts: float
    last_ts: float
    zone: Optional[str] = None
    person_id: Optional[str] = None
    cluster_id: Optional[str] = None
    name: Optional[str] = None
    conf: float = 0.0
    crop_path: Optional[str] = None
    subject_key: Optional[str] = None  # set once attached to a presence
    pend_zone: Optional[str] = None
    pend_zone_since: float = 0.0
    stationary_emitted: bool = False


@dataclass
class _Presence:
    key: str
    label: str
    subject_id: Optional[str]
    subject_name: Optional[str]
    cluster_id: Optional[str]
    zone: Optional[str]
    camera: str
    last_ts: float
    last_seen_event_ts: float
    tracks: set = field(default_factory=set)
    left_deadline: Optional[float] = None
    left_ts: float = 0.0
    snapshot: Optional[str] = None
    conf: float = 0.0


class EventRecorder:
    def __init__(self, store, config: Optional[RecorderConfig] = None, *,
                 clock: Callable[[], float] = time.time, camera: str = "default",
                 on_event: Optional[Callable[[BrainEvent], None]] = None):
        self.store, self.cfg, self.clock, self.camera = store, config or RecorderConfig(), clock, camera
        self.on_event = on_event
        self._lock = threading.RLock()
        self._tracks: "OrderedDict[tuple, _TrackRec]" = OrderedDict()
        self._pres: "OrderedDict[str, _Presence]" = OrderedDict()
        self._rate: dict[str, deque] = {}
        self.dropped = 0
        self.emitted = 0

    # ------------------------------------------------------------ public API
    def on_track_update(self, update: TrackUpdate, ts: Optional[float] = None,
                        camera: Optional[str] = None) -> list[BrainEvent]:
        now = self.clock() if ts is None else ts
        cam = camera or self.camera
        out: list[BrainEvent] = []
        with self._lock:
            for t in update.active:
                if self._tracked(t.label):
                    self._on_active(t, now, cam, out)
            for t in update.ended:
                rec = self._tracks.pop((cam, t.track_id), None)
                if rec:
                    rec.last_ts = max(rec.last_ts, t.last_ts)
                    self._detach(rec, now)
            self._sweep(now, out)
        return out

    def on_resolve(self, result, track_id: Optional[int] = None, ts: Optional[float] = None,
                   camera: Optional[str] = None) -> list[BrainEvent]:
        """identity/resolver.py ResolveResult. `track_id` defaults to result.identity.track_id."""
        now = self.clock() if ts is None else ts
        cam = camera or self.camera
        tid = result.identity.track_id if track_id is None else track_id
        out: list[BrainEvent] = []
        with self._lock:
            rec = self._tracks.get((cam, tid))
            if rec is None:
                return out
            dec = getattr(result.decision, "value", result.decision)
            if result.person_id and dec in ("matched", "low_confidence"):
                rec.person_id, rec.name = result.person_id, result.identity.name
            elif result.cluster_id:
                rec.cluster_id = result.cluster_id
            rec.conf = max(rec.conf, float(result.identity.conf or 0.0))
            if result.crop_path:
                rec.crop_path = result.crop_path
            if rec.subject_key is not None:
                self._late_identity(rec, now, out)
            else:
                self._try_enter(rec, now, out)
        return out

    def on_sighting(self, subject_id: Optional[str], label: str, ts: Optional[float] = None, *,
                    zone: Optional[str] = None, camera: Optional[str] = None, confidence: float = 0.8,
                    snapshot: Optional[str] = None, name: Optional[str] = None,
                    cluster_id: Optional[str] = None, track_id: Optional[int] = None) -> list[BrainEvent]:
        """An entity re-id / graph sighting (animal, object, person) without a live track."""
        now = self.clock() if ts is None else ts
        cam = camera or self.camera
        out: list[BrainEvent] = []
        with self._lock:
            key = (cam, ("entity", subject_id or cluster_id or track_id or label))
            rec = self._tracks.get(key)
            if rec is None:
                if not self._tracked(label):
                    return out
                rec = self._new_rec(key, cam, label, now)
                rec.first_ts = now - self.cfg.confirm_s     # a sighting is already confirmed
            rec.last_ts, rec.zone = now, zone or rec.zone
            rec.person_id, rec.name, rec.cluster_id = subject_id, name, cluster_id or rec.cluster_id
            rec.conf, rec.crop_path = confidence, snapshot or rec.crop_path
            self._attach_or_enter(rec, now, out)
            self._seen_tick(rec, now, out)
            self._sweep(now, out)
        return out

    def flush(self, now: Optional[float] = None) -> list[BrainEvent]:
        """Run the time-based rules (leave grace, absence) without new input."""
        now = self.clock() if now is None else now
        out: list[BrainEvent] = []
        with self._lock:
            self._sweep(now, out)
        return out

    def stats(self) -> dict:
        with self._lock:
            return dict(tracks=len(self._tracks), presences=len(self._pres),
                        emitted=self.emitted, dropped=self.dropped)

    # --------------------------------------------------------------- internals
    def _tracked(self, label: str) -> bool:
        pm = self.cfg.profile_map
        return label in pm or "*" in pm

    def _profiles(self, label: str) -> tuple[str, ...]:
        pm = self.cfg.profile_map
        return tuple(pm.get(label, pm.get("*", ())))

    def _new_rec(self, key, cam, label, now) -> _TrackRec:
        rec = _TrackRec(key=key, camera=cam, label=label, first_ts=now, last_ts=now)
        self._tracks[key] = rec
        while len(self._tracks) > self.cfg.max_tracks:
            _, old = self._tracks.popitem(last=False)
            self._detach(old, now)
        return rec

    def _on_active(self, t: Track, now: float, cam: str, out: list) -> None:
        key = (cam, t.track_id)
        rec = self._tracks.get(key)
        if rec is None:
            rec = self._new_rec(key, cam, t.label, t.first_ts)
        else:
            self._tracks.move_to_end(key)
        rec.last_ts = max(rec.last_ts, t.last_ts, now)
        zone = t.zones[0] if t.zones else None
        if rec.subject_key is None:
            rec.zone = zone
            if t.state != TrackState.NEW and (rec.last_ts - rec.first_ts) >= self.cfg.confirm_s:
                self._try_enter(rec, now, out, force=(now - rec.first_ts) >= self.cfg.identity_wait_s)
            return
        p = self._pres.get(rec.subject_key)
        if p is None:
            return
        p.last_ts = now
        p.left_deadline = None
        self._pres.move_to_end(p.key)
        # zone change (debounced)
        if zone != p.zone:
            if rec.pend_zone != zone:
                rec.pend_zone, rec.pend_zone_since = zone, now
            elif now - rec.pend_zone_since >= self.cfg.zone_min_s:
                old, p.zone = p.zone, zone
                rec.zone = zone
                rec.pend_zone = None
                self._emit(out, p, "zone_changed", now, detail={"from_zone": old, "to_zone": zone,
                                                                  "track_id": t.track_id}, zone=zone)
        else:
            rec.pend_zone = None
        # stationary
        if t.state == TrackState.STATIONARY or t.stationary_since is not None:
            since = t.stationary_since if t.stationary_since is not None else now
            if not rec.stationary_emitted and now - since >= self.cfg.stationary_after_s:
                rec.stationary_emitted = True
                self._emit(out, p, "stationary", now, detail={"since": since, "track_id": t.track_id})
        else:
            rec.stationary_emitted = False
        self._seen_tick(rec, now, out)

    def _subject_key(self, rec: _TrackRec) -> str:
        if rec.person_id:
            return "p:" + rec.person_id
        if rec.cluster_id:
            return "c:" + rec.cluster_id
        return f"t:{rec.camera}:{rec.key[1]}"

    def _try_enter(self, rec: _TrackRec, now: float, out: list, force: bool = False) -> None:
        known = bool(rec.person_id or rec.cluster_id) or rec.label != "person"
        if known or force:
            self._attach_or_enter(rec, now, out)

    def _attach_or_enter(self, rec: _TrackRec, now: float, out: list) -> None:
        if rec.subject_key is not None:
            return
        key = self._subject_key(rec)
        p = self._pres.get(key)
        if p is None:
            p = _Presence(key=key, label=rec.label, subject_id=rec.person_id, subject_name=rec.name,
                          cluster_id=rec.cluster_id, zone=rec.zone, camera=rec.camera, last_ts=now,
                          last_seen_event_ts=now, snapshot=rec.crop_path, conf=rec.conf)
            self._pres[key] = p
            while len(self._pres) > self.cfg.max_subjects:
                _, old = self._pres.popitem(last=False)
                self._rate.pop(old.key, None)
            p.tracks.add(rec.key)
            rec.subject_key = key
            self._emit(out, p, "entered", now, detail={"track_id": rec.key[1]})
        else:
            p.tracks.add(rec.key)
            rec.subject_key = key
            p.left_deadline = None       # came back inside the grace period: no left, no entered
            p.last_ts = now
            if rec.crop_path:
                p.snapshot = rec.crop_path
            self._pres.move_to_end(key)

    def _late_identity(self, rec: _TrackRec, now: float, out: list) -> None:
        """Identity arrived after `entered`. Re-key the presence when the subject changed."""
        old = self._pres.get(rec.subject_key)
        new_key = self._subject_key(rec)
        if old is None or new_key == old.key:
            if old:
                old.snapshot = rec.crop_path or old.snapshot
            return
        old.tracks.discard(rec.key)
        if not old.tracks:
            self._pres.pop(old.key, None)
        rec.subject_key = None
        p = self._pres.get(new_key)
        if p is None:
            p = _Presence(key=new_key, label=rec.label, subject_id=rec.person_id, subject_name=rec.name,
                          cluster_id=rec.cluster_id, zone=rec.zone or old.zone, camera=rec.camera,
                          last_ts=now, last_seen_event_ts=now, snapshot=rec.crop_path or old.snapshot,
                          conf=rec.conf)
            self._pres[new_key] = p
            p.tracks.add(rec.key)
            rec.subject_key = new_key
            self._emit(out, p, "seen", now, detail={"track_id": rec.key[1], "identified_late": True,
                                                    "previous_key": old.key})
        else:
            p.tracks.add(rec.key)
            rec.subject_key = new_key

    def _seen_tick(self, rec: _TrackRec, now: float, out: list) -> None:
        p = self._pres.get(rec.subject_key or "")
        if p and now - p.last_seen_event_ts >= self.cfg.seen_every_s:
            self._emit(out, p, "seen", now, detail={"track_id": rec.key[1]})

    def _detach(self, rec: _TrackRec, now: float) -> None:
        p = self._pres.get(rec.subject_key or "")
        if not p:
            return
        p.tracks.discard(rec.key)
        if not p.tracks:
            p.left_ts = rec.last_ts
            p.left_deadline = rec.last_ts + self.cfg.leave_grace_s

    def _sweep(self, now: float, out: list) -> None:
        for key, rec in list(self._tracks.items()):
            if now - rec.last_ts > self.cfg.absence_s:
                del self._tracks[key]
                self._detach(rec, now)
        for key, p in list(self._pres.items()):
            if p.left_deadline is not None and now >= p.left_deadline:
                self._emit(out, p, "left", p.left_ts)
                del self._pres[key]
                if len(self._rate.get(key, ())) == 0:
                    self._rate.pop(key, None)

    def _emit(self, out: list, p: _Presence, kind: str, ts: float, *, detail: Optional[dict] = None,
              zone: Optional[str] = None) -> None:
        rate = self._rate.setdefault(p.key, deque(maxlen=max(1, self.cfg.max_events_per_min)))
        wall = ts
        while rate and wall - rate[0] >= 60.0:
            rate.popleft()
        if len(rate) >= self.cfg.max_events_per_min:
            self.dropped += 1
            return
        rate.append(wall)
        d = {k: v for k, v in (detail or {}).items() if v is not None}
        if p.cluster_id:
            d["cluster_id"] = p.cluster_id
        if p.subject_id is None:
            d["unidentified"] = True
        label = p.subject_name or (self.cfg.unknown_label if p.label == "person" and not p.subject_id
                                   else p.label)
        ev = new_event(kind, ts=ts, subject_id=p.subject_id, subject_label=label,
                       zone=zone if zone is not None else p.zone, camera=p.camera,
                       confidence=round(p.conf, 3), source="camera",
                       snapshot_paths=(p.snapshot,) if p.snapshot else (),
                       profiles=self._profiles(p.label), detail=d)
        self.store.add_event(ev)
        p.last_seen_event_ts = max(p.last_seen_event_ts, ts)
        self.emitted += 1
        out.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass
