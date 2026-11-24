"""DoubtDetector: turns identity decisions into Doubts (rules from AMBIGUITY_ENGINE.md).

Input contract (`IdentitySignal`). The identity resolver is being built in
parallel, so this module owns a small local type; the coordinator maps the
resolver's output onto it:

  kind            meaning                                         required fields
  --------------  ----------------------------------------------  ---------------------------
  matched         best match >= MATCH_T, cluster labeled          person_id (or candidates[0])
  low_confidence  best match in [ASK_T, MATCH_T)                  cluster_id, candidates
  unknown_cluster face joined an existing, unlabeled cluster      cluster_id
  new_face        face started a new cluster (first sighting)     cluster_id
  conflict        one track got two identities, or one cluster    candidates (>= 2 person ids),
                  matches two persons                             cluster_id optional

  track_id, crop_path and ts are always filled when available. candidates is
  [(person_id, cosine score)] best first. The resolver must have called
  GraphStore.add_sighting() for this observation *before* the signal is passed
  in, because sighting counts are read from the store (so the detector is
  stateless and restart-safe).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from worldcam.graph.api import Doubt, GraphStore, new_id

SIGNAL_KINDS = ("matched", "low_confidence", "unknown_cluster", "new_face", "conflict")
ACTIVE_STATUSES = ("open", "asked", "answered")


@dataclass(frozen=True)
class IdentitySignal:
    kind: str                                   # one of SIGNAL_KINDS
    track_id: Optional[int] = None
    cluster_id: Optional[str] = None
    candidates: tuple[tuple[str, float], ...] = ()   # (person_id, score), best first
    crop_path: Optional[str] = None
    ts: float = 0.0
    person_id: Optional[str] = None             # set for `matched` (else candidates[0])

    def __post_init__(self):
        if self.kind not in SIGNAL_KINDS:
            raise ValueError(f"bad signal kind {self.kind!r}")


@dataclass
class TriggerConfig:
    unknown_min_sightings: int = 3     # N
    metadata_min_sightings: int = 5    # K
    match_t: float = 0.45
    ask_t: float = 0.35
    ignore_cooldown_days: float = 30.0  # owner tapped Ignore: don't re-ask this cluster for a while
    required_facts: tuple[str, ...] = ("relation_to_owner",)


class DoubtDetector:
    def __init__(self, store: GraphStore, config: Optional[TriggerConfig] = None):
        self.store = store
        self.cfg = config or TriggerConfig()

    # -- public -------------------------------------------------------------
    def observe(self, sig: IdentitySignal, now: Optional[float] = None) -> Optional[str]:
        """Return the id of a newly created Doubt, or None."""
        now = time.time() if now is None else now
        if sig.kind in ("new_face", "unknown_cluster"):
            return self._unknown(sig, now)
        if sig.kind == "low_confidence":
            return self._low_conf(sig, now)
        if sig.kind == "conflict":
            return self._conflict(sig, now)
        return self._metadata_gap(sig, now)

    # -- rules --------------------------------------------------------------
    def _unknown(self, sig, now):
        cid = sig.cluster_id
        if not cid or self._blocked(cid, now):
            return None
        n = self._count(cluster_id=cid)
        if n < self.cfg.unknown_min_sightings:
            return None
        return self._create("unknown_face", (cid,), sig, now, {"sightings": n})

    def _low_conf(self, sig, now):
        if not sig.candidates:
            return None
        pid, score = sig.candidates[0]
        if not (self.cfg.ask_t <= score < self.cfg.match_t):
            return None
        key = sig.cluster_id or pid
        if self._blocked(key, now):
            return None
        about = tuple(x for x in (sig.cluster_id, pid) if x)
        return self._create("low_confidence", about, sig, now,
                            {"candidate_person_id": pid, "score": round(score, 4)})

    def _conflict(self, sig, now):
        pids = tuple(dict.fromkeys(p for p, _ in sig.candidates))
        if len(pids) < 2:
            return None
        about = tuple(x for x in (sig.cluster_id,) if x) + pids
        if any(self._blocked(x, now) for x in about if x == sig.cluster_id) or self._same_conflict_open(pids):
            return None
        return self._create("conflict", about, sig, now,
                            {"candidates": [[p, round(s, 4)] for p, s in sig.candidates]})

    def _metadata_gap(self, sig, now):
        pid = sig.person_id or (sig.candidates[0][0] if sig.candidates else None)
        if not pid or self._blocked(pid, now):
            return None
        p = self.store.get_person(pid)
        if p is None or p.kind == "owner":
            return None
        if self._count(person_id=pid) < self.cfg.metadata_min_sightings:
            return None
        missing = []
        if not p.name:
            missing.append("name")
        keys = {f.key for f in self.store.facts(pid)}
        has_rel = bool(self.store.relations(pid))
        for k in self.cfg.required_facts:
            if k not in keys and not (k == "relation_to_owner" and has_rel):
                missing.append(k)
        if not missing:
            return None
        return self._create("metadata_gap", (pid,), sig, now, {"missing": missing})

    # -- helpers ------------------------------------------------------------
    def _count(self, **kw) -> int:
        need = max(self.cfg.unknown_min_sightings, self.cfg.metadata_min_sightings)
        return len(self.store.recent_sightings(limit=need, **kw))

    def _doubts(self, statuses: Sequence[str]) -> list[Doubt]:
        out: list[Doubt] = []
        for st in statuses:
            out.extend(self.store.list_doubts(st, limit=10_000))
        return out

    def _blocked(self, key: str, now: float) -> bool:
        """Dedup: an active doubt about `key`, or one the owner ignored recently."""
        if any(key in d.about_ids for d in self._doubts(ACTIVE_STATUSES)):
            return True
        cool = self.cfg.ignore_cooldown_days * 86400
        return any(key in d.about_ids and d.detail.get("ignored_by_owner")
                   and now - d.detail.get("dismissed_at", d.updated) < cool
                   for d in self._doubts(("dismissed",)))

    def _same_conflict_open(self, pids) -> bool:
        s = set(pids)
        return any(d.kind == "conflict" and s <= set(d.about_ids) for d in self._doubts(ACTIVE_STATUSES))

    def _create(self, kind, about, sig, now, detail) -> str:
        detail = dict(detail, signal_kind=sig.kind, track_id=sig.track_id)
        snaps = (sig.crop_path,) if sig.crop_path else ()
        return self.store.create_doubt(Doubt(id=new_id("d"), kind=kind, status="open",
                                             about_ids=tuple(about), snapshot_paths=snaps,
                                             created=now, detail=detail))
