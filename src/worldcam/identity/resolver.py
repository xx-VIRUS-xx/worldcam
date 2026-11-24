"""Turn a track's faces into an Identity plus a decision the ambiguity engine
can act on. This module never creates Doubts; ambiguity/triggers.py maps
decisions to Doubts (see docs/AMBIGUITY_ENGINE.md, "Doubts").

Decision rules (scores are cosine similarity against face-cluster centroids,
taken BEFORE the new face is added, so it can't vote for itself):

  CONFLICT         two different labeled persons both >= ask_t and within
                   conflict_margin of each other; or a re-verification of a
                   track that was MATCHED to person A now matches person B.
  MATCHED          best labeled cluster >= match_t.
  UNKNOWN_CLUSTER  best cluster >= ask_t is unlabeled (and no labeled cluster
                   reached match_t). cluster_id/cluster_size describe the
                   cluster the face was stored in, which is what the engine
                   counts sightings on (unknown_face needs N=3).
  LOW_CONFIDENCE   best labeled cluster in [ask_t, match_t).
  NEW_FACE         nothing >= ask_t; the face starts (or joins) a fresh cluster.
  NO_FACE          no usable face on this track (not cached for good; the
                   next call with a face resolves it).

Per-track cache: a track is resolved once, then returned from cache until
re-verification: every `reverify_s` seconds of track time, or sooner if the
cached decision was not MATCHED and a face of clearly better quality arrives
(+upgrade_margin). Call `forget(track_id)` when a track ends and
`invalidate(cluster_id=...)` after the owner labels a cluster.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from ..graph.api import FaceMatch, GraphStore, Sighting, new_id
from ..pipeline import FaceObservation, Identity, Track

UNKNOWN = "unknown"


class Decision(str, Enum):
    MATCHED = "matched"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_CLUSTER = "unknown_cluster"
    NEW_FACE = "new_face"
    CONFLICT = "conflict"
    NO_FACE = "no_face"


@dataclass(frozen=True)
class Candidate:
    person_id: Optional[str]      # None = unlabeled cluster
    cluster_id: str
    score: float                  # cosine to the cluster centroid
    name: Optional[str] = None


@dataclass(frozen=True)
class ResolveResult:
    identity: Identity
    decision: Decision
    person_id: Optional[str] = None           # MATCHED only
    cluster_id: Optional[str] = None          # cluster the face was stored in
    cluster_size: int = 0                     # after storing this face
    face_id: Optional[str] = None
    sighting_id: Optional[str] = None
    candidates: tuple[Candidate, ...] = ()    # labeled: best per person; then unlabeled; score desc
    crop_path: Optional[str] = None
    quality: float = 0.0
    ts: float = 0.0
    cached: bool = False                      # True when served from the per-track cache
    detail: dict = field(default_factory=dict)  # e.g. {"previous_person_id": ...} on a re-verify conflict


@dataclass
class _Entry:
    result: ResolveResult
    resolved_ts: float


class IdentityResolver:
    def __init__(self, store: GraphStore, match_t: float = 0.45, ask_t: float = 0.35, *,
                 conflict_margin: float = 0.05, min_quality: float = 0.0, k: int = 5,
                 reverify_s: float = 600.0, upgrade_margin: float = 0.2,
                 camera: str = "default", max_cache: int = 2048):
        if not 0.0 <= ask_t <= match_t <= 1.0:
            raise ValueError("need 0 <= ask_t <= match_t <= 1")
        self.store = store
        self.match_t, self.ask_t = match_t, ask_t
        self.conflict_margin = conflict_margin
        self.min_quality = min_quality
        self.k = k
        self.reverify_s = reverify_s
        self.upgrade_margin = upgrade_margin
        self.camera = camera
        self.max_cache = max_cache
        self._cache: OrderedDict[int, _Entry] = OrderedDict()
        self._lock = threading.RLock()

    # ---- cache management --------------------------------------------------

    def cached(self, track_id: int) -> Optional[ResolveResult]:
        with self._lock:
            e = self._cache.get(track_id)
            return e.result if e else None

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._cache.pop(track_id, None)

    def invalidate(self, *, cluster_id: Optional[str] = None, person_id: Optional[str] = None) -> int:
        """Drop cached results that mention a cluster/person (e.g. after the owner
        labels a cluster or merges people), so the next call re-resolves."""
        with self._lock:
            drop = [tid for tid, e in self._cache.items()
                    if (cluster_id and (e.result.cluster_id == cluster_id
                                        or any(c.cluster_id == cluster_id for c in e.result.candidates)))
                    or (person_id and (e.result.person_id == person_id
                                       or any(c.person_id == person_id for c in e.result.candidates)))]
            for tid in drop:
                del self._cache[tid]
            return len(drop)

    def _needs_resolve(self, prev: Optional[_Entry], face: Optional[FaceObservation], ts: float) -> bool:
        if prev is None:
            return True
        if face is None:
            return False                       # nothing new to look at
        r = prev.result
        if r.decision is Decision.NO_FACE:
            return True
        if ts - prev.resolved_ts >= self.reverify_s:
            return True
        if r.decision is not Decision.MATCHED and face.quality >= r.quality + self.upgrade_margin:
            return True
        return False

    # ---- main entry --------------------------------------------------------

    def resolve(self, track: Track, faces: Sequence[FaceObservation]) -> ResolveResult:
        usable = [f for f in faces if f.quality >= self.min_quality and len(f.embedding) > 0]
        face = max(usable, key=lambda f: f.quality) if usable else None
        ts = face.ts if face else (track.best.ts if track.best else track.last_ts)

        with self._lock:
            prev = self._cache.get(track.track_id)
            if not self._needs_resolve(prev, face, ts):
                self._cache.move_to_end(track.track_id)
                return _replace(prev.result, cached=True)

            if face is None:
                res = ResolveResult(Identity(track.track_id, UNKNOWN, 0.0), Decision.NO_FACE, ts=ts)
            else:
                prev_match = prev.result if prev and prev.result.decision is Decision.MATCHED else None
                res = self._resolve_face(track, face, prev_match)
            self._cache[track.track_id] = _Entry(res, ts)
            self._cache.move_to_end(track.track_id)
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
            return res

    # ---- internals ---------------------------------------------------------

    def _candidates(self, matches: Sequence[FaceMatch]) -> tuple[Candidate, ...]:
        best_by_person: dict[str, FaceMatch] = {}
        unlabeled: list[FaceMatch] = []
        for m in matches:
            if m.person_id is None:
                unlabeled.append(m)
            elif m.person_id not in best_by_person or m.score > best_by_person[m.person_id].score:
                best_by_person[m.person_id] = m
        out = [Candidate(m.person_id, m.cluster_id, float(m.score), self._name(m.person_id))
               for m in best_by_person.values()]
        out += [Candidate(None, m.cluster_id, float(m.score)) for m in unlabeled]
        return tuple(sorted(out, key=lambda c: c.score, reverse=True))

    def _name(self, person_id: Optional[str]) -> Optional[str]:
        if person_id is None:
            return None
        p = self.store.get_person(person_id)
        return p.name if p is not None else None

    def _resolve_face(self, track: Track, face: FaceObservation,
                      prev_match: Optional[ResolveResult] = None) -> ResolveResult:
        matches = self.store.match_face(face.embedding, k=self.k)
        cands = self._candidates(matches)
        detail: dict = {}
        labeled = [c for c in cands if c.person_id is not None]
        top = cands[0] if cands else None
        top_l = labeled[0] if labeled else None
        second_l = labeled[1] if len(labeled) > 1 else None

        force_cluster = None
        if (top_l and second_l and top_l.score >= self.ask_t and second_l.score >= self.ask_t
                and top_l.score - second_l.score <= self.conflict_margin):
            decision = Decision.CONFLICT
            detail = {"reason": "two_persons_close"}
        elif top_l and top_l.score >= self.match_t:
            decision = Decision.MATCHED
            force_cluster = top_l.cluster_id     # keep the person's cluster growing
        elif top and top.person_id is None and top.score >= self.ask_t:
            decision = Decision.UNKNOWN_CLUSTER
        elif top_l and top_l.score >= self.ask_t:
            decision = Decision.LOW_CONFIDENCE
        else:
            decision = Decision.NEW_FACE

        if (decision is Decision.MATCHED and prev_match is not None
                and prev_match.person_id != top_l.person_id):
            # re-verification: same track, a different confident identity -> ask, don't flip
            decision, force_cluster = Decision.CONFLICT, None
            detail = {"reason": "track_identity_changed",
                      "previous_person_id": prev_match.person_id,
                      "previous_crop_path": prev_match.crop_path}
            if all(c.person_id != prev_match.person_id for c in cands):
                prev_c = next((c for c in prev_match.candidates if c.person_id == prev_match.person_id), None)
                if prev_c:
                    cands = tuple(sorted(cands + (prev_c,), key=lambda c: c.score, reverse=True))

        face_id, cluster_id = self.store.add_face(
            face.embedding, quality=face.quality, ts=face.ts, snapshot_path=face.crop_path,
            cluster_id=force_cluster, new_cluster=decision is Decision.CONFLICT)
        size = self.store.cluster_size(cluster_id)
        person_id = top_l.person_id if decision is Decision.MATCHED else None
        sighting_id = self.store.add_sighting(Sighting(
            id=new_id("s"), ts=face.ts, camera=self.camera, track_id=track.track_id,
            person_id=person_id, cluster_id=cluster_id, zones=tuple(track.zones),
            snapshot_path=face.crop_path))

        if decision is Decision.MATCHED:
            ident = Identity(track.track_id, top_l.name or top_l.person_id, round(top_l.score, 4))
        else:
            ident = Identity(track.track_id, UNKNOWN, round(top.score, 4) if top else 0.0)
        return ResolveResult(ident, decision, person_id=person_id, cluster_id=cluster_id,
                             cluster_size=size, face_id=face_id, sighting_id=sighting_id,
                             candidates=cands, crop_path=face.crop_path, quality=face.quality,
                             ts=face.ts, detail=detail)


def _replace(r: ResolveResult, **kw) -> ResolveResult:
    from dataclasses import replace
    return replace(r, **kw)
