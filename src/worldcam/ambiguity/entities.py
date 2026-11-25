"""Non-face entity observer: recognise animals and objects before asking about them.

Person tracks are handled by FaceAnalyzer + IdentityResolver (ArcFace embeddings).
Animals and objects use appearance embeddings (perception/reid.py) and the same
decision structure as faces, scoped by YOLO label (a cup never matches a dog):

  MATCHED        best labeled entity cluster >= match_t  -> Sighting of the KNOWN entity, no doubt
  CONFLICT       two labeled entities both >= ask_t within `conflict_margin` of each other
                 -> `conflict` doubt (e.g. two similar white mugs); the view starts its own cluster
  LOW_CONFIDENCE best labeled in [ask_t, match_t)        -> `low_confidence` doubt naming the candidate
  UNKNOWN        matches an unlabeled cluster (>= cluster_t) -> joins it; its ONE doubt already
                 exists or accumulates sightings (`unknown_entity` after entity_min_sightings)
  NEW            nothing close                             -> new cluster

One doubt per entity cluster (never per label+zone). Doubts are counted on the cluster's
own sightings, and the triggers dedup on the cluster id, so an answered or ignored entity is
not asked about again. Stationary repeats are rate-limited by `cooldown_s` per track and per cluster.

Data written per recorded observation: an EntityView (embedding, crop path) in the cluster and a
Sighting(cluster_id=..., entity_label=...); labeling the cluster to a Person back-fills sightings.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

from ..graph.api import GraphStore, Sighting, new_id
from ..pipeline import Track
from .triggers import DoubtDetector, IdentitySignal

log = logging.getLogger(__name__)

ANIMAL_LABELS = frozenset({
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
})

# Defaults are deliberately few: things that matter to an owner and that stay recognisable.
# Cups, bottles, books and phones are opt-in (`--entity-labels cup,bottle,...`): many look alike
# and the agent would ask about each of them.
DEFAULT_ENTITY_LABELS: tuple[str, ...] = (
    "cat", "dog", "backpack", "handbag", "suitcase", "laptop",
)
OPT_IN_ENTITY_LABELS: tuple[str, ...] = ("cup", "bottle", "book", "cell phone")

_SAFE_LABEL_RE = re.compile(r"[^a-z0-9_]+")


def classify_label(label: str) -> str:
    low = (label or "").strip().lower()
    if low == "person":
        return "person"
    if low in ANIMAL_LABELS:
        return "animal"
    return "object" if low else "other"


def _slug(s: str) -> str:
    return _SAFE_LABEL_RE.sub("_", (s or "").strip().lower()).strip("_") or "entity"


def make_entity_key(entity_type: str, label: str, zones: Sequence[str] = ()) -> str:
    """Legacy label+zone key (kept for callers that have no embedder); the observer
    itself now dedups on the entity cluster id."""
    z = "_".join(_slug(x) for x in zones if x)
    base = f"ek_{_slug(entity_type)}_{_slug(label)}"
    return f"{base}_{z}"[:64] if z else base[:64]


class EntityObserver:
    """Thread-safe observer for confirmed non-person tracks (animals, objects, etc.)."""

    def __init__(
        self,
        store: GraphStore,
        detector: Optional[DoubtDetector],
        *,
        embedder=None,                    # EntityEmbedder-like: .embed(bgr) -> unit vector, .thresholds
        thresholds=None,                  # ReidThresholds override
        snapshot_dir: Optional[str] = None,
        camera: str = "default",
        labels: Optional[Sequence[str]] = DEFAULT_ENTITY_LABELS,
        cooldown_s: float = 300.0,
        save_crops: bool = True,
        max_views_per_cluster: int = 30,
    ):
        self.store = store
        self.detector = detector
        self._embedder = embedder
        self._thresholds = thresholds
        self.snapshot_dir = snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR", "captures/snapshots")
        self.camera = camera
        self.labels = None if labels is None else {x.strip().lower() for x in labels if x.strip()}
        self.cooldown_s = cooldown_s
        self.save_crops = save_crops
        self.max_views = max_views_per_cluster
        self.conflict_margin = 0.03
        self._done_seq: dict[int, int] = {}
        self._track_last: dict[int, float] = {}
        self._cluster_last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._resolve_lock = threading.Lock()     # serialises match -> add so clusters are not duplicated
        self.stats = {"submitted": 0, "stored": 0, "doubts": 0, "cooldown_skips": 0,
                      "matched": 0, "new_cluster": 0, "embed_errors": 0}

    # ---------------------------------------------------------------- plumbing
    @property
    def embedder(self):
        if self._embedder is None:
            from ..perception.reid import EntityEmbedder
            self._embedder = EntityEmbedder("auto")
        return self._embedder

    @property
    def th(self):
        return self._thresholds or self.embedder.thresholds

    # ------------------------------------------------------------------ public
    def observe(self, track: Track, *, ended: bool = False) -> Optional[str]:
        """Inspect a confirmed Track; returns the created Doubt id (if any) or None."""
        label = (track.label or "").strip().lower()
        tid = track.track_id
        if label == "person" or (self.labels is not None and label not in self.labels):
            if ended:
                self._forget_track(tid)
            return None
        best = track.best
        if best is None or best.crop is None:
            if ended:
                self._forget_track(tid)
            return None
        ts = float(best.ts or track.last_ts or time.time())
        with self._lock:
            if self._done_seq.get(tid) == best.seq:
                if ended:
                    self._forget_track(tid, locked=True)
                return None
            self._done_seq[tid] = best.seq
            self.stats["submitted"] += 1
            prev = self._track_last.get(tid)
            if prev is not None and ts - prev < self.cooldown_s:     # stationary repeat of this track
                self.stats["cooldown_skips"] += 1
                if ended:
                    self._forget_track(tid, locked=True)
                return None
            self._track_last[tid] = ts
            if ended:
                self._forget_track(tid, locked=True)
        try:
            return self._resolve(track, label, best, ts)
        except Exception:
            self.stats["embed_errors"] += 1
            log.exception("entity re-id failed for track %s", tid)
            return None

    # ----------------------------------------------------------------- resolve
    def _resolve(self, track: Track, label: str, best, ts: float) -> Optional[str]:
        entity_type = classify_label(label)
        emb = self.embedder.embed(best.crop)
        th = self.th
        with self._resolve_lock:
            matches = self.store.match_entity(label, emb, k=8)
            labeled: dict[str, tuple[str, float]] = {}      # person_id -> (cluster_id, score), best first
            unlabeled = None
            for m in matches:                               # already sorted by score desc
                if m.person_id:
                    labeled.setdefault(m.person_id, (m.cluster_id, m.score))
                elif unlabeled is None and m.score >= th.cluster_t:
                    unlabeled = m
            cands = sorted(((pid, cid, sc) for pid, (cid, sc) in labeled.items()), key=lambda x: -x[2])

            conflict = (len(cands) >= 2 and cands[0][2] >= th.ask_t and cands[1][2] >= th.ask_t
                        and cands[0][2] - cands[1][2] <= self.conflict_margin)
            if not conflict and cands and cands[0][2] >= th.match_t:
                kind, target = "matched", cands[0][1]
            else:
                kind = ("conflict" if conflict else
                        "low_confidence" if cands and cands[0][2] >= th.ask_t else "unknown_entity")
                target = unlabeled.cluster_id if unlabeled is not None else None
            new_cluster = target is None

            # cluster-level cooldown: do not record the same known/stationary entity again too soon
            if target is not None:
                with self._lock:
                    last = self._cluster_last.get(target)
                    if last is not None and ts - last < self.cooldown_s:
                        self.stats["cooldown_skips"] += 1
                        return None
            crop_path = None
            if kind == "matched" and self.store.entity_cluster_size(target) >= self.max_views:
                view_cluster = target                       # cluster is full: sighting only
            else:
                crop_path = self._save_crop(best.crop, label, track.track_id, ts) if self.save_crops else None
                _, view_cluster = self.store.add_entity_view(
                    label, emb, quality=float(best.score or 0.0), ts=ts, snapshot_path=crop_path,
                    cluster_id=target, new_cluster=new_cluster)
                if new_cluster:
                    self.stats["new_cluster"] += 1
            with self._lock:
                self._cluster_last[view_cluster] = ts
                if len(self._cluster_last) > 5000:
                    for k in sorted(self._cluster_last, key=self._cluster_last.get)[:1000]:
                        del self._cluster_last[k]
            sid = new_id("s")
            self.store.add_sighting(Sighting(
                id=sid, ts=ts, camera=self.camera, track_id=track.track_id, cluster_id=view_cluster,
                zones=tuple(track.zones), snapshot_path=crop_path, entity_label=label))
            self.stats["stored"] += 1
            if kind == "matched":
                self.stats["matched"] += 1
                return None                                 # a KNOWN entity: sighting only, no doubt

        if self.detector is None:
            return None
        sig = IdentitySignal(
            kind=kind, track_id=track.track_id, cluster_id=view_cluster, crop_path=crop_path, ts=ts,
            entity_type=entity_type, entity_label=label, entity_key=view_cluster, sighting_id=sid,
            zones=tuple(track.zones),
            candidates=tuple((pid, sc) for pid, _, sc in cands[:2]) if kind in ("low_confidence", "conflict") else (),
        )
        did = self.detector.observe(sig, now=ts)
        if did:
            self.stats["doubts"] += 1
        return did

    # ----------------------------------------------------------------- helpers
    def _forget_track(self, tid: int, locked: bool = False) -> None:
        if locked:
            self._done_seq.pop(tid, None)
            self._track_last.pop(tid, None)
            return
        with self._lock:
            self._done_seq.pop(tid, None)
            self._track_last.pop(tid, None)

    def _save_crop(self, crop_bgr, label: str, track_id: int, ts: float) -> Optional[str]:
        import cv2
        try:
            d = Path(self.snapshot_dir)
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"entity_{_slug(label)}_t{track_id}_{int(ts * 1000)}.jpg"
            if not cv2.imwrite(str(p), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                return None
            return str(p)
        except Exception:
            log.exception("could not save entity crop")
            return None
