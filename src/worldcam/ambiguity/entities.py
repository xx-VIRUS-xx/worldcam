"""Non-face entity observer: turns tracked animals and objects into Sightings + Doubts.

Person tracks are handled by FaceAnalyzer + IdentityResolver (ArcFace embeddings).
Animals (`cat`, `dog`, ...) and objects (`backpack`, `laptop`, `cell phone`, ...)
have no face embeddings, so this module:
  1. Classifies the YOLO class label into `entity_type` (`animal`, `object`, `other`).
  2. Saves the track's `BestSnapshot` crop under `snapshot_dir` (once per track).
  3. Writes a `Sighting(entity_label=..., zones=..., snapshot_path=...)` to `GraphStore`.
  4. Emits an `IdentitySignal(kind="unknown_entity", ...)` to `DoubtDetector`.

Stationary / repeated sightings are rate-limited in memory by `cooldown_s` per
`entity_key` so a bag sitting on a desk does not flood sightings or doubts.
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

# Notable movable/personal objects from DEFAULT_CLASSES (avoids spamming static furniture).
DEFAULT_ENTITY_LABELS: tuple[str, ...] = (
    "cat", "dog", "backpack", "handbag", "suitcase", "laptop", "cell phone", "book", "bottle", "cup",
)

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
        snapshot_dir: Optional[str] = None,
        camera: str = "default",
        labels: Optional[Sequence[str]] = DEFAULT_ENTITY_LABELS,
        cooldown_s: float = 300.0,
        save_crops: bool = True,
    ):
        self.store = store
        self.detector = detector
        self.snapshot_dir = snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR", "captures/snapshots")
        self.camera = camera
        self.labels = None if labels is None else {x.strip().lower() for x in labels if x.strip()}
        self.cooldown_s = cooldown_s
        self.save_crops = save_crops
        self._done_seq: dict[int, int] = {}
        self._last_seen_key: dict[str, float] = {}
        self._lock = threading.Lock()
        self.stats = {"submitted": 0, "stored": 0, "doubts": 0, "cooldown_skips": 0}

    def observe(self, track: Track, *, ended: bool = False) -> Optional[str]:
        """Inspect a confirmed Track; returns created Doubt id (if any) or None."""
        label = (track.label or "").strip().lower()
        if label == "person" or (self.labels is not None and label not in self.labels):
            if ended:
                with self._lock:
                    self._done_seq.pop(track.track_id, None)
            return None
        best = track.best
        if best is None or best.crop is None:
            if ended:
                with self._lock:
                    self._done_seq.pop(track.track_id, None)
            return None
        with self._lock:
            if self._done_seq.get(track.track_id) == best.seq:
                if ended:
                    self._done_seq.pop(track.track_id, None)
                return None
            self.stats["submitted"] += 1
            entity_type = classify_label(label)
            ekey = make_entity_key(entity_type, label, track.zones)
            ts = float(best.ts or track.last_ts or time.time())
            prev_ts = self._last_seen_key.get(ekey)
            if prev_ts is not None and ts - prev_ts < self.cooldown_s:
                self._done_seq[track.track_id] = best.seq
                if ended:
                    self._done_seq.pop(track.track_id, None)
                self.stats["cooldown_skips"] += 1
                return None
            self._done_seq[track.track_id] = best.seq
            self._last_seen_key[ekey] = ts
            if ended:
                self._done_seq.pop(track.track_id, None)

        crop_path = self._save_crop(best.crop, label, track.track_id, ts) if self.save_crops else None
        sid = new_id("s")
        self.store.add_sighting(Sighting(
            id=sid,
            ts=ts,
            camera=self.camera,
            track_id=track.track_id,
            zones=tuple(track.zones),
            snapshot_path=crop_path,
            entity_label=label,
        ))
        self.stats["stored"] += 1
        if self.detector is None:
            return None
        sig = IdentitySignal(
            kind="unknown_entity",
            track_id=track.track_id,
            crop_path=crop_path,
            ts=ts,
            entity_type=entity_type,
            entity_label=label,
            entity_key=ekey,
            sighting_id=sid,
            zones=tuple(track.zones),
        )
        did = self.detector.observe(sig, now=ts)
        if did:
            self.stats["doubts"] += 1
        return did

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
