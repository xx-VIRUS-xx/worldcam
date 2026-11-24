"""Face detection + ArcFace embeddings on a track's best-snapshot crop.

FaceAnalyzer wraps InsightFace FaceAnalysis(buffalo_l): SCRFD-10G detection
(with 5 landmarks) and the w600k_r50 ArcFace recognizer (512-d). Only those two
modules are loaded (no 3D landmarks, no gender/age), which keeps load time and
RAM down on an 8 GB machine.

Provider choice (measured on an M2 Air, see docs/identity.md): CPU is the
default. CoreML gives identical embeddings (cosine 1.00000 vs CPU on all 6
sample faces) but was not reliably faster: under the load of 5 concurrent
agents it won 2 of 3 paired runs and lost 1, while loading 10-20x slower
(23-35 s vs 1-3 s). Identity runs once per track, so ~0.5-1 s per person crop
is fine. Re-run `python -m worldcam.identity.bench` on an idle machine; if
CoreML is clearly faster there, set WORLDCAM_FACE_PROVIDER=coreml (and call
`warmup()` at startup). A CoreML load failure falls back to CPU.

The crop passed to `analyze` is the BestSnapshot crop. Its pixels cover the
track box padded by `crop_pad` of its own size and clamped to the frame (the
same rule as geometry.crop_norm, which the dual grabber uses with pad=0.1).
Face boxes are mapped back through that region into normalized full-frame
coordinates.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..geometry import Box
from ..pipeline import FaceObservation, Track

log = logging.getLogger(__name__)

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}
DEFAULT_PROVIDER = "cpu"

# ArcFace works on 112x112 aligned crops; a face this size or larger loses nothing.
FULL_SIZE_PX = 112.0


# ---- pure helpers (no model needed; unit-tested directly) ------------------

def crop_region(box: Box, pad: float) -> Box:
    """Normalized frame region covered by a crop of `box` padded by `pad`
    (fraction of the box's own size) and clamped to the frame. Mirrors
    geometry.crop_norm."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    return (max(0.0, box[0] - pad * bw), max(0.0, box[1] - pad * bh),
            min(1.0, box[2] + pad * bw), min(1.0, box[3] + pad * bh))


def map_box_to_frame(face_px: Sequence[float], crop_shape: Sequence[int], region: Box) -> Box:
    """Face box in crop pixels (x1, y1, x2, y2) -> normalized full-frame box."""
    h, w = int(crop_shape[0]), int(crop_shape[1])
    x1 = min(max(float(face_px[0]), 0.0), w) / w
    y1 = min(max(float(face_px[1]), 0.0), h) / h
    x2 = min(max(float(face_px[2]), 0.0), w) / w
    y2 = min(max(float(face_px[3]), 0.0), h) / h
    rw, rh = region[2] - region[0], region[3] - region[1]
    return (region[0] + x1 * rw, region[1] + y1 * rh,
            region[0] + x2 * rw, region[1] + y2 * rh)


def l2_normalize(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("cannot normalize a zero/invalid embedding")
    return a / n


def frontal_score(kps: Optional[np.ndarray]) -> float:
    """0..1 from the 5 SCRFD landmarks (left eye, right eye, nose, left mouth,
    right mouth), after rotating the eye line level (roll is harmless: the
    aligner undoes it). Yaw = nose offset from the eye/mouth midline over the
    eye distance (~0 frontal, ~0.7 near profile). Pitch = nose height between
    the eye and mouth lines (~0.5 frontal, ~0.85 looking down)."""
    if kps is None or len(kps) < 5:
        return 0.5
    k = np.asarray(kps, dtype=np.float64)[:5]
    d = k[1] - k[0]
    eye_d = float(np.hypot(d[0], d[1]))
    if eye_d < 1e-6:
        return 0.0
    c, s = d[0] / eye_d, d[1] / eye_d
    k = (k - k[0]) @ np.array([[c, -s], [s, c]])     # rotate so the eyes are level
    le, re, nose, lm, rm = k
    yaw = abs(nose[0] - (le[0] + re[0] + lm[0] + rm[0]) / 4.0) / eye_d
    eye_y, mouth_y = (le[1] + re[1]) / 2.0, (lm[1] + rm[1]) / 2.0
    span = mouth_y - eye_y
    if span <= 1e-6:
        return 0.0
    pitch = abs((nose[1] - eye_y) / span - 0.55)
    yaw_s = min(1.0, max(0.0, 1.0 - max(0.0, yaw - 0.15) / 0.55))
    pitch_s = min(1.0, max(0.0, 1.0 - max(0.0, pitch - 0.15) / 0.4))
    return float(yaw_s * pitch_s)


def sharpness_score(face_bgr: np.ndarray) -> float:
    """0..1 from Laplacian std / intensity std on the face resized to 112 px:
    independent of the face's pixel size and of exposure (a dark but sharp
    face scores like a bright one). ~0.28 sharp, ~0.16 Gaussian k5, ~0.11 k9."""
    import cv2
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    g = cv2.resize(g, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float64)
    sd = float(g.std())
    if sd < 2.0:                      # flat patch: no face detail at all
        return 0.0
    r = float(cv2.Laplacian(g, cv2.CV_64F).std()) / sd
    return float(min(1.0, max(0.0, (r - 0.05) / 0.2)))


def quality_score(det_score: float, face_px: float, sharp: float, frontal: float) -> float:
    """Product of four 0..1 terms (FaceObservation.quality contract)."""
    det = min(1.0, max(0.0, (det_score - 0.5) / 0.3))   # 0.8+ counts as full
    size = min(1.0, max(0.0, face_px / FULL_SIZE_PX))
    return float(det * size * sharp * frontal)


# ---- analyzer --------------------------------------------------------------

@dataclass
class RawFace:
    """Model output before mapping, kept for diagnostics and the bench."""
    bbox: np.ndarray         # crop pixels
    kps: Optional[np.ndarray]
    det_score: float
    embedding: np.ndarray    # L2-normalized float64, 512


class FaceAnalyzer:
    def __init__(self, *, providers: Optional[str] = None, det_size: tuple[int, int] = (640, 640),
                 det_thresh: float = 0.5, min_face_px: int = 32, min_quality: float = 0.1,
                 crop_pad: float = 0.1, snapshot_dir: Optional[str] = None,
                 save_crops: bool = True, model_name: str = "buffalo_l",
                 model_root: str = "~/.insightface"):
        self.provider = (providers or os.environ.get("WORLDCAM_FACE_PROVIDER") or DEFAULT_PROVIDER).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"providers must be one of {sorted(PROVIDERS)}")
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.min_face_px = min_face_px
        self.min_quality = min_quality
        self.crop_pad = crop_pad
        self.snapshot_dir = snapshot_dir or os.environ.get("WORLDCAM_SNAPSHOT_DIR", "captures/snapshots")
        self.save_crops = save_crops
        self.model_name = model_name
        self.model_root = model_root
        self._app = None
        self._lock = threading.Lock()
        self.load_s: Optional[float] = None

    # lazy load: importing this module never touches insightface/onnxruntime
    def _get_app(self):
        if self._app is None:
            with self._lock:
                if self._app is None:
                    t0 = time.perf_counter()
                    try:
                        app = self._build(self.provider)
                    except Exception:
                        if self.provider == "cpu":
                            raise
                        log.exception("FaceAnalyzer: %s failed to load, falling back to CPU", self.provider)
                        self.provider = "cpu"
                        app = self._build("cpu")
                    self._app = app
                    self.load_s = time.perf_counter() - t0
                    log.info("FaceAnalyzer loaded %s on %s in %.1fs",
                             self.model_name, self.provider, self.load_s)
        return self._app

    def _build(self, provider: str):
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=self.model_name, root=self.model_root,
                           allowed_modules=["detection", "recognition"],
                           providers=PROVIDERS[provider])
        app.prepare(ctx_id=-1, det_thresh=self.det_thresh, det_size=self.det_size)
        return app

    def warmup(self) -> float:
        """Load the models and run one inference (CoreML compiles on first use).
        Call once at startup. Returns seconds spent."""
        t0 = time.perf_counter()
        app = self._get_app()
        app.get(np.zeros((256, 256, 3), np.uint8))                          # detector
        app.models["recognition"].get_feat(np.zeros((112, 112, 3), np.uint8))  # recognizer
        return time.perf_counter() - t0

    def detect(self, image_bgr: np.ndarray) -> list[RawFace]:
        """All faces in an image (crop pixels), embeddings L2-normalized."""
        out = []
        for f in self._get_app().get(image_bgr):
            emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            out.append(RawFace(np.asarray(f.bbox, dtype=np.float64),
                               None if f.kps is None else np.asarray(f.kps, dtype=np.float64),
                               float(f.det_score), l2_normalize(emb)))
        return out

    def embed_aligned(self, face112_bgr: np.ndarray) -> np.ndarray:
        """Embedding of an already-aligned 112x112 face (bench/tests only)."""
        rec = self._get_app().models["recognition"]
        return l2_normalize(rec.get_feat(face112_bgr).ravel())

    def analyze(self, crop_bgr: Optional[np.ndarray], track: Track,
                ts: Optional[float] = None) -> list[FaceObservation]:
        if crop_bgr is None and track.best is not None:
            crop_bgr = track.best.crop
        if crop_bgr is None or crop_bgr.size == 0:
            return []
        src_box = track.best.box if track.best is not None else track.box
        ts = float(ts if ts is not None else (track.best.ts if track.best is not None else track.last_ts))
        region = crop_region(src_box, self.crop_pad)
        h, w = crop_bgr.shape[:2]

        obs: list[FaceObservation] = []
        for i, f in enumerate(self.detect(crop_bgr)):
            x1, y1, x2, y2 = (int(max(0, f.bbox[0])), int(max(0, f.bbox[1])),
                              int(min(w, f.bbox[2])), int(min(h, f.bbox[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            face_px = float(min(x2 - x1, y2 - y1))
            if face_px < self.min_face_px:
                continue
            face_img = crop_bgr[y1:y2, x1:x2]
            q = quality_score(f.det_score, face_px, sharpness_score(face_img), frontal_score(f.kps))
            if q < self.min_quality:
                continue
            path = self._save(crop_bgr, (x1, y1, x2, y2), track.track_id, ts, i) if self.save_crops else None
            obs.append(FaceObservation(
                track_id=track.track_id, ts=ts,
                box=map_box_to_frame((x1, y1, x2, y2), crop_bgr.shape, region),
                embedding=tuple(float(x) for x in f.embedding),
                quality=round(q, 4), crop_path=path))
        obs.sort(key=lambda o: o.quality, reverse=True)
        return obs

    def _save(self, crop_bgr, px, track_id, ts, i) -> Optional[str]:
        """Save the face with 30% context for the owner. Failure is non-fatal."""
        import cv2
        try:
            x1, y1, x2, y2 = px
            h, w = crop_bgr.shape[:2]
            m = int(0.3 * max(x2 - x1, y2 - y1))
            img = crop_bgr[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
            d = Path(self.snapshot_dir)
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"face_t{track_id}_{int(ts * 1000)}_{i}.jpg"
            if not cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                return None
            return str(p)
        except Exception:  # disk full, permissions: identity still works
            log.exception("could not save face crop")
            return None
