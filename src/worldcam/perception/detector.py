"""YOLO detector (Ultralytics yolo11n) on the SUBstream frame.

Returns pipeline.Detection with NORMALIZED boxes. ultralytics is an optional
extra (`pip install -e '.[perception]'`) and is imported lazily.

Region handling (`region_mode`), when the motion gate passes regions:
  "filter"  run on the whole sub frame, drop detections that don't overlap a
            region (boxes.overlaps_any). Default: see docs/perception.md for
            the measurement that picked it.
  "crop"    run on the union of the regions only, at an imgsz matched to the
            crop size (no upscaling), and map boxes back.
  "full"    ignore regions (detections everywhere).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..geometry import Box
from ..pipeline import Detection
from ..types import Frame
from .boxes import clamp_box, overlaps_any, union_box

# person + a short list of household objects (COCO names).
DEFAULT_CLASSES: tuple[str, ...] = (
    "person", "cat", "dog", "backpack", "handbag", "bottle", "cup", "chair",
    "couch", "bed", "dining table", "tv", "laptop", "remote", "keyboard",
    "cell phone", "book",
)

DEFAULT_WEIGHTS = "yolo11n.pt"


def find_weights(name: str = DEFAULT_WEIGHTS) -> Optional[Path]:
    """Look for local weights without downloading: $WORLDCAM_YOLO_WEIGHTS,
    ./models/<name>, <repo>/models/<name>, ./<name>. None if not found."""
    env = os.environ.get("WORLDCAM_YOLO_WEIGHTS")
    cands = [Path(env)] if env else []
    repo = Path(__file__).resolve().parents[3]
    cands += [Path("models") / name, repo / "models" / name, Path(name)]
    for c in cands:
        if c.is_file():
            return c
    return None


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


class YoloDetector:
    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        classes: Optional[Sequence[str]] = DEFAULT_CLASSES,
        conf: float = 0.35,
        imgsz: int = 640,
        region_mode: str = "filter",
        min_region_ioa: float = 0.1,
        crop_min_imgsz: int = 160,
        warmup: bool = True,
    ):
        from ultralytics import YOLO  # optional extra

        if region_mode not in ("filter", "crop", "full"):
            raise ValueError(f"region_mode must be filter|crop|full, got {region_mode!r}")
        if weights is None:
            found = find_weights()
            # Not found locally: ultralytics downloads it into the CWD. Point it
            # at models/ (git-ignored) instead.
            if found is None:
                Path("models").mkdir(exist_ok=True)
                weights = str(Path("models") / DEFAULT_WEIGHTS)
            else:
                weights = str(found)
        self.weights = weights
        self.model = YOLO(weights)
        self.device = pick_device(device)
        self.conf = conf
        self.imgsz = imgsz
        self.region_mode = region_mode
        self.min_region_ioa = min_region_ioa
        self.crop_min_imgsz = crop_min_imgsz
        names: dict[int, str] = self.model.names  # id -> name
        if classes is None:
            self.class_ids = None
        else:
            by_name = {v: k for k, v in names.items()}
            unknown = [c for c in classes if c not in by_name]
            if unknown:
                raise ValueError(f"unknown class names for this model: {unknown}")
            self.class_ids = sorted(by_name[c] for c in classes)
        self.names = names
        self.last_mode = "full"
        if warmup:
            self._run(np.zeros((360, 640, 3), np.uint8), self.imgsz)

    def _run(self, image: np.ndarray, imgsz: int) -> list[tuple[str, float, Box]]:
        """Inference on one BGR image; boxes normalized to THAT image."""
        res = self.model.predict(
            image, imgsz=imgsz, conf=self.conf, classes=self.class_ids,
            device=self.device, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxyn = res.boxes.xyxyn.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [(self.names[int(c)], float(p), clamp_box(tuple(b)))  # type: ignore[arg-type]
                for b, p, c in zip(xyxyn, confs, cls)]

    def detect(self, frame: Frame, regions: Optional[Sequence[Box]] = None) -> Sequence[Detection]:
        img = frame.image
        full = regions is None or self.region_mode == "full" or any(
            r[0] <= 0 and r[1] <= 0 and r[2] >= 1 and r[3] >= 1 for r in regions)
        if regions is not None and len(regions) == 0:
            self.last_mode = "skip"
            return []
        if full:
            self.last_mode = "full"
            return [Detection(l, c, b) for l, c, b in self._run(img, self.imgsz)]

        if self.region_mode == "filter":
            self.last_mode = "filter"
            return [Detection(l, c, b) for l, c, b in self._run(img, self.imgsz)
                    if overlaps_any(b, regions, self.min_region_ioa)]

        # crop: one inference on the union of the regions, no upscaling.
        self.last_mode = "crop"
        h, w = img.shape[:2]
        u = union_box(regions)  # type: ignore[arg-type]
        px1, py1 = int(u[0] * w), int(u[1] * h)
        px2, py2 = max(px1 + 1, int(round(u[2] * w))), max(py1 + 1, int(round(u[3] * h)))
        crop = img[py1:py2, px1:px2]
        cw, ch = px2 - px1, py2 - py1
        sz = int(min(self.imgsz, max(self.crop_min_imgsz, -(-max(cw, ch) // 32) * 32)))
        out = []
        for l, c, b in self._run(crop, sz):
            nb = clamp_box(((px1 + b[0] * cw) / w, (py1 + b[1] * ch) / h,
                            (px1 + b[2] * cw) / w, (py1 + b[3] * ch) / h))
            out.append(Detection(l, c, nb))
        return out
