"""Cheap motion gate: frame differencing against a running-average background.

Runs on a tiny grayscale copy of the SUBstream frame (default 160 px wide), so
it costs well under a millisecond-ish on an M2 and decides whether the YOLO
detector runs at all. Output boxes are NORMALIZED (0..1).

Behaviour:
- first frame               -> whole frame (nothing is known yet)
- every `full_every_s`      -> whole frame, so stationary people/objects are
                               still re-checked by the detector
- motion over > `global_frac` of the frame (light switch, auto-exposure jump)
                            -> whole frame
- otherwise                 -> padded, merged bounding boxes of the blobs whose
                               area >= `min_area` (fraction of the frame); empty
                               list = no motion = skip the detector.

The clock is Frame.ts (time.monotonic at capture), which keeps it
deterministic for tests and immune to processing jitter.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ..geometry import Box
from ..types import Frame

FULL_FRAME: Box = (0.0, 0.0, 1.0, 1.0)


def _overlap_or_near(a: Box, b: Box, gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def merge_boxes(boxes: Sequence[Box], gap: float = 0.0) -> list[Box]:
    """Union boxes that overlap (or are within `gap`) until none do."""
    out = [tuple(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        res: list[Box] = []
        for b in out:
            for i, r in enumerate(res):
                if _overlap_or_near(b, r, gap):
                    res[i] = (min(b[0], r[0]), min(b[1], r[1]), max(b[2], r[2]), max(b[3], r[3]))
                    merged = True
                    break
            else:
                res.append(b)
        out = res
    return out  # type: ignore[return-value]


def pad_box(box: Box, pad: float) -> Box:
    """Pad by `pad` of the box's own size on each side, clamped to 0..1."""
    w, h = box[2] - box[0], box[3] - box[1]
    return (max(0.0, box[0] - pad * w), max(0.0, box[1] - pad * h),
            min(1.0, box[2] + pad * w), min(1.0, box[3] + pad * h))


class FrameDiffMotionGate:
    """MotionGate implementation (see pipeline.MotionGate).

    width:         analysis width in px (height keeps the aspect ratio)
    threshold:     per-pixel |gray - background| needed to count as motion (0..255)
    min_area:      minimum blob area as a fraction of the frame (0.002 = 0.2 %)
    alpha:         background learning rate (cv2.accumulateWeighted)
    pad:           padding added around each blob, as a fraction of its size
    merge_gap:     blobs closer than this (normalized) are merged
    full_every_s:  return the whole frame at least this often (0 = never)
    global_frac:   if motion covers more than this fraction, return the whole frame
    """

    def __init__(
        self,
        width: int = 160,
        threshold: int = 25,
        min_area: float = 0.002,
        alpha: float = 0.05,
        blur: int = 5,
        pad: float = 0.25,
        merge_gap: float = 0.02,
        full_every_s: float = 5.0,
        global_frac: float = 0.5,
    ):
        self.width = width
        self.threshold = threshold
        self.min_area = min_area
        self.alpha = alpha
        self.blur = blur | 1  # odd kernel
        self.pad = pad
        self.merge_gap = merge_gap
        self.full_every_s = full_every_s
        self.global_frac = global_frac
        self._bg: Optional[np.ndarray] = None
        self._last_full_ts: Optional[float] = None
        self.last_reason = "none"          # init | periodic | global | motion | none
        self.last_motion_frac = 0.0

    def reset(self) -> None:
        self._bg = None
        self._last_full_ts = None

    def _gray(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w != self.width:
            nh = max(1, int(round(h * self.width / w)))
            image = cv2.resize(image, (self.width, nh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.blur > 1:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        return gray

    def _full(self, ts: float, reason: str) -> list[Box]:
        self._last_full_ts = ts
        self.last_reason = reason
        return [FULL_FRAME]

    def regions(self, frame: Frame) -> Sequence[Box]:
        gray = self._gray(frame.image)
        if self._bg is None or self._bg.shape != gray.shape:
            self._bg = gray.astype(np.float32)
            self.last_motion_frac = 0.0
            return self._full(frame.ts, "init")

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._bg))
        cv2.accumulateWeighted(gray, self._bg, self.alpha)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        h, w = mask.shape
        self.last_motion_frac = float(np.count_nonzero(mask)) / (h * w)

        if self.full_every_s > 0 and frame.ts - (self._last_full_ts or -1e18) >= self.full_every_s:
            return self._full(frame.ts, "periodic")
        if self.last_motion_frac > self.global_frac:
            return self._full(frame.ts, "global")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_px = self.min_area * h * w
        boxes: list[Box] = []
        for c in contours:
            if cv2.contourArea(c) < min_px:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append(pad_box((x / w, y / h, (x + bw) / w, (y + bh) / h), self.pad))
        if not boxes:
            self.last_reason = "none"
            return []
        self.last_reason = "motion"
        return merge_boxes(boxes, self.merge_gap)
