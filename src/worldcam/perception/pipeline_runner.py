"""PerceptionLoop: grabber -> motion gate -> detector -> tracker -> TrackUpdate.

Stage-only (no identity, no graph). Works with
- DualGrabber: get() returns the SUB frame; best-snapshot crops come from the
  main stream via crop_with_frame(box, ref=sub_frame, max_skew).
- LatestFrameGrabber (one camera): get() returns the FULL-RES frame. The loop
  makes a `sub_width`-wide copy for motion + detection and crops best
  snapshots from the full-res frame it already holds (exact same instant).

No queues: the loop always takes the newest frame (after_seq), so a slow
detector drops frames instead of building latency.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Iterator, Optional

import cv2

from ..geometry import Box, crop_norm
from ..metrics import percentile
from ..pipeline import Detection, TrackUpdate
from ..types import Frame
from .tracker import CropFn, IouTracker

STAGES = ("wait", "resize", "motion", "detect", "track", "total")


class StageTimes:
    """Bounded per-stage latency samples (ms) for p50/p95."""

    def __init__(self, maxlen: int = 20000):
        self.samples = {s: deque(maxlen=maxlen) for s in STAGES}

    def add(self, stage: str, ms: float) -> None:
        self.samples[stage].append(ms)

    def pct(self, stage: str, p: float) -> float:
        return percentile(list(self.samples[stage]), p)

    def summary(self) -> dict[str, tuple[float, float, int]]:
        return {s: (self.pct(s, 50), self.pct(s, 95), len(v)) for s, v in self.samples.items()}


def downscale(frame: Frame, width: int) -> Frame:
    h, w = frame.image.shape[:2]
    if width <= 0 or w <= width:
        return frame
    nh = int(round(h * width / w))
    return Frame(frame.seq, frame.ts, cv2.resize(frame.image, (width, nh), interpolation=cv2.INTER_AREA))


class PerceptionLoop:
    def __init__(
        self,
        grabber,
        detector,
        tracker: Optional[IouTracker] = None,
        gate=None,
        sub_width: int = 640,
        max_skew: Optional[float] = 0.25,
        crop_pad: float = 0.1,
    ):
        self.grabber = grabber
        self.detector = detector
        self.tracker = tracker or IouTracker()
        self.gate = gate
        self.sub_width = sub_width
        self.max_skew = max_skew
        self.crop_pad = crop_pad
        self.dual = hasattr(grabber, "crop_with_frame")
        self.times = StageTimes()
        self.frames = 0
        self.detector_runs = 0
        self.motion_skips = 0
        self.max_active = 0
        self._last_seq = 0
        # last-iteration state for overlays
        self.last_main: Optional[Frame] = None
        self.last_sub: Optional[Frame] = None
        self.last_regions: Optional[list[Box]] = None
        self.last_dets: list[Detection] = []

    def _crop_fn(self, sub: Frame, main: Optional[Frame]) -> CropFn:
        if self.dual:
            g = self.grabber

            def fn(box: Box):
                r = g.crop_with_frame(box, pad=self.crop_pad, ref=sub, max_skew=self.max_skew)
                if r.fallback:
                    return r.image, ("src=sub", "main_unavailable_or_skewed")
                skew = abs(r.frame.ts - sub.ts) * 1000 if r.frame is not None else -1
                return r.image, ("src=main", f"skew_ms={skew:.0f}")
            return fn
        assert main is not None

        def fn2(box: Box):
            h, w = main.image.shape[:2]
            return crop_norm(main.image, box, self.crop_pad), ("src=main", f"main={w}x{h}")
        return fn2

    def step(self, frame: Frame) -> TrackUpdate:
        """Process one frame from the grabber (sub frame if dual, else full-res)."""
        t0 = time.perf_counter()
        if self.dual:
            sub, main = frame, None
        else:
            main = frame
            sub = downscale(frame, self.sub_width)
        t1 = time.perf_counter()
        regions = None
        if self.gate is not None:
            regions = list(self.gate.regions(sub))
        t2 = time.perf_counter()
        ran = not (regions is not None and not regions)
        if ran:
            dets: list[Detection] = list(self.detector.detect(sub, regions))
            self.detector_runs += 1
        else:
            dets = []
            self.motion_skips += 1
        t3 = time.perf_counter()
        upd = self.tracker.update(sub, dets, observed_regions=regions,
                                  crop_fn=self._crop_fn(sub, main))
        t4 = time.perf_counter()
        self.frames += 1
        self.max_active = max(self.max_active, len(upd.active))
        self.times.add("resize", (t1 - t0) * 1000)
        self.times.add("motion", (t2 - t1) * 1000)
        if ran:  # detect latency only counts frames where YOLO actually ran
            self.times.add("detect", (t3 - t2) * 1000)
        self.times.add("track", (t4 - t3) * 1000)
        self.times.add("total", (t4 - t0) * 1000)
        self.last_main, self.last_sub, self.last_regions, self.last_dets = main, sub, regions, dets
        return upd

    def __iter__(self) -> Iterator[TrackUpdate]:
        return self.run()

    def run(self, seconds: Optional[float] = None, timeout: float = 1.0) -> Iterator[TrackUpdate]:
        """Yield one TrackUpdate per processed frame until EOF or `seconds`."""
        t_end = None if not seconds else time.monotonic() + seconds
        while t_end is None or time.monotonic() < t_end:
            tw = time.perf_counter()
            f = self.grabber.get(after_seq=self._last_seq, timeout=timeout)
            if f is None:
                if getattr(self.grabber, "eof", False):
                    return
                continue
            self.times.add("wait", (time.perf_counter() - tw) * 1000)
            self._last_seq = f.seq
            yield self.step(f)

    @property
    def skip_ratio(self) -> float:
        return self.motion_skips / self.frames if self.frames else 0.0
