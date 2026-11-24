from __future__ import annotations

import threading
from collections import deque
from typing import NamedTuple, Optional

import numpy as np

from ..capture import LatestFrameGrabber
from ..geometry import Box, crop_norm
from ..types import Frame
from .base import FrameSource


class CropResult(NamedTuple):
    """What crop_with_frame() returns.

    image:    the crop (None if no frame was available or the box was empty).
    frame:    the frame the crop was cut from, so callers can compare Frame.ts
              with the substream frame they detected on.
    fallback: True if the crop came from the substream (mainstream down, not
              started yet, or too far off in time) instead of the mainstream.
    """

    image: Optional[np.ndarray]
    frame: Optional[Frame]
    fallback: bool


class _RingTap(FrameSource):
    """Wraps the mainstream source and remembers its last `size` frames.

    Bounded (deque maxlen), filled by the grabber thread only, never consumed:
    it is a lookup table for "the main frame closest in time to this sub
    frame", not a queue. Frames are references, not copies.
    """

    def __init__(self, inner: FrameSource, size: int):
        self.inner = inner
        self.ring: deque[Frame] = deque(maxlen=max(1, size))
        self.lock = threading.Lock()

    def open(self) -> None:
        self.inner.open()

    def read(self) -> Optional[Frame]:
        f = self.inner.read()
        if f is not None:
            with self.lock:
                self.ring.append(f)
        return f

    def close(self) -> None:
        self.inner.close()

    def describe(self) -> dict:
        return self.inner.describe()

    def closest(self, ts: float) -> Optional[Frame]:
        with self.lock:
            if not self.ring:
                return None
            return min(self.ring, key=lambda f: abs(f.ts - ts))


class DualGrabber:
    """Two streams from one camera: a low-res SUBstream for detection and a
    high-res MAINstream for crops (faces, VLM keyframes). Same pattern NVRs use.

    - get() paces the pipeline on the substream (cheap, newest-frame-only).
    - crop() cuts a NORMALIZED box from the newest mainstream frame, so
      detector coordinates never need rescaling.
    - crop_with_frame(box, ref=sub_frame) also returns the main frame used and
      whether it fell back to the substream. With main_ring > 1 it picks the
      remembered main frame whose ts is closest to ref.ts (see docs/dual.md).

    Known limitation: the two streams are independent connections, and
    Frame.ts is software arrival time, so matching by ts aligns arrival times,
    not sensor capture times. Fine for identity/VLM crops of slow-moving
    subjects; do not use for pixel-exact work.
    """

    def __init__(self, sub: FrameSource, main: FrameSource, main_ring: int = 1):
        self._tap = _RingTap(main, main_ring)
        self.sub = LatestFrameGrabber(sub)
        self.main = LatestFrameGrabber(self._tap)
        self.fallbacks = 0

    def start(self) -> "DualGrabber":
        self.sub.start()
        try:
            self.main.start()
        except Exception:
            self.sub.stop()
            raise
        return self

    def stop(self) -> None:
        self.main.stop()
        self.sub.stop()

    def __enter__(self) -> "DualGrabber":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def get(self, after_seq: int = 0, timeout: float = 1.0) -> Optional[Frame]:
        return self.sub.get(after_seq=after_seq, timeout=timeout)

    @property
    def eof(self) -> bool:
        return self.sub.eof

    def crop(self, box: Box, pad: float = 0.1) -> Optional[np.ndarray]:
        """Backward-compatible: crop from the newest main frame (or newest sub
        frame if main has none yet)."""
        return self.crop_with_frame(box, pad).image

    def crop_with_frame(
        self,
        box: Box,
        pad: float = 0.1,
        ref: Optional[Frame] = None,
        max_skew: Optional[float] = None,
    ) -> CropResult:
        """Crop a normalized box and report which frame it came from.

        ref:      the substream frame the box was detected on. If given, the
                  main frame closest to ref.ts is used (newest if main_ring=1),
                  and fallback crops come from ref itself rather than whatever
                  sub frame is newest now.
        max_skew: seconds. If the chosen main frame is further than this from
                  ref.ts (e.g. mainstream stalled), fall back to the substream.
        """
        main_f = self._tap.closest(ref.ts) if ref is not None else self.main.peek()
        if (
            main_f is not None
            and ref is not None
            and max_skew is not None
            and abs(main_f.ts - ref.ts) > max_skew
        ):
            main_f = None
        if main_f is not None:
            return CropResult(crop_norm(main_f.image, box, pad), main_f, False)
        sub_f = ref if ref is not None else self.sub.peek()
        if sub_f is None:
            return CropResult(None, None, True)
        self.fallbacks += 1
        return CropResult(crop_norm(sub_f.image, box, pad), sub_f, True)
