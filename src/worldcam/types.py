from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured frame.

    ts: time.monotonic() taken immediately after the source's read() returned.
        This is the earliest timestamp software can see; it does NOT include
        sensor/driver/USB latency (true glass-to-software latency needs an
        external reference, e.g. filming a running clock).
    seq: monotonically increasing per source, so consumers can detect drops.
    image: BGR uint8 HxWx3, full native resolution. Downscale in consumers.
    """

    seq: int
    ts: float
    image: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h
