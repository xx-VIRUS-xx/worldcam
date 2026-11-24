from __future__ import annotations

from typing import Optional

import numpy as np

from ..capture import LatestFrameGrabber
from ..geometry import Box, crop_norm
from ..types import Frame
from .base import FrameSource


class DualGrabber:
    """Two streams from one camera: a low-res SUBstream for detection and a
    high-res MAINstream for crops (faces, VLM keyframes). Same pattern NVRs use.

    - get() paces the pipeline on the substream (cheap, newest-frame-only).
    - crop() cuts a NORMALIZED box from the newest mainstream frame, so
      detector coordinates never need rescaling.

    Known limitation: the two streams are independent connections, so the main
    frame can be a few tens of ms newer or older than the sub frame. Fine for
    identity/VLM crops of slow-moving subjects; do not use for pixel-exact work.
    If the mainstream is down, crop() falls back to the substream.
    """

    def __init__(self, sub: FrameSource, main: FrameSource):
        self.sub = LatestFrameGrabber(sub)
        self.main = LatestFrameGrabber(main)

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
        f = self.main.peek() or self.sub.peek()
        if f is None:
            return None
        return crop_norm(f.image, box, pad)
