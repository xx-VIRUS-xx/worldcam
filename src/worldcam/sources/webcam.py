from __future__ import annotations

import sys
import time
from typing import Optional

import cv2

from ..types import Frame
from .base import FrameSource


def _default_backend() -> int:
    # The default backend is the most common cause of laggy webcams on Windows.
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def _fourcc_to_str(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


class WebcamSource(FrameSource):
    """Local camera (laptop webcam or a phone exposed as a webcam by
    DroidCam/Camo). Requests MJPG first: YUYV caps FPS at high resolutions
    because of USB bandwidth.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fourcc: str = "MJPG",
        backend: Optional[int] = None,
    ):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.fourcc = fourcc
        self.backend = _default_backend() if backend is None else backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera index {self.index}")
        # Order matters: fourcc first, then size, then fps.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # not honoured by every backend
        self._cap = cap

    def read(self) -> Optional[Frame]:
        assert self._cap is not None, "open() first"
        ok, img = self._cap.read()
        ts = time.monotonic()
        if not ok or img is None:
            return None
        self._seq += 1
        return Frame(self._seq, ts, img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict:
        if self._cap is None:
            return {}
        c = self._cap
        return {
            "requested": f"{self.width}x{self.height}@{self.fps} {self.fourcc}",
            "actual": (
                f"{int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                f"@{c.get(cv2.CAP_PROP_FPS):.0f} "
                f"{_fourcc_to_str(c.get(cv2.CAP_PROP_FOURCC))}"
            ),
        }
