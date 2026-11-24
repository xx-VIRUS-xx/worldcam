from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from ..types import Frame
from .base import FrameSource


class FileSource(FrameSource):
    """Video file, paced to real time. Lets you develop and test the whole
    pipeline without a camera, and replay recorded CCTV clips reproducibly.
    """

    def __init__(self, path: str, realtime: bool = True, loop: bool = False):
        self.path, self.realtime, self.loop = path, realtime, loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self._period = 1 / 30
        self._next = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open file {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._period = 1.0 / fps
        self._next = time.monotonic()
        self._cap = cap

    def read(self) -> Optional[Frame]:
        assert self._cap is not None, "open() first"
        if self.realtime:
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next += self._period
        ok, img = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None
            raise EOFError
        self._seq += 1
        return Frame(self._seq, time.monotonic(), img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticSource(FrameSource):
    """Generated frames with a moving box and the seq number burned in.
    No hardware, no files. Used by tests and by `--source synthetic`.
    `read_delay` simulates camera frame period.
    """

    def __init__(self, width=640, height=360, fps=30, n_frames: Optional[int] = None):
        self.w, self.h, self.fps, self.n = width, height, fps, n_frames
        self._seq = 0
        self._next = 0.0

    def open(self) -> None:
        self._next = time.monotonic()

    def read(self) -> Optional[Frame]:
        if self.n is not None and self._seq >= self.n:
            raise EOFError
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next += 1.0 / self.fps
        self._seq += 1
        img = np.zeros((self.h, self.w, 3), np.uint8)
        x = (self._seq * 7) % (self.w - 60)
        cv2.rectangle(img, (x, 100), (x + 60, 160), (0, 200, 0), -1)
        cv2.putText(img, f"#{self._seq}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return Frame(self._seq, time.monotonic(), img)

    def close(self) -> None:
        pass
