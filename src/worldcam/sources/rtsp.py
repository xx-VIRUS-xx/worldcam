from __future__ import annotations

import os
import time
from typing import Optional

import cv2

from ..types import Frame
from .base import FrameSource

# TCP avoids the packet-loss smearing UDP gives on Wi-Fi; nobuffer/low_delay
# cut FFmpeg's internal buffering. Must be set before the capture is opened.
_LOW_LATENCY_OPTS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


class StreamSource(FrameSource):
    """RTSP/HTTP stream: CCTV, or a phone running an IP-camera app.

    Reconnects with capped exponential backoff. read() returns None while it
    is reconnecting so the grabber thread keeps running.
    """

    def __init__(self, url: str, max_backoff: float = 8.0):
        self.url = url
        self.max_backoff = max_backoff
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self._fail = 0
        self._next_try = 0.0

    def _connect(self) -> bool:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _LOW_LATENCY_OPTS)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def open(self) -> None:
        if not self._connect():
            raise RuntimeError(f"cannot open stream {self._redacted()}")

    def _redacted(self) -> str:
        # Never log credentials embedded in rtsp://user:pass@host/...
        if "@" in self.url:
            scheme, rest = self.url.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return self.url

    def read(self) -> Optional[Frame]:
        now = time.monotonic()
        if self._cap is None:
            if now < self._next_try:
                time.sleep(0.05)
                return None
            if self._connect():
                self._fail = 0
            else:
                self._fail += 1
                self._next_try = now + min(self.max_backoff, 2 ** self._fail * 0.25)
                return None
        ok, img = self._cap.read()
        ts = time.monotonic()
        if not ok or img is None:
            self._cap.release()
            self._cap = None
            self._fail += 1
            self._next_try = ts + min(self.max_backoff, 2 ** self._fail * 0.25)
            return None
        self._seq += 1
        return Frame(self._seq, ts, img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict:
        d = {"url": self._redacted()}
        if self._cap is not None:
            d["actual"] = (
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )
        return d
