from __future__ import annotations

import threading
import time
from typing import Optional

from .sources.base import FrameSource
from .types import Frame


class LatestFrameGrabber:
    """Reads a FrameSource on a dedicated thread and keeps ONLY the newest
    frame. Slow consumers skip frames instead of building latency, which is
    what makes a single-loop OpenCV app feel laggy.

    Usage:
        with LatestFrameGrabber(source) as g:
            last = 0
            while True:
                f = g.get(after_seq=last, timeout=1.0)
                if f is None: continue
                last = f.seq
                ...process f...
    """

    def __init__(self, source: FrameSource):
        self.source = source
        self._cv = threading.Condition()
        self._latest: Optional[Frame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._eof = False
        self.frames_read = 0
        self.frames_consumed = 0
        self.read_failures = 0

    # ---- lifecycle -------------------------------------------------
    def start(self) -> "LatestFrameGrabber":
        self.source.open()
        self._thread = threading.Thread(target=self._run, name="grabber", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.source.close()

    def __enter__(self) -> "LatestFrameGrabber":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- producer --------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                f = self.source.read()
            except EOFError:
                with self._cv:
                    self._eof = True
                    self._cv.notify_all()
                return
            if f is None:
                self.read_failures += 1
                time.sleep(0.005)
                continue
            with self._cv:
                self._latest = f
                self.frames_read += 1
                self._cv.notify_all()

    # ---- consumer --------------------------------------------------
    @property
    def eof(self) -> bool:
        return self._eof

    def get(self, after_seq: int = 0, timeout: float = 1.0) -> Optional[Frame]:
        """Newest frame with seq > after_seq, or None on timeout/EOF."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._latest is None or self._latest.seq <= after_seq:
                if self._eof:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            self.frames_consumed += 1
            return self._latest

    def peek(self) -> Optional[Frame]:
        """Non-blocking: the newest frame so far (or None), without
        affecting drop accounting. Used for on-demand full-res crops."""
        with self._cv:
            return self._latest

    @property
    def dropped(self) -> int:
        """Frames read from the camera that no consumer ever saw."""
        return max(0, self.frames_read - self.frames_consumed)
