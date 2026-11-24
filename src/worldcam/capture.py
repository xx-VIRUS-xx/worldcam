from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .sources.base import FrameSource
from .types import Frame

log = logging.getLogger(__name__)


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

    def __init__(self, source: FrameSource, join_timeout: float = 3.0):
        self.source = source
        self.join_timeout = join_timeout
        self._cv = threading.Condition()
        self._latest: Optional[Frame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._eof = False
        self.frames_read = 0
        # Distinct frames handed to consumers. A frame returned twice (e.g.
        # get(after_seq=0) repeatedly, or two consumers) counts once, so
        # `dropped` really means "read but never seen".
        self.frames_consumed = 0
        self._consumed_seq = 0
        self.read_failures = 0
        # Who closes the source: stop() if the thread has exited, otherwise
        # the thread itself once its in-flight read() returns. Closing (e.g.
        # cv2.VideoCapture.release) while read() is running on another thread
        # is a use-after-release in native code.
        self._close_lock = threading.Lock()
        self._thread_done = False
        self._close_deferred = False

    # ---- lifecycle -------------------------------------------------
    def start(self) -> "LatestFrameGrabber":
        self.source.open()
        self._thread = threading.Thread(target=self._run, name="grabber", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.join_timeout)
            with self._close_lock:
                if not self._thread_done:
                    # read() is still blocked; the thread closes the source
                    # when it returns (it sees _stop and exits).
                    self._close_deferred = True
                    log.warning("grabber thread still in read() after %.1fs; "
                                "deferring source.close() to the thread", self.join_timeout)
                    return
        self.source.close()

    def __enter__(self) -> "LatestFrameGrabber":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- producer --------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        finally:
            with self._close_lock:
                self._thread_done = True
                deferred = self._close_deferred
            if deferred:
                self.source.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                f = self.source.read()
            except EOFError:
                with self._cv:
                    self._eof = True
                    self._cv.notify_all()
                return
            except Exception:
                # The contract says transient failures return None. A source
                # that raises anyway must not silently kill the thread (the
                # consumer would then wait forever without EOF): count it,
                # back off, keep going.
                if self.read_failures == 0 or self.read_failures % 100 == 0:
                    log.exception("source.read() raised; treating as a read failure")
                self.read_failures += 1
                time.sleep(0.05)
                continue
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
        """Newest frame with seq > after_seq, or None on timeout/EOF.

        Pass the last seq you received as after_seq; with the default 0 the
        same newest frame is returned again on every call."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._latest is None or self._latest.seq <= after_seq:
                if self._eof:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            f = self._latest
            if f.seq > self._consumed_seq:
                self._consumed_seq = f.seq
                self.frames_consumed += 1
            return f

    def peek(self) -> Optional[Frame]:
        """Non-blocking: the newest frame so far (or None), without
        affecting drop accounting. Used for on-demand full-res crops."""
        with self._cv:
            return self._latest

    @property
    def dropped(self) -> int:
        """Frames read from the camera that no consumer ever saw."""
        return max(0, self.frames_read - self.frames_consumed)
