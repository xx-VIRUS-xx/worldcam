"""Grabber fixes relevant to long soak runs (no camera needed)."""
import threading
import time
from typing import Optional

from worldcam.capture import LatestFrameGrabber
from worldcam.sources import SyntheticSource
from worldcam.sources.base import FrameSource
from worldcam.types import Frame


def test_same_frame_returned_twice_is_counted_once():
    src = SyntheticSource(fps=5)  # slow producer: repeated get(0) hits the same frame
    with LatestFrameGrabber(src) as g:
        f1 = g.get(after_seq=0, timeout=1.0)
        f2 = g.get(after_seq=0, timeout=1.0)
        f3 = g.get(after_seq=0, timeout=1.0)
        assert f1 is not None and f1 is f2 is f3
        assert g.frames_consumed == 1
        assert g.dropped == g.frames_read - 1


class _BlockingSource(FrameSource):
    """read() blocks until released, like a stalled camera/stream. Records
    whether close() ever overlapped an in-flight read()."""

    def __init__(self):
        self.release = threading.Event()
        self.in_read = threading.Event()
        self.closed_during_read = False
        self.close_calls = 0
        self._reading = False
        self._seq = 0

    def open(self) -> None:
        pass

    def read(self) -> Optional[Frame]:
        self._reading = True
        self.in_read.set()
        self.release.wait(10)
        self._reading = False
        return None

    def close(self) -> None:
        self.close_calls += 1
        if self._reading:
            self.closed_during_read = True


def test_stop_does_not_close_source_while_read_in_flight():
    src = _BlockingSource()
    g = LatestFrameGrabber(src, join_timeout=0.2).start()
    assert src.in_read.wait(1.0)
    t0 = time.monotonic()
    g.stop()  # join times out: read() still blocked
    assert time.monotonic() - t0 < 1.0
    assert src.close_calls == 0, "must not release the source under a running read()"
    src.release.set()  # read() returns; the thread sees _stop and closes
    g._thread.join(2.0)
    assert not g._thread.is_alive()
    assert src.close_calls == 1
    assert not src.closed_during_read


def test_stop_closes_source_when_thread_exits_normally():
    src = _BlockingSource()
    src.release.set()  # read() returns immediately
    g = LatestFrameGrabber(src).start()
    time.sleep(0.05)
    g.stop()
    assert src.close_calls == 1
    assert not g._thread.is_alive()


class _RaisingSource(SyntheticSource):
    """Raises a non-EOF error on some reads (e.g. a cv2.error)."""

    def read(self):
        f = super().read()
        if f.seq % 3 == 0:
            raise RuntimeError("boom")
        return f


def test_unexpected_read_exception_does_not_kill_grabber():
    src = _RaisingSource(fps=200)
    with LatestFrameGrabber(src) as g:
        last = 0
        for _ in range(10):
            f = g.get(after_seq=last, timeout=1.0)
            assert f is not None, "grabber thread died on a read exception"
            last = f.seq
        assert g.read_failures > 0
        assert g._thread.is_alive()
