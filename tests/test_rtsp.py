"""StreamSource tests. No network: cv2.VideoCapture is replaced by a fake."""
import numpy as np
import pytest

import worldcam.sources.rtsp as rtsp
from worldcam.capture import LatestFrameGrabber
from worldcam.sources.rtsp import StreamSource, _backoff_delay, redact_url


# ---- redaction -------------------------------------------------------
@pytest.mark.parametrize("url,secrets,keep", [
    ("rtsp://admin:hunter2@192.168.1.10:554/stream1", ["admin", "hunter2"], "192.168.1.10:554/stream1"),
    ("rtsp://admin:p@ss@cam.local/live", ["admin", "p@ss", "ss@"], "cam.local/live"),
    ("rtsp://admin:a:b@c@10.0.0.2/s", ["admin", "a:b", "b@c", "c@"], "10.0.0.2/s"),
    ("rtsp://user@10.0.0.3/s", ["user"], "10.0.0.3/s"),
    ("rtsp://admin:pa/ss@10.0.0.4/s", ["admin", "pa/ss", "pa"], "10.0.0.4/s"),
    ("http://10.0.0.5:8080/video?user=admin&pwd=secret", ["admin", "secret"], "10.0.0.5:8080/video"),
])
def test_redaction_hides_credentials(url, secrets, keep):
    r = StreamSource(url)._redacted()
    for s in secrets:
        assert s not in r, (s, r)
    assert keep in r
    assert r == redact_url(url)


def test_redaction_leaves_plain_urls_alone():
    assert redact_url("rtsp://127.0.0.1:8554/test") == "rtsp://127.0.0.1:8554/test"


def test_old_first_at_split_would_have_leaked():
    # The previous implementation split on the first '@'; document why it changed.
    url = "rtsp://admin:p@ss@cam/live"
    old = "rtsp://***@" + url.split("://", 1)[1].split("@", 1)[1]
    assert "ss@" in old
    assert "ss" not in redact_url(url).replace("***", "")


# ---- backoff ---------------------------------------------------------
def test_backoff_doubles_and_caps():
    assert [_backoff_delay(n, 8.0) for n in range(1, 7)] == [0.5, 1, 2, 4, 8, 8]
    assert _backoff_delay(1000, 8.0) == 8.0  # no overflow


# ---- fake capture ----------------------------------------------------
class FakeCap:
    """Scripted cv2.VideoCapture. `world` controls what the 'server' does."""

    def __init__(self, world, url, api=None, params=None):
        self.world = world
        world["opens"].append(params)
        self._open = world["up"]
        if world.get("raise_on_open"):
            raise RuntimeError("cv2.error stand-in")

    def isOpened(self):
        return self._open

    def set(self, *a):
        return True

    def get(self, prop):
        return 0

    def read(self):
        if self.world.get("raise_on_read"):
            raise RuntimeError("cv2.error stand-in")
        if self.world.get("read_delay"):  # simulated time spent blocked in FFmpeg
            self.world["clock"]["t"] += self.world["read_delay"]
        if not self.world["up"]:
            return False, None
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self._open = False


@pytest.fixture
def world(monkeypatch):
    w = {"up": True, "opens": []}
    monkeypatch.setattr(rtsp.cv2, "VideoCapture", lambda *a: FakeCap(w, *a))
    clock = {"t": 1000.0}
    monkeypatch.setattr(rtsp.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(rtsp.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    w["clock"] = clock
    return w


def test_timeouts_are_passed_as_open_params(world):
    s = StreamSource("rtsp://h/x", timeout_ms=4000)
    s.open()
    p = world["opens"][0]
    assert p[p.index(rtsp.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC) + 1] == 4000
    assert p[p.index(rtsp.cv2.CAP_PROP_READ_TIMEOUT_MSEC) + 1] == 4000


def test_read_that_hits_timeout_is_a_stall_not_a_frame(world):
    # FFmpeg returns ok=True with a stale decoder-drained frame on timeout.
    s = StreamSource("rtsp://h/x", timeout_ms=4000)
    s.open()
    world["read_delay"] = 0.03
    assert s.read() is not None
    world["read_delay"] = 4.0
    assert s.read() is None
    assert not s.connected
    world["read_delay"] = 0.03
    world["clock"]["t"] += 10
    while s.read() is None:
        pass
    assert s.reconnects == 1


def test_slow_but_live_reads_are_kept(world):
    s = StreamSource("rtsp://h/x", timeout_ms=4000)
    s.open()
    world["read_delay"] = 2.0  # e.g. a 0.5 fps camera
    assert s.read() is not None and s.connected


def test_reconnect_counting_and_backoff_schedule(world):
    clock = world["clock"]
    s = StreamSource("rtsp://u:p@h/x", max_backoff=2.0)
    s.open()
    assert s.connected and s.reconnects == 0
    assert s.read() is not None

    world["up"] = False
    attempts = []
    n_opens = len(world["opens"])
    for _ in range(200):
        assert s.read() is None
        if len(world["opens"]) != n_opens:
            n_opens = len(world["opens"])
            attempts.append(clock["t"])
    assert not s.connected
    gaps = [round(b - a, 2) for a, b in zip(attempts, attempts[1:])]
    # Failed read -> wait 0.5 -> failed open -> 1.0 -> 2.0 (capped) -> 2.0 ...
    assert gaps[:3] == [1.0, 2.0, 2.0]
    assert s.reconnects == 0

    world["up"] = True
    clock["t"] += 10
    f = None
    while f is None:
        f = s.read()
    assert s.connected and s.reconnects == 1

    # A second outage counts as a second reconnect.
    world["up"] = False
    assert s.read() is None
    world["up"] = True
    clock["t"] += 10
    while s.read() is None:
        pass
    assert s.reconnects == 2
    assert s.describe()["reconnects"] == 2
    assert "p@" not in s.describe()["url"]


def test_read_never_raises_on_cv2_exceptions(world):
    s = StreamSource("rtsp://h/x")
    s.open()
    world["raise_on_read"] = True
    assert s.read() is None
    world["raise_on_read"] = False
    world["raise_on_open"] = True
    world["clock"]["t"] += 10
    for _ in range(5):
        assert s.read() is None
    world["raise_on_open"] = False
    world["clock"]["t"] += 10
    while s.read() is None:
        pass
    assert s.reconnects == 1


def test_initial_open_failure_raises_without_leaking_credentials(world):
    world["up"] = False
    with pytest.raises(RuntimeError) as e:
        StreamSource("rtsp://admin:s3cret@h/x", open_deadline=0).open()
    assert "s3cret" not in str(e.value)


def test_grabber_survives_outage(monkeypatch):
    # Real clock here; the fake server just toggles.
    w = {"up": True, "opens": []}
    monkeypatch.setattr(rtsp.cv2, "VideoCapture", lambda *a: FakeCap(w, *a))
    import time
    s = StreamSource("rtsp://h/x", max_backoff=0.2)
    with LatestFrameGrabber(s) as g:
        f = g.get(timeout=1.0)
        assert f is not None
        w["up"] = False
        time.sleep(0.3)
        assert g._thread.is_alive()
        w["up"] = True
        f2 = g.get(after_seq=g.peek().seq, timeout=2.0)
        assert f2 is not None
        assert g._thread.is_alive()
    assert s.reconnects >= 1


# ---- open() at startup -------------------------------------------------
def test_open_retries_until_camera_appears(world):
    clock = world["clock"]
    world["up"] = False
    t0 = clock["t"]
    real_sleep = rtsp.time.sleep

    def sleep(s):  # camera comes up 2.5 s after start
        real_sleep(s)
        if clock["t"] - t0 >= 2.5:
            world["up"] = True

    rtsp.time.sleep = sleep
    try:
        s = StreamSource("rtsp://h/x", open_deadline=10.0)
        s.open()
    finally:
        rtsp.time.sleep = real_sleep
    assert s.connected and s.reconnects == 0
    # attempts at 0, 0.5, 1.5, 3.5 -> 4 opens
    assert len(world["opens"]) == 4
    assert s.read() is not None


def test_open_gives_up_at_deadline_with_clear_error(world):
    clock = world["clock"]
    world["up"] = False
    t0 = clock["t"]
    with pytest.raises(RuntimeError) as e:
        StreamSource("rtsp://admin:s3cret@h/x", open_deadline=10.0, max_backoff=8.0).open()
    msg = str(e.value)
    assert "s3cret" not in msg and "admin" not in msg
    assert "attempt" in msg and "open_deadline=10s" in msg
    # attempts at 0, 0.5, 1.5, 3.5, 7.5 (next would be 15.5 > 10)
    assert len(world["opens"]) == 5
    assert clock["t"] - t0 <= 10.0


def test_open_deadline_zero_is_single_attempt(world):
    world["up"] = False
    with pytest.raises(RuntimeError):
        StreamSource("rtsp://h/x", open_deadline=0).open()
    assert len(world["opens"]) == 1
