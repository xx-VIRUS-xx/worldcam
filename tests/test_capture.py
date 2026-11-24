import time

from worldcam.capture import LatestFrameGrabber
from worldcam.sources import SyntheticSource, make_source
from worldcam.sources.rtsp import StreamSource


def test_slow_consumer_gets_fresh_frames_and_drops_the_rest():
    src = SyntheticSource(fps=60)
    ages = []
    with LatestFrameGrabber(src) as g:
        last = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:
            f = g.get(after_seq=last, timeout=1.0)
            assert f is not None
            last = f.seq
            ages.append(time.monotonic() - f.ts)
            time.sleep(0.2)  # consumer ~5 fps vs producer 60 fps
        dropped = g.dropped
    assert dropped > 30, "producer should have outrun the consumer"
    # No backlog: a frame is never older than ~one producer period at hand-off
    # (small allowance for scheduling jitter).
    assert max(ages) < 0.1, f"stale frame delivered: {max(ages):.3f}s"


def test_sequence_is_monotonic_and_get_after_seq_never_repeats():
    src = SyntheticSource(fps=100)
    seen = []
    with LatestFrameGrabber(src) as g:
        last = 0
        for _ in range(20):
            f = g.get(after_seq=last, timeout=1.0)
            assert f is not None
            seen.append(f.seq)
            last = f.seq
    assert seen == sorted(set(seen))


def test_eof_ends_cleanly():
    src = SyntheticSource(fps=200, n_frames=10)
    with LatestFrameGrabber(src) as g:
        last, got = 0, 0
        while True:
            f = g.get(after_seq=last, timeout=1.0)
            if f is None:
                assert g.eof
                break
            last = f.seq
            got += 1
    assert 1 <= got <= 10


def test_make_source_routing():
    assert make_source("synthetic").__class__.__name__ == "SyntheticSource"
    assert make_source("0").__class__.__name__ == "WebcamSource"
    assert make_source("rtsp://u:p@h/x").__class__.__name__ == "StreamSource"
    assert make_source("clip.mp4").__class__.__name__ == "FileSource"


def test_stream_url_credentials_are_redacted():
    s = StreamSource("rtsp://admin:hunter2@192.168.1.10:554/stream1")
    assert "hunter2" not in s._redacted()
    assert "admin" not in s._redacted()
    assert "192.168.1.10" in s._redacted()
