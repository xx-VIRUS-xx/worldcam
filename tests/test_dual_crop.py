import time
from typing import Optional

import numpy as np

from worldcam.sources import SyntheticSource
from worldcam.sources.base import FrameSource
from worldcam.sources.dual import CropResult, DualGrabber
from worldcam.types import Frame

BOX = (0.0, 0.0, 0.5, 0.5)


class _DeadSource(FrameSource):
    """Opens fine, never yields a frame (mainstream down)."""

    def open(self) -> None:
        pass

    def read(self) -> Optional[Frame]:
        time.sleep(0.01)
        return None

    def close(self) -> None:
        pass


class _StallAfter(SyntheticSource):
    """Delivers `n_ok` frames, then stalls (returns None) forever."""

    def __init__(self, n_ok: int, **kw):
        super().__init__(**kw)
        self.n_ok = n_ok

    def read(self) -> Optional[Frame]:
        if self._seq >= self.n_ok:
            time.sleep(0.01)
            return None
        return super().read()


def _wait_main(d: DualGrabber, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while d.main.peek() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert d.main.peek() is not None


def test_crop_with_frame_returns_main_frame_used():
    sub = SyntheticSource(width=320, height=180, fps=30)
    main = SyntheticSource(width=1280, height=720, fps=30)
    with DualGrabber(sub, main) as d:
        f = d.get(timeout=2.0)
        assert f is not None
        _wait_main(d)
        r = d.crop_with_frame(BOX, pad=0.0, ref=f)
        assert isinstance(r, CropResult)
        assert r.fallback is False
        assert r.frame is not None and r.frame.shape == (1280, 720)
        assert r.image is not None and r.image.shape[:2] == (360, 640)
        # The crop really is cut from the returned frame.
        assert np.array_equal(r.image, r.frame.image[:360, :640])
        assert abs(r.frame.ts - f.ts) < 1.0
        # Without ref it uses the newest main frame (old crop() semantics).
        r2 = d.crop_with_frame(BOX, pad=0.0)
        assert r2.frame is not None and r2.frame.shape == (1280, 720) and not r2.fallback
        # Backward compatible crop() still returns a bare ndarray.
        c = d.crop(BOX, pad=0.0)
        assert isinstance(c, np.ndarray) and c.shape[:2] == (360, 640)
        assert d.fallbacks == 0


def test_falls_back_to_sub_frame_when_main_is_down():
    sub = SyntheticSource(width=320, height=180, fps=30)
    with DualGrabber(sub, _DeadSource()) as d:
        f = d.get(timeout=2.0)
        assert f is not None
        r = d.crop_with_frame(BOX, pad=0.0, ref=f)
        assert r.fallback is True
        assert r.frame is f, "fallback must crop the sub frame the box came from"
        assert r.image is not None and r.image.shape[:2] == (90, 160)
        r2 = d.crop_with_frame(BOX, pad=0.0)  # no ref: newest sub frame
        assert r2.fallback is True and r2.frame is not None and r2.frame.shape == (320, 180)
        assert d.crop(BOX, pad=0.0).shape[:2] == (90, 160)
        assert d.fallbacks == 3


def test_max_skew_falls_back_when_main_stalls():
    sub = SyntheticSource(width=320, height=180, fps=30)
    main = _StallAfter(3, width=1280, height=720, fps=30)
    with DualGrabber(sub, main, main_ring=4) as d:
        _wait_main(d)
        time.sleep(0.4)  # main is now >= ~300 ms stale
        f = d.get(after_seq=0, timeout=2.0)
        assert f is not None
        stale = d.crop_with_frame(BOX, pad=0.0, ref=f)
        assert stale.fallback is False and stale.frame.ts < f.ts - 0.2
        r = d.crop_with_frame(BOX, pad=0.0, ref=f, max_skew=0.1)
        assert r.fallback is True and r.frame is f
        assert d.fallbacks == 1


def test_ring_picks_main_frame_closest_to_sub_ts_and_is_bounded():
    offsets = {}
    for ring in (1, 6):
        sub = SyntheticSource(width=320, height=180, fps=30)
        main = SyntheticSource(width=640, height=360, fps=30)
        vals = []
        with DualGrabber(sub, main, main_ring=ring) as d:
            _wait_main(d)
            last = 0
            for _ in range(5):
                f = d.get(after_seq=last, timeout=2.0)
                assert f is not None
                last = f.seq
                time.sleep(0.1)  # simulated detector work
                r = d.crop_with_frame(BOX, pad=0.0, ref=f)
                assert not r.fallback
                vals.append(abs(r.frame.ts - f.ts))
            assert len(d._tap.ring) <= ring
        offsets[ring] = sorted(vals)[len(vals) // 2]
    # Newest-only: main frame is ~work time newer than the sub frame.
    assert offsets[1] > 0.05, offsets
    # Ring of 6 (~200 ms at 30 fps) covers the 100 ms work: within ~1 period.
    assert offsets[6] < 0.03, offsets
