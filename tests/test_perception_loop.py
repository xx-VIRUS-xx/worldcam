from typing import Optional

import numpy as np
import pytest

from worldcam.perception.boxes import overlaps_any
from worldcam.perception.detector import YoloDetector, find_weights
from worldcam.perception.motion import FrameDiffMotionGate
from worldcam.perception.pipeline_runner import PerceptionLoop
from worldcam.perception.tracker import IouTracker
from worldcam.pipeline import Detection, TrackState
from worldcam.sources.dual import CropResult
from worldcam.types import Frame


class FakeGrabber:
    """Single-camera grabber stand-in: full-res frames, then EOF."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.eof = False

    def get(self, after_seq=0, timeout=1.0) -> Optional[Frame]:
        if self.i >= len(self.frames):
            self.eof = True
            return None
        f = self.frames[self.i]
        self.i += 1
        return f


class FakeDualGrabber(FakeGrabber):
    def __init__(self, subs, main_img):
        super().__init__(subs)
        self.main_img = main_img
        self.refs = []

    def crop_with_frame(self, box, pad=0.1, ref=None, max_skew=None):
        from worldcam.geometry import crop_norm
        self.refs.append(ref)
        mf = Frame(ref.seq, ref.ts + 0.01, self.main_img)
        return CropResult(crop_norm(self.main_img, box, pad), mf, False)


class FakeDetector:
    """Detects a white square anywhere in the (sub) frame."""

    def __init__(self):
        self.calls = []

    def detect(self, frame, regions=None):
        self.calls.append((frame.image.shape, regions))
        g = frame.image[..., 0]
        ys, xs = np.nonzero(g > 200)
        if len(xs) == 0:
            return []
        h, w = g.shape
        box = (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)
        if not overlaps_any(box, regions):
            return []
        return [Detection("person", 0.9, box)]


def _full_frame(seq, ts, x=None, w=1280, h=720):
    img = np.full((h, w, 3), 30, np.uint8)
    if x is not None:
        img[200:500, x:x + 150] = 255
        img[200:500:4, x:x + 150] = 120  # texture so sharpness > 0
    return Frame(seq, ts, img)


def test_loop_downscales_skips_static_and_crops_full_res():
    frames = [_full_frame(i + 1, i * 0.1) for i in range(10)]            # empty, static
    frames += [_full_frame(11 + i, 1.0 + i * 0.1, x=200 + 20 * i) for i in range(10)]  # moving
    frames += [_full_frame(21 + i, 2.0 + i * 0.1) for i in range(40)]    # gone
    det = FakeDetector()
    loop = PerceptionLoop(FakeGrabber(frames), det,
                          IouTracker(n_init=2, end_after_s=1.0),
                          FrameDiffMotionGate(full_every_s=0), sub_width=640)
    ups = list(loop)
    assert loop.frames == 60
    assert all(shape[1] == 640 for shape, _ in det.calls)  # detector saw the sub copy only
    assert loop.motion_skips > 0 and loop.detector_runs < loop.frames
    ended = [t for u in ups for t in u.ended]
    assert len(ended) == 1 and ended[0].state == TrackState.ENDED
    best = ended[0].best
    assert best is not None and "src=main" in best.reasons and "main=1280x720" in best.reasons
    # crop came from the 1280-wide frame: ~150 px wide object + padding
    assert best.crop.shape[1] > 150


def test_loop_with_dual_grabber_uses_crop_with_frame():
    subs = [Frame(i + 1, i * 0.1, _full_frame(0, 0, x=100 + 10 * i, w=640, h=360).image)
            for i in range(5)]
    main_img = np.full((1080, 1920, 3), 9, np.uint8)
    g = FakeDualGrabber(subs, main_img)
    loop = PerceptionLoop(g, FakeDetector(), IouTracker(n_init=1), gate=None)
    ups = list(loop)
    best = ups[-1].active[0].best
    assert "src=main" in best.reasons and any(r.startswith("skew_ms=") for r in best.reasons)
    assert int(best.crop[0, 0, 0]) == 9
    assert all(r is not None for r in g.refs)


# ---- detector region logic without weights --------------------------------

def _stub_detector(mode, results):
    d = YoloDetector.__new__(YoloDetector)
    d.region_mode, d.imgsz, d.min_region_ioa, d.crop_min_imgsz = mode, 640, 0.1, 160
    d.calls = []

    def _run(image, imgsz):
        d.calls.append((image.shape, imgsz))
        return results
    d._run = _run
    return d


def test_detector_filter_mode_drops_outside_regions():
    d = _stub_detector("filter", [("person", 0.9, (0.1, 0.1, 0.2, 0.2)),
                                  ("person", 0.8, (0.6, 0.6, 0.8, 0.8))])
    f = Frame(1, 0.0, np.zeros((360, 640, 3), np.uint8))
    out = d.detect(f, [(0.5, 0.5, 1.0, 1.0)])
    assert [x.conf for x in out] == [0.8]
    assert d.detect(f, []) == []  # skip
    assert len(d.detect(f, None)) == 2


def test_detector_crop_mode_maps_boxes_back():
    d = _stub_detector("crop", [("person", 0.9, (0.0, 0.0, 1.0, 1.0))])
    f = Frame(1, 0.0, np.zeros((360, 640, 3), np.uint8))
    out = d.detect(f, [(0.5, 0.5, 0.75, 1.0)])
    (shape, imgsz), = d.calls
    assert shape[:2] == (180, 160) and imgsz == 192
    b = out[0].box
    assert b == pytest.approx((0.5, 0.5, 0.75, 1.0), abs=1e-3)


@pytest.mark.skipif(find_weights() is None, reason="yolo11n.pt not present (models/ or $WORLDCAM_YOLO_WEIGHTS)")
def test_yolo_detector_smoke_normalized_boxes():
    pytest.importorskip("ultralytics")
    det = YoloDetector(device="cpu", warmup=False)
    f = Frame(1, 0.0, np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8))
    for x in det.detect(f):
        assert all(0.0 <= v <= 1.0 for v in x.box) and x.box[0] < x.box[2]
    assert det.detect(f, []) == []
