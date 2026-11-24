import time

from worldcam.geometry import box_foot, crop_norm, iou, point_in_polygon
from worldcam.sources import SyntheticSource
from worldcam.sources.dual import DualGrabber

DESK = [(0.1, 0.4), (0.6, 0.4), (0.6, 0.9), (0.1, 0.9)]


def test_point_in_polygon():
    assert point_in_polygon((0.3, 0.6), DESK)
    assert not point_in_polygon((0.8, 0.6), DESK)
    assert not point_in_polygon((0.3, 0.2), DESK)
    # concave L-shape: the notch must be outside
    L = [(0, 0), (1, 0), (1, 0.4), (0.4, 0.4), (0.4, 1), (0, 1)]
    assert point_in_polygon((0.2, 0.8), L)
    assert not point_in_polygon((0.8, 0.8), L)


def test_foot_point_is_bottom_center():
    x, y = box_foot((0.2, 0.1, 0.4, 0.7))
    assert abs(x - 0.3) < 1e-9 and y == 0.7


def test_iou():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert iou((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1)) == 0.0
    assert abs(iou((0, 0, 1, 1), (0.5, 0, 1.5, 1)) - (0.5 / 1.5)) < 1e-9


def test_crop_norm_is_resolution_independent_and_clamped():
    import numpy as np
    small = np.zeros((180, 320, 3), np.uint8)
    big = np.zeros((720, 1280, 3), np.uint8)
    box = (0.25, 0.25, 0.75, 0.75)
    cs, cb = crop_norm(small, box, pad=0.0), crop_norm(big, box, pad=0.0)
    assert cs.shape[:2] == (90, 160)
    assert cb.shape[:2] == (360, 640)
    edge = crop_norm(big, (0.9, 0.9, 1.0, 1.0), pad=1.0)  # padding must clamp
    assert edge is not None and edge.shape[0] <= 720 and edge.shape[1] <= 1280
    assert crop_norm(big, (0.5, 0.5, 0.5, 0.5), pad=0.0) is None


def test_dual_grabber_crops_from_mainstream_at_full_res():
    sub = SyntheticSource(width=320, height=180, fps=60)
    main = SyntheticSource(width=1280, height=720, fps=30)
    with DualGrabber(sub, main) as d:
        f = d.get(timeout=2.0)
        assert f is not None and f.shape == (320, 180)
        deadline = time.monotonic() + 2.0
        crop = None
        while time.monotonic() < deadline:
            crop = d.crop((0.0, 0.0, 0.5, 0.5), pad=0.0)
            if crop is not None and crop.shape[0] == 360:
                break
            time.sleep(0.02)
    assert crop is not None
    assert crop.shape[:2] == (360, 640), "crop must come from the 1280x720 mainstream"
