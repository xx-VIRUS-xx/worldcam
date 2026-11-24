import numpy as np

from worldcam.perception.motion import FULL_FRAME, FrameDiffMotionGate, merge_boxes
from worldcam.types import Frame


def _frame(seq, ts, square_x=None, w=640, h=360, size=60):
    img = np.full((h, w, 3), 40, np.uint8)
    if square_x is not None:
        img[150:150 + size, square_x:square_x + size] = 230
    return Frame(seq, ts, img)


def _valid(b):
    return 0.0 <= b[0] < b[2] <= 1.0 and 0.0 <= b[1] < b[3] <= 1.0


def test_first_frame_is_full_then_static_scene_is_empty():
    g = FrameDiffMotionGate(full_every_s=0)
    assert list(g.regions(_frame(1, 0.0))) == [FULL_FRAME]
    assert g.last_reason == "init"
    for i in range(2, 20):
        assert list(g.regions(_frame(i, i * 0.1))) == []
    assert g.last_reason == "none"


def test_moving_square_gives_region_around_it():
    g = FrameDiffMotionGate(full_every_s=0)
    g.regions(_frame(1, 0.0))
    for i in range(5):
        g.regions(_frame(2 + i, 0.1 * (i + 1)))  # static background
    regs = list(g.regions(_frame(10, 1.0, square_x=300)))
    assert len(regs) == 1 and g.last_reason == "motion"
    r = regs[0]
    assert _valid(r)
    cx, cy = (300 + 30) / 640, (150 + 30) / 360
    assert r[0] < cx < r[2] and r[1] < cy < r[3]
    assert (r[2] - r[0]) * (r[3] - r[1]) < 0.25  # local, not the whole frame


def test_min_area_filters_small_blobs():
    g = FrameDiffMotionGate(full_every_s=0, min_area=0.05)
    g.regions(_frame(1, 0.0))
    g.regions(_frame(2, 0.1))
    assert list(g.regions(_frame(3, 0.2, square_x=100, size=20))) == []


def test_periodic_full_frame_recheck():
    g = FrameDiffMotionGate(full_every_s=5.0)
    assert g.regions(_frame(1, 100.0)) == [FULL_FRAME]
    assert g.regions(_frame(2, 101.0)) == []
    assert g.regions(_frame(3, 104.9)) == []
    assert g.regions(_frame(4, 105.0)) == [FULL_FRAME]
    assert g.last_reason == "periodic"
    assert g.regions(_frame(5, 106.0)) == []


def test_global_change_returns_full_frame():
    g = FrameDiffMotionGate(full_every_s=0)
    g.regions(_frame(1, 0.0))
    bright = Frame(2, 0.1, np.full((360, 640, 3), 200, np.uint8))
    assert g.regions(bright) == [FULL_FRAME]
    assert g.last_reason == "global"


def test_merge_boxes():
    out = merge_boxes([(0.1, 0.1, 0.2, 0.2), (0.15, 0.15, 0.3, 0.3), (0.7, 0.7, 0.8, 0.8)])
    assert sorted(out) == [(0.1, 0.1, 0.3, 0.3), (0.7, 0.7, 0.8, 0.8)]
    assert merge_boxes([(0.1, 0.1, 0.2, 0.2), (0.21, 0.1, 0.3, 0.2)], gap=0.02) == [(0.1, 0.1, 0.3, 0.2)]
