from collections import Counter

import cv2
import numpy as np

from worldcam.perception.tracker import IouTracker, sharpness
from worldcam.pipeline import Detection, TrackState
from worldcam.types import Frame


def _f(seq, ts, img=None):
    return Frame(seq, ts, img if img is not None else np.zeros((360, 640, 3), np.uint8))


def _person(box, conf=0.9):
    return Detection("person", conf, box)


def _run(tr, script, dt=0.1, regions=None):
    """script: list of detection lists, one per frame. Returns all updates."""
    out = []
    for i, dets in enumerate(script):
        out.append(tr.update(_f(i + 1, i * dt), dets, observed_regions=regions))
    return out


B = (0.4, 0.3, 0.6, 0.9)


def test_new_to_active_after_n_hits():
    tr = IouTracker(n_init=3)
    ups = _run(tr, [[_person(B)]] * 3)
    assert ups[0].active[0].state == TrackState.NEW
    assert ups[1].active[0].state == TrackState.NEW
    assert ups[2].active[0].state == TrackState.ACTIVE
    assert len({u.active[0].track_id for u in ups}) == 1


def test_lost_then_ended_exactly_once():
    tr = IouTracker(n_init=2, lost_misses=2, end_after_s=1.0)
    script = [[_person(B)]] * 3 + [[]] * 30
    ups = _run(tr, script)
    states = [u.active[0].state if u.active else None for u in ups]
    assert TrackState.LOST in states
    ended = [t for u in ups for t in u.ended]
    assert len(ended) == 1
    assert ended[0].state == TrackState.ENDED and ended[0].track_id == 1
    assert tr.live == 0 and tr.stats["ended"] == 1


def test_lost_track_recovers_same_id():
    tr = IouTracker(n_init=2, lost_misses=1, end_after_s=5.0)
    ups = _run(tr, [[_person(B)]] * 3 + [[]] * 3 + [[_person(B)]])
    assert ups[4].active[0].state == TrackState.LOST
    assert ups[-1].active[0].state == TrackState.ACTIVE
    assert ups[-1].active[0].track_id == 1


def test_tentative_track_dropped_silently():
    tr = IouTracker(n_init=3)
    ups = _run(tr, [[_person(B)], []])
    assert ups[-1].active == [] and ups[-1].ended == []
    assert tr.stats["dropped_tentative"] == 1


def test_stationary_and_back_to_active():
    tr = IouTracker(n_init=2, still_after_s=2.0, still_eps=0.02)
    # 4 s still at 10 fps, then a big move
    script = [[_person(B)]] * 40 + [[_person((0.45, 0.3, 0.65, 0.9))]]
    ups = _run(tr, script)
    assert ups[15].active[0].state == TrackState.ACTIVE
    assert ups[30].active[0].state == TrackState.STATIONARY
    assert ups[30].active[0].stationary_since is not None
    last = ups[-1].active[0]
    assert last.state == TrackState.ACTIVE and last.stationary_since is None
    assert last.track_id == 1


def test_unobserved_tracks_are_not_missed():
    """Detector skipped (observed_regions=[]) or looked elsewhere: the track
    must not go LOST."""
    tr = IouTracker(n_init=2, lost_misses=1, end_after_s=0.5)
    _run(tr, [[_person(B)]] * 3)
    for i in range(20):
        u = tr.update(_f(10 + i, 1.0 + i * 0.1), [], observed_regions=[])
        assert u.active[0].state == TrackState.ACTIVE
    u = tr.update(_f(40, 4.0), [], observed_regions=[(0.0, 0.0, 0.2, 0.2)])
    assert u.active[0].state == TrackState.ACTIVE
    u = tr.update(_f(41, 4.1), [], observed_regions=None)  # full frame, missing
    assert u.active[0].state == TrackState.LOST


def test_two_people_keep_ids_and_labels_dont_cross():
    tr = IouTracker(n_init=1)
    a, b = (0.1, 0.2, 0.3, 0.9), (0.6, 0.2, 0.8, 0.9)
    script = []
    for i in range(10):
        dx = 0.01 * i
        script.append([_person((a[0] + dx, a[1], a[2] + dx, a[3])),
                       _person((b[0] - dx, b[1], b[2] - dx, b[3])),
                       Detection("cup", 0.8, (0.6, 0.2, 0.8, 0.9))])
    ups = _run(tr, script)
    ids = Counter((t.label, t.track_id) for u in ups for t in u.active)
    assert len(ids) == 3 and all(n == 10 for n in ids.values())


def test_boxes_stay_normalized():
    tr = IouTracker(n_init=1)
    rng = np.random.default_rng(0)
    for i in range(50):
        dets = []
        for _ in range(3):
            x, y = rng.uniform(0, 0.8, 2)
            dets.append(_person((x, y, x + 0.2, y + 0.2), float(rng.uniform(0.3, 1))))
        u = tr.update(_f(i + 1, i * 0.1), dets)
        for t in list(u.active) + list(u.ended):
            assert all(0.0 <= v <= 1.0 for v in t.box)
            assert t.box[0] < t.box[2] and t.box[1] < t.box[3]
            if t.best is not None:
                assert all(0.0 <= v <= 1.0 for v in t.best.box)


# ---- best snapshot -----------------------------------------------------

def _textured(h=360, w=640, blur=0):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


def test_sharpness_orders_blur():
    box = (0.2, 0.2, 0.6, 0.8)
    assert sharpness(_textured(), box) > sharpness(_textured(blur=15), box) > 0


def test_best_snapshot_prefers_sharp_large_confident():
    tr = IouTracker(n_init=1, best_margin=1.0)
    sharp, blurry = _textured(), _textured(blur=21)
    box_small, box_big = (0.4, 0.3, 0.6, 0.9), (0.38, 0.28, 0.62, 0.92)
    # frame 1: blurry, frame 2: sharp + larger, frame 3: sharp but low conf
    tr.update(_f(1, 0.0, blurry), [_person(box_small, 0.9)])
    u = tr.update(_f(2, 0.1, sharp), [_person(box_big, 0.9)])
    best = u.active[0].best
    assert best.seq == 2
    assert "sharpest" in best.reasons and "largest" in best.reasons
    u = tr.update(_f(3, 0.2, sharp), [_person(box_big, 0.4)])
    assert u.active[0].best.seq == 2  # lower conf did not win


def test_best_snapshot_crop_from_main_via_crop_fn():
    tr = IouTracker(n_init=1)
    main = np.zeros((720, 1280, 3), np.uint8)
    main[:] = 7
    calls = []

    def crop_fn(box):
        calls.append(box)
        return main[0:100, 0:50].copy(), ("src=main", "main=1280x720")

    u = tr.update(_f(1, 0.0, _textured()), [_person(B)], crop_fn=crop_fn)
    best = u.active[0].best
    assert calls == [B]
    assert best.crop.shape == (100, 50, 3) and int(best.crop[0, 0, 0]) == 7
    assert "src=main" in best.reasons


def test_best_snapshot_without_main_records_fallback():
    tr = IouTracker(n_init=1)
    u = tr.update(_f(1, 0.0, _textured()), [_person(B)])
    best = u.active[0].best
    assert "src=sub" in best.reasons and "main_unavailable" in best.reasons
    assert best.crop.shape[1] < 640  # cut from the 640-wide sub frame
    assert tr.stats["best_fallback_sub"] == 1
