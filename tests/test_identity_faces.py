"""FaceAnalyzer: pure helpers always; model tests skip if buffalo_l can't load.
Uses InsightFace's bundled sample image 't1' (6 faces). No camera."""
import numpy as np
import pytest

from worldcam.geometry import crop_norm, iou
from worldcam.identity.faces import (FaceAnalyzer, crop_region, frontal_score, l2_normalize,
                                     map_box_to_frame, quality_score)
from worldcam.pipeline import BestSnapshot, Track, TrackState


# ---- pure helpers ----------------------------------------------------------

def test_crop_region_matches_crop_norm_pixels():
    img = np.zeros((900, 1600, 3), np.uint8)
    for box in [(0.2, 0.3, 0.5, 0.9), (0.0, 0.0, 0.3, 0.3), (0.8, 0.7, 1.0, 1.0)]:
        r = crop_region(box, 0.1)
        c = crop_norm(img, box, pad=0.1)
        assert abs(c.shape[1] - (r[2] - r[0]) * 1600) <= 1.5
        assert abs(c.shape[0] - (r[3] - r[1]) * 900) <= 1.5
        assert 0.0 <= r[0] <= r[2] <= 1.0 and 0.0 <= r[1] <= r[3] <= 1.0


def test_map_box_to_frame():
    # crop 200x400 px covering frame region x 0.25..0.75, y 0.0..0.5
    b = map_box_to_frame((50, 100, 150, 200), (400, 200, 3), (0.25, 0.0, 0.75, 0.5))
    assert b == pytest.approx((0.375, 0.125, 0.625, 0.25))
    # out-of-crop pixels are clamped to the crop
    b = map_box_to_frame((-10, -10, 300, 500), (400, 200, 3), (0.25, 0.0, 0.75, 0.5))
    assert b == pytest.approx((0.25, 0.0, 0.75, 0.5))


def test_l2_normalize():
    v = l2_normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])


def test_frontal_score_prefers_frontal():
    frontal = np.array([[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]])  # ArcFace template
    turned = frontal.copy()
    turned[2, 0] += 20                         # nose shifted toward one side (yaw)
    assert frontal_score(frontal) > 0.9
    assert frontal_score(turned) < 0.5
    assert frontal_score(None) == 0.5
    # roll alone must not count as non-frontal
    th = np.radians(25)
    rot = frontal @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    assert frontal_score(rot) == pytest.approx(frontal_score(frontal), abs=1e-6)


def test_quality_score_bounds():
    assert quality_score(0.9, 200, 1.0, 1.0) == pytest.approx(1.0)
    assert quality_score(0.9, 56, 1.0, 1.0) == pytest.approx(0.5)
    assert quality_score(0.4, 200, 1.0, 1.0) == 0.0


# ---- model tests -----------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    try:
        from insightface.data import get_image  # noqa: F401
        fa = FaceAnalyzer(providers="cpu", save_crops=False)
        fa._get_app()
    except Exception as e:  # model not downloadable / onnxruntime missing
        pytest.skip(f"buffalo_l not loadable: {e}")
    return fa


@pytest.fixture(scope="module")
def t1():
    from insightface.data import get_image
    return get_image("t1")


def _track(box, crop, tid=7, ts=123.0):
    return Track(tid, "person", TrackState.ACTIVE, box, ts, ts,
                 best=BestSnapshot(seq=3, ts=ts, box=box, score=0.9, crop=crop))


def _person_box(t1, raw):
    """Normalized person-like box around a face found in the full frame."""
    H, W = t1.shape[:2]
    x1, y1, x2, y2 = raw.bbox
    fw, fh = x2 - x1, y2 - y1
    return (max(0, x1 - 0.6 * fw) / W, max(0, y1 - 0.4 * fh) / H,
            min(W, x2 + 0.6 * fw) / W, min(H, y2 + 2.0 * fh) / H)


def test_embeddings_are_unit_512(analyzer, t1):
    faces = analyzer.detect(t1)
    assert len(faces) == 6
    for f in faces:
        assert f.embedding.shape == (512,)
        assert np.linalg.norm(f.embedding) == pytest.approx(1.0, abs=1e-6)


def test_analyze_maps_box_back_to_frame(analyzer, t1, tmp_path):
    H, W = t1.shape[:2]
    full = sorted(analyzer.detect(t1), key=lambda f: f.bbox[0])
    target = full[2]
    box = _person_box(t1, target)
    crop = crop_norm(t1, box, pad=0.1)        # what the dual grabber hands the tracker
    analyzer.snapshot_dir, analyzer.save_crops = str(tmp_path), True
    try:
        obs = analyzer.analyze(crop, _track(box, crop))
    finally:
        analyzer.save_crops = False
    assert obs, "face not found in person crop"
    o = obs[0]
    want = (target.bbox[0] / W, target.bbox[1] / H, target.bbox[2] / W, target.bbox[3] / H)
    assert iou(o.box, want) > 0.85
    assert o.track_id == 7 and o.ts == 123.0 and 0.0 < o.quality <= 1.0
    assert len(o.embedding) == 512 and np.linalg.norm(o.embedding) == pytest.approx(1.0, abs=1e-6)
    assert float(np.dot(o.embedding, target.embedding)) > 0.8   # same face, same identity
    assert o.crop_path and (tmp_path / o.crop_path.split("/")[-1]).exists()


def test_analyze_uses_track_crop_when_none_given(analyzer, t1):
    box = _person_box(t1, analyzer.detect(t1)[0])
    crop = crop_norm(t1, box, pad=0.1)
    assert analyzer.analyze(None, _track(box, crop))


def test_tiny_faces_are_dropped(analyzer, t1):
    import cv2
    box = _person_box(t1, analyzer.detect(t1)[0])
    crop = cv2.resize(crop_norm(t1, box, pad=0.1), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    assert analyzer.analyze(crop, _track(box, crop)) == []


def test_blur_lowers_quality(analyzer, t1):
    import cv2
    box = _person_box(t1, analyzer.detect(t1)[1])
    crop = crop_norm(t1, box, pad=0.1)
    sharp = analyzer.analyze(crop, _track(box, crop))
    blurred = cv2.GaussianBlur(crop, (9, 9), 0)
    soft = analyzer.analyze(blurred, _track(box, blurred))
    assert sharp and (not soft or soft[0].quality < sharp[0].quality)


def test_no_crop_no_faces(analyzer):
    t = Track(1, "person", TrackState.NEW, (0, 0, 1, 1), 0.0, 0.0)
    assert analyzer.analyze(None, t) == []
