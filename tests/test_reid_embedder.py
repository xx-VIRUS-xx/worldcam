"""EntityEmbedder: shape, normalisation, determinism, and basic separation on synthetic crops."""
import numpy as np
import pytest

from worldcam.perception.reid import (CLASSICAL_DIM, CNN_DIM, EntityEmbedder, THRESHOLDS, _views,
                                      cnn_model_path, reid_thresholds)


def textured(seed, h=90, w=70):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, 3)
    img = np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    import cv2
    for _ in range(6):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, 20)),
                   tuple(int(x) for x in rng.integers(0, 255, 3)), -1)
    return img


def test_classical_embedding_is_unit_float32_and_deterministic():
    e = EntityEmbedder("classical")
    a = e.embed(textured(1))
    assert a.shape == (CLASSICAL_DIM,) and a.dtype == np.float32 and e.dim == CLASSICAL_DIM
    assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(a, e.embed(textured(1)))


def test_classical_separates_same_from_different_instances():
    e = EntityEmbedder("classical")
    rng = np.random.default_rng(0)
    same, diff = [], []
    imgs = [textured(i) for i in range(12)]
    for i, im in enumerate(imgs):
        a, b = e.embed(_views(im, rng)), e.embed(_views(im, rng))
        same.append(float(a @ b))
        diff.append(float(a @ e.embed(imgs[(i + 1) % 12])))
    assert np.median(same) > np.median(diff) + 0.15


def test_rejects_tiny_or_wrong_crops():
    e = EntityEmbedder("classical")
    with pytest.raises(ValueError):
        e.embed(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        e.embed(np.zeros((40, 40), np.uint8))


def test_thresholds_are_ordered():
    for m in THRESHOLDS:
        t = reid_thresholds(m)
        assert 0 < t.ask_t < t.cluster_t <= t.match_t < 1


@pytest.mark.skipif(not cnn_model_path().exists(), reason="models/mobilenetv2-12-feat.onnx not fetched")
def test_cnn_embedding_unit_and_separates():
    e = EntityEmbedder("cnn")
    assert e.dim == CNN_DIM
    a = e.embed(textured(3))
    assert a.shape == (CNN_DIM,) and float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-5)
    img = textured(3)
    b = e.embed(np.clip(img[2:-2, 2:-2].astype(np.float32) * 1.1, 0, 255).astype(np.uint8))  # mild re-sighting
    c = e.embed(textured(4))
    assert float(a @ b) > float(a @ c)
