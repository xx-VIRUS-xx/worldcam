"""Appearance embeddings for animals and objects (the face analogue for non-person entities).

Two interchangeable embedders behind `EntityEmbedder`:

  classical  HSV colour histogram + HOG + uniform LBP on a 64x64 normalised crop
             (331 dims, no model file, ~1 ms). Weak on texture-less objects.
  cnn        MobileNetV2 (ImageNet, ONNX, 14 MB) pooled features (1280 dims, ~15 ms
             on the M2 CPU). Weights live in models/ (never committed) and are
             fetched with `python -m worldcam.perception.reid --fetch`.

`EntityEmbedder("auto")` uses the CNN when its weights are present, else the
classical features. Output is float32, L2-normalised, so cosine == dot product,
the same convention as faces. Scores are only comparable within one method:
`reid_thresholds(method)` returns the measured match/ask thresholds (docs/reid.md).

Honest limits: two identical products (two white mugs) look alike to any
appearance embedding. That is an ambiguity the agent should ask about, not a
case to guess; ambiguity/entities.py turns the in-between scores into
`low_confidence` / `conflict` doubts instead of silently merging.
"""
from __future__ import annotations

import argparse
import glob
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

CNN_MODEL_URL = ("https://github.com/onnx/models/raw/main/validated/vision/classification/"
                 "mobilenet/model/mobilenetv2-12.onnx")
CNN_FEATURE_TENSOR = "472"        # pooled 1280-d features before the classifier Gemm
CLASSICAL_DIM = 511
CNN_DIM = 1280
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class ReidThresholds:
    match_t: float      # >= : same entity, no question
    ask_t: float        # in [ask_t, match_t): probably the same, ask (low_confidence)
    cluster_t: float    # store-level join threshold for unlabeled clusters
    conflict_margin: float = 0.03


# Filled from the measurements in docs/reid.md (re-run `--bench` to refresh).
THRESHOLDS = {
    "cnn": ReidThresholds(match_t=0.90, ask_t=0.80, cluster_t=0.88),
    "classical": ReidThresholds(match_t=0.95, ask_t=0.88, cluster_t=0.93),
}


def reid_thresholds(method: str) -> ReidThresholds:
    return THRESHOLDS[method]


def default_model_dir() -> Path:
    return Path(os.environ.get("WORLDCAM_MODEL_DIR", "models"))


def cnn_model_path(model_dir: Optional[Path] = None) -> Path:
    return (model_dir or default_model_dir()) / "mobilenetv2-12-feat.onnx"


def fetch_cnn_model(model_dir: Optional[Path] = None) -> Path:
    """Download the ONNX classifier and trim it to a feature extractor (needs `onnx` once)."""
    import urllib.request
    d = model_dir or default_model_dir()
    d.mkdir(parents=True, exist_ok=True)
    raw, out = d / "mobilenetv2-12.onnx", cnn_model_path(d)
    if out.exists():
        return out
    if not raw.exists():
        urllib.request.urlretrieve(CNN_MODEL_URL, raw)
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(str(raw))
    del m.graph.output[:]      # expose the pooled features instead of the 1000-class logits
    m.graph.output.append(helper.make_tensor_value_info(CNN_FEATURE_TENSOR, TensorProto.FLOAT, [None, CNN_DIM]))
    onnx.save(m, str(out))
    return out


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


class EntityEmbedder:
    """Thread-safe crop -> L2-normalised embedding."""

    def __init__(self, method: str = "auto", model_dir: Optional[Path] = None):
        if method not in ("auto", "classical", "cnn"):
            raise ValueError(f"bad method {method!r}")
        path = cnn_model_path(model_dir)
        if method == "auto":
            method = "cnn" if path.exists() else "classical"
        self.method = method
        self.dim = CNN_DIM if method == "cnn" else CLASSICAL_DIM
        self._lock = threading.Lock()
        self._sess = None
        if method == "cnn":
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run `python -m worldcam.perception.reid --fetch`")
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2     # shared 8 GB laptop: stay small
            self._sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self._in = self._sess.get_inputs()[0].name

    @property
    def thresholds(self) -> ReidThresholds:
        return THRESHOLDS[self.method]

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.ndim != 3 or min(crop_bgr.shape[:2]) < 8:
            raise ValueError("crop too small or not a BGR image")
        return self._cnn(crop_bgr) if self.method == "cnn" else self._classical(crop_bgr)

    def embed_many(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [self.embed(c) for c in crops]

    # ---------------------------------------------------------------- features
    def _cnn(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        x = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        with self._lock:
            f = self._sess.run(None, {self._in: np.ascontiguousarray(x)})[0]
        return _l2(f.reshape(-1))

    def _classical(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        g = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
        # Centre 75 % only: borders are mostly background for animals / objects.
        c = hsv[8:56, 8:56]
        hist = cv2.calcHist([c], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
        hist = np.sqrt(hist / max(hist.sum(), 1.0))                     # Hellinger
        gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
        hog = _hog(gray)
        lbp = _lbp_hist(gray)
        return _l2(np.concatenate([_l2(hist) * 1.0, _l2(hog) * 0.8, _l2(lbp) * 0.5]))


def _hog(gray: np.ndarray) -> np.ndarray:
    """HOG on a 64x64 image: 4x4 cells of 16 px, 9 unsigned bins, 2x2 overlapping blocks (324 dims).
    (OpenCV 5 dropped cv2.HOGDescriptor, so this is plain numpy.)"""
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    b = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * 9).astype(np.int32), 8)
    cells = np.zeros((4, 4, 9), np.float32)
    for cy in range(4):
        for cx in range(4):
            sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
            cells[cy, cx] = np.bincount(b[sl].reshape(-1), weights=mag[sl].reshape(-1), minlength=9)
    blocks = [cells[y:y + 2, x:x + 2].reshape(-1) for y in range(3) for x in range(3)]
    return np.concatenate([bl / (np.linalg.norm(bl) + 1e-6) for bl in blocks]).astype(np.float32)


_UNIFORM = None


def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    """59-bin uniform LBP histogram (8 neighbours, radius 1)."""
    global _UNIFORM
    if _UNIFORM is None:
        table, nxt = np.zeros(256, np.int32), 0
        for v in range(256):
            bits = [(v >> i) & 1 for i in range(8)]
            trans = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
            if trans <= 2:
                table[v], nxt = nxt, nxt + 1
            else:
                table[v] = 58
        _UNIFORM = table
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.int32)
    for i, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))):
        n = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= (n >= c).astype(np.int32) << i
    h = np.bincount(_UNIFORM[code].reshape(-1), minlength=59).astype(np.float32)
    return np.sqrt(h / max(h.sum(), 1.0))


# --------------------------------------------------------------------- bench
def _views(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A plausible re-sighting of the same instance: crop jitter, scale, brightness, blur."""
    import cv2
    h, w = img.shape[:2]
    dx, dy = int(w * rng.uniform(0, 0.08)), int(h * rng.uniform(0, 0.08))
    ex, ey = int(w * rng.uniform(0, 0.08)), int(h * rng.uniform(0, 0.08))
    v = img[dy:h - ey, dx:w - ex]
    s = rng.uniform(0.5, 1.2)
    v = cv2.resize(v, (max(16, int(v.shape[1] * s)), max(16, int(v.shape[0] * s))))
    v = np.clip(v.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        v = cv2.GaussianBlur(v, (0, 0), rng.uniform(0.8, 2.0))
    return v


def bench(crop_dir: str, methods: Sequence[str] = ("classical", "cnn"), seed: int = 0) -> dict:
    import cv2
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(crop_dir, "*.jpg")))
    imgs = [cv2.imread(f) for f in files]
    labels = [os.path.basename(f).split("__")[0] for f in files]
    A = [_views(i, rng) for i in imgs]
    B = [_views(i, rng) for i in imgs]
    out = {"n": len(files), "labels": sorted(set(labels))}
    import time
    for m in methods:
        try:
            emb = EntityEmbedder(m)
        except FileNotFoundError as e:
            out[m] = {"error": str(e)}
            continue
        t0 = time.perf_counter()
        ea = np.stack(emb.embed_many(A))
        ms = (time.perf_counter() - t0) / len(A) * 1000
        eb = np.stack(emb.embed_many(B))
        sim = ea @ eb.T
        n = len(files)
        same = np.diag(sim)
        ds, dd = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    (ds if labels[i] == labels[j] else dd).append(sim[i, j])
        ds, dd = np.array(ds), np.array(dd)
        hard = ds if len(ds) else dd

        def pct(a, p):
            return round(float(np.percentile(a, p)), 3)
        # AUC: P(same-instance score > different-instance same-label score)
        auc = float((same[:, None] > hard[None, :]).mean()) if len(hard) else float("nan")
        out[m] = {"dim": emb.dim, "ms_per_crop": round(ms, 1),
                  "same": {"mean": pct(same, 50), "p5": pct(same, 5), "min": pct(same, 0)},
                  "diff_same_label": {"p50": pct(ds, 50), "p95": pct(ds, 95), "p99": pct(ds, 99), "max": pct(ds, 100)} if len(ds) else None,
                  "diff_other_label": {"p50": pct(dd, 50), "p95": pct(dd, 95), "max": pct(dd, 100)} if len(dd) else None,
                  "auc_same_vs_samelabel_diff": round(auc, 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="entity re-id embedder tools")
    ap.add_argument("--fetch", action="store_true", help="download + trim the MobileNetV2 feature model")
    ap.add_argument("--bench", metavar="CROP_DIR", help="dir of crops named <label>__<id>.jpg")
    a = ap.parse_args()
    if a.fetch:
        print(fetch_cnn_model())
    if a.bench:
        import json
        print(json.dumps(bench(a.bench), indent=2))


if __name__ == "__main__":
    main()
