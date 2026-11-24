"""Identity sanity bench on InsightFace's bundled sample images. No camera.

    PYTHONPATH=src python -m worldcam.identity.bench [--providers cpu,coreml] [--runs 20]

Measures (1) load time and latency per provider, plus CPU-vs-CoreML embedding
agreement, and (2) same-person vs different-person cosine distributions under
synthetic transforms of the 6 faces in sample image 't1' (+ the aligned
Tom_Hanks sample as a 7th identity). This is a sanity check of the pipeline
and thresholds, NOT a FAR/FRR evaluation: see docs/identity.md.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import statistics
import time

import numpy as np

from .faces import FaceAnalyzer, frontal_score, quality_score, sharpness_score


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))]


def _timeit(fn, runs):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), _pct(ts, 90)


def person_crops(img, faces):
    """A person-like crop around each face: 1.6x face width, face + 2 face heights below."""
    H, W = img.shape[:2]
    crops = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        fw, fh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        px1, px2 = int(max(0, cx - 0.8 * fw)), int(min(W, cx + 0.8 * fw))
        py1, py2 = int(max(0, y1 - 0.4 * fh)), int(min(H, y2 + 2.0 * fh))
        crops.append(img[py1:py2, px1:px2].copy())
    return crops


def transforms():
    import cv2

    def scale(s):
        return lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def blur(k):
        return lambda im: cv2.GaussianBlur(im, (k, k), 0)

    def bright(a, b=0):
        return lambda im: cv2.convertScaleAbs(im, alpha=a, beta=b)

    def jpeg(q):
        return lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)

    def rot(deg):
        def f(im):
            h, w = im.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
            return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return f

    def noise(sd):
        rng = np.random.default_rng(0)
        return lambda im: np.clip(im + rng.normal(0, sd, im.shape), 0, 255).astype(np.uint8)

    def upscale_small(s):  # low-res capture then upscaled (CCTV substream-ish)
        return lambda im: cv2.resize(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                     (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)

    def tight(im):  # tighter crop, top 60% only
        h, w = im.shape[:2]
        return im[: int(h * 0.6), int(w * 0.1): int(w * 0.9)]

    return {
        "orig": lambda im: im,
        "flip": lambda im: im[:, ::-1].copy(),
        "tight_crop": tight,
        "scale_2x": scale(2.0),
        "scale_0.6": scale(0.6),
        "scale_0.4": scale(0.4),
        "blur_k5": blur(5),
        "blur_k9": blur(9),
        "dark_0.4": bright(0.4),
        "bright_1.6": bright(1.6, 20),
        "jpeg_q15": jpeg(15),
        "rot_15": rot(15),
        "noise_12": noise(12),
        "lowres_0.3_up": upscale_small(0.3),
    }


def bench_latency(providers, runs):
    from insightface.data import get_image
    import cv2
    t1 = get_image("t1")
    results, embs = {}, {}
    for prov in providers:
        fa = FaceAnalyzer(providers=prov, save_crops=False)
        t0 = time.perf_counter()
        faces = fa.detect(t1)
        first = (time.perf_counter() - t0) * 1000
        crop = person_crops(t1, faces)[0]
        aligned = cv2.resize(crop[: crop.shape[1]], (112, 112))
        rec = fa._get_app().models["recognition"]
        det = fa._get_app().det_model
        r = {
            "load_s": fa.load_s,
            "first_call_ms": first,
            "frame_1280x886_6faces_ms": _timeit(lambda: fa.detect(t1), runs),
            "person_crop_1face_ms": _timeit(lambda: fa.detect(crop), runs),
            "detector_only_crop_ms": _timeit(lambda: det.detect(crop), runs),
            "embed_only_112_ms": _timeit(lambda: rec.get_feat(aligned), runs),
        }
        embs[prov] = [f.embedding for f in sorted(faces, key=lambda f: f.bbox[0])]
        results[prov] = r
        del fa, rec, det
        gc.collect()
    print("\n== latency (median, p90 ms; M2 Air) ==")
    for prov, r in results.items():
        print(f"[{prov}] load {r['load_s']:.1f}s  first call {r['first_call_ms']:.0f} ms")
        for k, v in r.items():
            if isinstance(v, tuple):
                print(f"   {k:28s} {v[0]:7.1f}  {v[1]:7.1f}")
    if len(embs) == 2:
        a, b = embs.values()
        sims = [float(np.dot(x, y)) for x, y in zip(a, b)]
        print(f"CPU vs CoreML embedding cosine per face: min {min(sims):.5f} mean {np.mean(sims):.5f}")


def bench_similarity(provider):
    from insightface.data import get_image
    import cv2
    fa = FaceAnalyzer(providers=provider, save_crops=False)
    t1 = get_image("t1")
    faces = sorted(fa.detect(t1), key=lambda f: f.bbox[0])
    crops = person_crops(t1, faces)
    tf = transforms()

    emb: dict[tuple[int, str], np.ndarray] = {}
    qual: dict[str, list[float]] = {k: [] for k in tf}
    missed: dict[str, int] = {k: 0 for k in tf}
    for pid, crop in enumerate(crops):
        for name, fn in tf.items():
            im = np.ascontiguousarray(fn(crop))
            fs = fa.detect(im)
            if not fs:
                missed[name] += 1
                continue
            f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb[(pid, name)] = f.embedding
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            px = min(x2 - x1, y2 - y1)
            sub = im[max(0, y1):y2, max(0, x1):x2]
            qual[name].append(quality_score(f.det_score, px, sharpness_score(sub), frontal_score(f.kps)))

    # 7th identity: Tom_Hanks sample is already an aligned 112x112 face
    th = get_image("Tom_Hanks_54745")
    extra = {"th_orig": fa.embed_aligned(th),
             "th_blur": fa.embed_aligned(cv2.GaussianBlur(th, (5, 5), 0)),
             "th_dark": fa.embed_aligned(cv2.convertScaleAbs(th, alpha=0.5))}
    for k, v in extra.items():
        emb[(len(crops), k)] = v

    keys = list(emb)
    same, diff = [], []
    by_tf: dict[str, list[float]] = {k: [] for k in tf}
    for a, b in itertools.combinations(keys, 2):
        s = float(np.dot(emb[a], emb[b]))
        (same if a[0] == b[0] else diff).append(s)
        if a[0] == b[0] and a[1] == "orig" and b[1] in by_tf:
            by_tf[b[1]].append(s)

    def summ(xs):
        return (f"n={len(xs):4d} min {min(xs):.3f} p5 {_pct(xs, 5):.3f} median {statistics.median(xs):.3f} "
                f"p95 {_pct(xs, 95):.3f} max {max(xs):.3f}")

    print(f"\n== cosine similarity ({provider}; {len(crops)} identities from t1 + Tom_Hanks) ==")
    print("same-person:", summ(same))
    print("diff-person:", summ(diff))
    for t in (0.30, 0.35, 0.40, 0.45, 0.50):
        print(f"  t={t:.2f}: same >= t {sum(s >= t for s in same) / len(same):6.1%}   "
              f"diff >= t {sum(s >= t for s in diff) / len(diff):6.1%}")
    print("\nper transform: cosine to the person's own 'orig' (min / median), quality (median), faces missed")
    for name in tf:
        xs, qs = by_tf[name], qual[name]
        if name == "orig":
            print(f"  {name:14s}   ref                 q {statistics.median(qs):.2f}  missed {missed[name]}")
            continue
        s = f"{min(xs):.3f} / {statistics.median(xs):.3f}" if xs else "   n/a      "
        q = f"{statistics.median(qs):.2f}" if qs else "n/a"
        print(f"  {name:14s} {s}   q {q}  missed {missed[name]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="cpu,coreml")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args(argv)
    if not a.skip_latency:
        bench_latency(a.providers.split(","), a.runs)
    bench_similarity("cpu")


if __name__ == "__main__":
    main()
