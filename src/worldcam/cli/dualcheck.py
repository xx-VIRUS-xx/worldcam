"""Dual-stream (sub for pacing, main for crops) timing check.

Paces on substream frames via DualGrabber.get(), simulates detector work
(--work-ms), then crops a fixed normalized box from the mainstream and records

    offset_ms = (main_frame.ts - sub_frame.ts) * 1000

Positive = the main frame arrived after the sub frame it is paired with.
Frame.ts is software arrival time on each connection, so this measures
arrival skew, not sensor/content skew (see docs/dual.md).

  python -m worldcam.cli.dualcheck --main rtsp://127.0.0.1:8654/main \\
      --sub rtsp://127.0.0.1:8654/sub --seconds 30 --work-ms 80
  python -m worldcam.cli.dualcheck --main 1 --sub 0 --seconds 30 --work-ms 80
  python -m worldcam.cli.dualcheck --main synthetic --sub synthetic --seconds 10
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..metrics import percentile
from ..sources import make_source
from ..sources.dual import DualGrabber

BOX = (0.25, 0.25, 0.75, 0.75)


def _stats(name: str, v: list[float], signed: bool = False) -> str:
    lo = f"min={min(v, default=0.0):.1f} p5={percentile(v, 5):.1f} " if signed else ""
    return (
        f"{name} n={len(v)} {lo}p50={percentile(v, 50):.1f} p95={percentile(v, 95):.1f} "
        f"max={max(v, default=0.0):.1f}"
    )


class _ClipIndex:
    """Content-skew estimator for a looping reference clip (measurement only).

    Holds one tiny grayscale thumbnail per clip frame (bounded by clip length)
    and maps a received frame to its nearest clip frame index, so
    (main_idx - sub_idx) / fps is the CONTENT offset between the two frames,
    independent of software arrival time.
    """

    SIZE = (64, 36)

    def __init__(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open ref clip {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        thumbs = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            thumbs.append(self._thumb(img))
        cap.release()
        self.thumbs = np.stack(thumbs)
        self.n = len(thumbs)

    def _thumb(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(img, self.SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return g.astype(np.float32)

    def index(self, img: np.ndarray) -> int:
        return self.index_thumb(self._thumb(img))

    def index_thumb(self, t: np.ndarray) -> int:
        d = np.abs(self.thumbs - t).mean(axis=(1, 2))
        return int(np.argmin(d))

    def skew_ms(self, main_t: np.ndarray, sub_t: np.ndarray) -> float:
        k = (self.index_thumb(main_t) - self.index_thumb(sub_t)) % self.n
        if k >= self.n // 2:
            k -= self.n
        return k * 1000.0 / self.fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True, help="mainstream spec (index, rtsp://, file, synthetic)")
    ap.add_argument("--sub", required=True, help="substream spec")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--work-ms", type=float, default=0.0)
    ap.add_argument("--ring", type=int, default=1,
                    help="main frames remembered for ts matching (1 = newest only)")
    ap.add_argument("--max-skew-ms", type=float, default=None,
                    help="fall back to the sub frame if main is further off than this")
    ap.add_argument("--main-size", default="1280x720", help="requested WxH for camera-index main")
    ap.add_argument("--sub-size", default="640x360", help="requested WxH for camera-index sub")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="seconds after start excluded from stats (connection setup)")
    ap.add_argument("--ref-clip", default=None,
                    help="looping source clip; enables content-skew estimate by frame matching")
    args = ap.parse_args()

    mw, mh = (int(x) for x in args.main_size.split("x"))
    sw, sh = (int(x) for x in args.sub_size.split("x"))
    main_src = make_source(args.main, mw, mh, args.fps)
    sub_src = make_source(args.sub, sw, sh, args.fps)
    max_skew = None if args.max_skew_ms is None else args.max_skew_ms / 1000

    signed: list[float] = []
    absolute: list[float] = []
    sub_age: list[float] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []  # thumbnails; matched after the loop
    clip = _ClipIndex(args.ref_clip) if args.ref_clip else None
    n_crops = fallbacks = 0
    shapes: set[tuple[int, int]] = set()
    with DualGrabber(sub_src, main_src, main_ring=args.ring) as d:
        print("sub :", sub_src.describe())
        print("main:", main_src.describe())
        tw = time.monotonic()
        last = 0
        while time.monotonic() - tw < args.warmup:
            f = d.get(after_seq=last, timeout=1.0)
            if f is not None:
                last = f.seq
            elif d.eof:
                break
        sub0, main0 = d.sub.frames_read, d.main.frames_read
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            f = d.get(after_seq=last, timeout=1.0)
            if f is None:
                if d.eof:
                    break
                continue
            last = f.seq
            sub_age.append((time.monotonic() - f.ts) * 1000)
            if args.work_ms > 0:
                time.sleep(args.work_ms / 1000)  # stand-in for detector cost
            r = d.crop_with_frame(BOX, pad=0.0, ref=f, max_skew=max_skew)
            if r.frame is None:
                continue
            n_crops += 1
            if r.image is not None:
                shapes.add(r.frame.shape)
            if r.fallback:
                fallbacks += 1
                continue
            off = (r.frame.ts - f.ts) * 1000
            signed.append(off)
            absolute.append(abs(off))
            if clip is not None:
                pairs.append((clip._thumb(r.frame.image), clip._thumb(f.image)))
        dur = max(1e-6, time.monotonic() - t0)
        print(
            f"[dualcheck work={args.work_ms:.0f}ms ring={args.ring} {args.seconds:.0f}s] "
            f"crops={n_crops} fallbacks={fallbacks} | "
            f"sub read={(d.sub.frames_read - sub0) / dur:.1f} fps "
            f"main read={(d.main.frames_read - main0) / dur:.1f} fps | "
            f"crop-source shapes={sorted(shapes)}"
        )
        print("  " + _stats("ts offset ms (signed)", signed, signed=True))
        print("  " + _stats("ts offset ms |abs|   ", absolute))
        if clip is not None:
            content = [clip.skew_ms(m, s) for m, s in pairs]
            print("  " + _stats("content skew ms (sgn)", content, signed=True))
            print("  " + _stats("content skew |abs|   ", [abs(x) for x in content]))
        print("  " + _stats("sub age at get ms    ", sub_age))


if __name__ == "__main__":
    main()
