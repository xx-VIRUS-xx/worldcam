"""Two-phase capture benchmark.

Phase 1  raw read loop, no processing, no display.
         Slow here  => driver / backend / pixel format / low light (not your code).
Phase 2  threaded grabber + simulated per-frame work (--work-ms).
         Slow here but phase 1 fine => your loop structure.

  worldcam-bench --source 0 --width 1280 --height 720 --fps 30
  worldcam-bench --source rtsp://user:pass@192.168.1.10/stream1
  worldcam-bench --source synthetic --work-ms 80
"""
from __future__ import annotations

import argparse
import time

from ..capture import LatestFrameGrabber
from ..metrics import percentile
from ..sources import make_source


def phase1(args) -> None:
    src = make_source(args.source, args.width, args.height, args.fps)
    src.open()
    print("negotiated:", src.describe())
    n, fails, t0 = 0, 0, time.monotonic()
    gaps: list[float] = []
    last = t0
    while time.monotonic() - t0 < args.seconds:
        try:
            f = src.read()
        except EOFError:
            break
        now = time.monotonic()
        if f is None:
            fails += 1
            continue
        gaps.append((now - last) * 1000)
        last = now
        n += 1
    src.close()
    dur = max(1e-6, time.monotonic() - t0)
    print(
        f"[phase1 raw read] {n / dur:.1f} fps | frame gap ms p50={percentile(gaps, 50):.1f} "
        f"p95={percentile(gaps, 95):.1f} max={max(gaps, default=0):.1f} | failures={fails}"
    )


def phase2(args) -> None:
    src = make_source(args.source, args.width, args.height, args.fps)
    ages: list[float] = []
    t0 = time.monotonic()
    with LatestFrameGrabber(src) as g:
        last = 0
        while time.monotonic() - t0 < args.seconds:
            f = g.get(after_seq=last, timeout=1.0)
            if f is None:
                if g.eof:
                    break
                continue
            last = f.seq
            ages.append((time.monotonic() - f.ts) * 1000)
            time.sleep(args.work_ms / 1000)  # stand-in for detector/VLM cost
        dur = max(1e-6, time.monotonic() - t0)
        print(
            f"[phase2 grabber, work={args.work_ms:.0f}ms] consumed={g.frames_consumed / dur:.1f} fps "
            f"read={g.frames_read / dur:.1f} fps dropped={g.dropped} | "
            f"age at use ms p50={percentile(ages, 50):.1f} p95={percentile(ages, 95):.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--work-ms", type=float, default=50.0)
    args = ap.parse_args()
    phase1(args)
    phase2(args)


if __name__ == "__main__":
    main()
