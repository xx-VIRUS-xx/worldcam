"""Two-phase capture benchmark.

Phase 1  raw read loop, no processing, no display.
         Slow here  => driver / backend / pixel format / low light (not your code).
Phase 2  threaded grabber + simulated per-frame work (--work-ms).
         Slow here but phase 1 fine => your loop structure.

  worldcam-bench --source 0 --width 1280 --height 720 --fps 30
  worldcam-bench --source rtsp://user:pass@192.168.1.10/stream1
  worldcam-bench --source synthetic --work-ms 80
  worldcam-bench --source 0 --backend any --fourcc none

Warm-up: the first --warmup seconds (default 1.0) after open are excluded from
the phase 1 fps / frame-gap stats and the phase 2 age stats, because the first
"gap" is camera start-up (auto-exposure, AVFoundation session start) rather
than steady-state delivery. Time-to-first-frame is printed separately so the
warm-up cost is still visible. Use --warmup 0 to include everything.
"""
from __future__ import annotations

import argparse
import time

from ..capture import LatestFrameGrabber
from ..metrics import percentile
from ..sources import make_source
from ..sources.webcam import BACKENDS, FOURCCS


def _make(args):
    return make_source(
        args.source, args.width, args.height, args.fps,
        backend=args.backend, fourcc=args.fourcc,
    )


def phase1(args) -> None:
    src = _make(args)
    src.open()
    print("negotiated:", src.describe())
    n, fails, t0 = 0, 0, time.monotonic()
    warm_end = t0 + args.warmup
    gaps: list[float] = []
    first_frame_ms = None
    last = None  # monotonic time of the previous frame, once past warm-up
    t_meas = None  # start of the measured window
    while time.monotonic() - t0 < args.seconds:
        try:
            f = src.read()
        except EOFError:
            break
        now = time.monotonic()
        if f is None:
            fails += 1
            continue
        if first_frame_ms is None:
            first_frame_ms = (now - t0) * 1000
        if now < warm_end:
            continue  # warm-up: not counted
        if last is None:
            t_meas = now  # first post-warm-up frame anchors the window
        else:
            gaps.append((now - last) * 1000)
            n += 1
        last = now
    src.close()
    dur = max(1e-6, (last or 0) - (t_meas or 0))
    ffm = "n/a" if first_frame_ms is None else f"{first_frame_ms:.0f}"
    print(
        f"[phase1 raw read] {n / dur:.1f} fps | frame gap ms p50={percentile(gaps, 50):.1f} "
        f"p95={percentile(gaps, 95):.1f} max={max(gaps, default=0):.1f} | failures={fails} "
        f"| first frame {ffm} ms, warm-up {args.warmup:.1f}s excluded"
    )


def phase2(args) -> None:
    src = _make(args)
    ages: list[float] = []
    t0 = time.monotonic()
    warm_end = t0 + args.warmup
    with LatestFrameGrabber(src) as g:
        last = 0
        while time.monotonic() - t0 < args.seconds:
            f = g.get(after_seq=last, timeout=1.0)
            if f is None:
                if g.eof:
                    break
                continue
            last = f.seq
            if time.monotonic() >= warm_end:
                ages.append((time.monotonic() - f.ts) * 1000)
            time.sleep(args.work_ms / 1000)  # stand-in for detector/VLM cost
        dur = max(1e-6, time.monotonic() - t0)
        print(
            f"[phase2 grabber, work={args.work_ms:.0f}ms] consumed={g.frames_consumed / dur:.1f} fps "
            f"read={g.frames_read / dur:.1f} fps dropped={g.dropped} | "
            f"age at use ms p50={percentile(ages, 50):.1f} p95={percentile(ages, 95):.1f}"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--work-ms", type=float, default=50.0)
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds after open excluded from stats (default 1.0)")
    ap.add_argument("--backend", type=str.lower, choices=sorted(BACKENDS), default=None,
                    help="camera capture API (default: platform default)")
    ap.add_argument("--fourcc", type=lambda v: "none" if v.lower() == "none" else v.upper(),
                    choices=FOURCCS, default=None,
                    help="requested pixel format; 'none' = don't set (default: MJPG)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.warmup >= args.seconds:
        raise SystemExit("--warmup must be shorter than --seconds")
    phase1(args)
    phase2(args)


if __name__ == "__main__":
    main()
