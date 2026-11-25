"""Live detection + tracking overlay, or a headless benchmark.

  python -m worldcam.cli.detect --source 0                       # live window, q/ESC quits
  python -m worldcam.cli.detect --source 0 --no-display --seconds 30
  python -m worldcam.cli.detect --source 0 --no-display --seconds 30 --no-motion
  python -m worldcam.cli.detect --source 0 --region-mode crop ...
  python -m worldcam.cli.detect --source <sub> --main <main> ...  # DualGrabber

Capture is full-res (--width/--height); motion + detection run on a
--det-width copy (or on the sub stream with --main); best-snapshot crops are
cut from the full-res frame.

Benchmark output: end-to-end fps (frames through the loop / s), detector fps
(YOLO calls / s) and detector capacity (1000 / detect p50), motion-skip ratio,
per-stage ms p50/p95, track counts.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import cv2

from ..capture import LatestFrameGrabber
from ..metrics import RateMeter
from ..pipeline import TrackState
from ..sources import DEFAULT_SOURCE, make_source
from ..sources.dual import DualGrabber
from ..perception.detector import DEFAULT_CLASSES, YoloDetector
from ..perception.motion import FrameDiffMotionGate
from ..perception.pipeline_runner import STAGES, PerceptionLoop
from ..perception.tracker import IouTracker

COLORS = {
    TrackState.NEW: (200, 200, 200),
    TrackState.ACTIVE: (0, 220, 0),
    TrackState.STATIONARY: (255, 160, 0),
    TrackState.LOST: (0, 0, 255),
}


def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="camera index / URL / file; the SUB stream if --main is set")
    ap.add_argument("--main", default=None, help="optional full-res main stream (enables DualGrabber)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--main-size", default="1920x1080", help="WxH for --main")
    ap.add_argument("--det-width", type=int, default=640, help="width of the detection copy (single source)")
    ap.add_argument("--weights", default=None, help="YOLO weights (default: models/yolo11n.pt)")
    ap.add_argument("--device", default=None, help="mps | cpu (default: mps if available)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma-separated COCO names, or 'all'")
    ap.add_argument("--region-mode", choices=("filter", "crop", "full"), default="filter")
    ap.add_argument("--no-motion", action="store_true", help="disable the motion gate (detect every frame)")
    ap.add_argument("--motion-threshold", type=int, default=25)
    ap.add_argument("--motion-min-area", type=float, default=0.002)
    ap.add_argument("--full-every", type=float, default=5.0, help="force a full-frame detect every N s")
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--still-after", type=float, default=10.0)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until quit")
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds excluded from the stats")
    ap.add_argument("--display-width", type=int, default=960)
    return ap


def _draw(loop: PerceptionLoop, upd, fps: float) -> "cv2.typing.MatLike":
    base = loop.last_main if loop.last_main is not None else loop.last_sub
    img = base.image.copy()  # never draw on a frame the grabber/tracker may still crop from
    h, w = img.shape[:2]
    for r in loop.last_regions or []:
        cv2.rectangle(img, (int(r[0] * w), int(r[1] * h)), (int(r[2] * w), int(r[3] * h)), (90, 90, 90), 1)
    for t in upd.active:
        c = COLORS.get(t.state, (255, 255, 255))
        p1 = (int(t.box[0] * w), int(t.box[1] * h))
        p2 = (int(t.box[2] * w), int(t.box[3] * h))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(img, f"#{t.track_id} {t.label} {t.state.value}", (p1[0], max(14, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    mode = getattr(loop.gate, "last_reason", "-") if loop.gate else "off"
    txt = f"{fps:.1f} fps | det {loop.detector_runs} skip {loop.skip_ratio:.0%} | motion {mode}"
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    classes = None if args.classes == "all" else [c.strip() for c in args.classes.split(",") if c.strip()]

    t_load = time.perf_counter()
    det = YoloDetector(weights=args.weights, device=args.device, classes=classes, conf=args.conf,
                       imgsz=args.imgsz, region_mode=args.region_mode)
    print(f"detector: {det.weights} device={det.device} imgsz={det.imgsz} conf={det.conf} "
          f"mode={det.region_mode} load+warmup={(time.perf_counter() - t_load):.1f}s", flush=True)
    gate = None if args.no_motion else FrameDiffMotionGate(
        threshold=args.motion_threshold, min_area=args.motion_min_area, full_every_s=args.full_every)
    tracker = IouTracker(n_init=args.n_init, still_after_s=args.still_after)

    sub_src = make_source(args.source, args.width, args.height, args.fps)
    if args.main:
        mw, mh = _parse_size(args.main_size)
        grabber = DualGrabber(sub_src, make_source(args.main, mw, mh, args.fps), main_ring=5)
    else:
        grabber = LatestFrameGrabber(sub_src)

    meter = RateMeter()
    ended_states: Counter = Counter()
    ended_labels: Counter = Counter()
    best_src: Counter = Counter()
    best_sizes: list[tuple[int, int]] = []
    measuring = args.warmup <= 0
    try:
        with grabber:
            print("negotiated:", sub_src.describe(), flush=True)
            loop = PerceptionLoop(grabber, det, tracker, gate, sub_width=args.det_width)
            t0 = t_meas = time.monotonic()  # after the camera is open
            for upd in loop.run(seconds=(args.seconds + args.warmup) if args.seconds else None):
                if not measuring and time.monotonic() - t0 >= args.warmup:
                    measuring = True
                    loop.times = type(loop.times)()
                    loop.frames = loop.detector_runs = loop.motion_skips = 0
                    t_meas = time.monotonic()
                meter.tick()
                for t in upd.ended:
                    ended_labels[t.label] += 1
                    if t.best is not None:
                        best_src[next((r for r in t.best.reasons if r.startswith("src=")), "?")] += 1
                        if t.best.crop is not None:
                            best_sizes.append(t.best.crop.shape[1::-1])
                if args.no_display:
                    continue
                img = _draw(loop, upd, meter.fps)
                h, w = img.shape[:2]
                if w > args.display_width:
                    img = cv2.resize(img, (args.display_width, int(h * args.display_width / w)))
                cv2.imshow("worldcam-detect", img)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            t_stop = time.monotonic()  # before the grabber shuts down
            # still-active tracks at the end, for the report
            live = list(tracker._tracks.values())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        t_stop = time.monotonic()
        live = list(tracker._tracks.values())
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()

    dur = max(1e-6, t_stop - t_meas)
    print(f"\n=== perception bench ({dur:.1f}s measured, warm-up {args.warmup:.1f}s excluded, "
          f"wall {t_stop - t0:.1f}s) ===")
    print(f"frames={loop.frames} e2e_fps={loop.frames / dur:.1f} "
          f"detector_calls={loop.detector_runs} detector_fps={loop.detector_runs / dur:.1f} "
          f"detector_capacity_fps={1000 / max(1e-6, loop.times.pct('detect', 50)):.1f}")
    print(f"motion_skip_ratio={loop.skip_ratio:.3f} (gate={'off' if gate is None else 'on'}) "
          f"grabber_dropped={getattr(grabber, 'dropped', getattr(getattr(grabber, 'sub', None), 'dropped', '-'))}")
    print("stage ms        p50     p95     max       n")
    for s in STAGES:
        p50, p95, n = loop.times.summary()[s]
        mx = max(loop.times.samples[s], default=0.0)
        print(f"  {s:<10} {p50:7.2f} {p95:7.2f} {mx:7.1f} {n:7d}")
    st = tracker.stats
    print(f"tracks: created={st['created']} confirmed={st['confirmed']} ended={st['ended']} "
          f"dropped_tentative={st['dropped_tentative']} max_concurrent={loop.max_active} live_at_end={len(live)}")
    live_by = Counter(f"{t.label}:{t.state.value}" for t in live)
    print(f"  ended by label: {dict(ended_labels)} | live by label/state: {dict(live_by)}")
    for t in live:
        if t.best is not None:
            best_src[next((r for r in t.best.reasons if r.startswith("src=")), "?")] += 1
            if t.best.crop is not None:
                best_sizes.append(t.best.crop.shape[1::-1])
    print(f"  best snapshots: src={dict(best_src)} updates={st['best_updates']} "
          f"sub_fallbacks={st['best_fallback_sub']} crop sizes (w,h) sample={best_sizes[:5]}")


if __name__ == "__main__":
    main()
