"""Live preview with FPS and frame age overlay, plus periodic soak stats.

  worldcam-preview --source 0
  worldcam-preview --source http://192.168.1.5:8080/video
  worldcam-preview --source synthetic --no-display --seconds 3   # headless check
  worldcam-preview --source 0 --no-display --seconds 1800 --stats-every 10 \\
      --stats-csv soak.csv                                       # soak test

Every --stats-every seconds a `stats` line is printed (and appended to
--stats-csv): interval fps, grabber read fps, frames read/consumed/dropped,
read failures, max frame gap (interval and overall), max frame age, current
and peak RSS, and source reconnects. A `summary` line is printed at the end,
also on Ctrl-C. Its rss_growth_mb compares the mean RSS of the first minute
(after warm-up) with the last minute; see SoakStats for why.

Quit with q or ESC (or Ctrl-C).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import cv2

from ..capture import LatestFrameGrabber
from ..metrics import STATS_FIELDS, RateMeter, SoakStats, format_row
from ..sources import make_source


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview-width", type=int, default=640, help="display/detection copy width")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until quit")
    ap.add_argument("--stats-every", type=float, default=10.0, metavar="SEC",
                    help="print periodic stats every SEC seconds (0 = off)")
    ap.add_argument("--stats-csv", metavar="PATH", help="also append stats rows to this CSV file")
    args = ap.parse_args(argv)

    src = make_source(args.source, args.width, args.height, args.fps)
    meter = RateMeter()
    csv_fh = None
    writer = None
    if args.stats_csv:
        csv_fh = open(args.stats_csv, "w", newline="")
        writer = csv.DictWriter(csv_fh, fieldnames=STATS_FIELDS)
        writer.writeheader()
        csv_fh.flush()

    def emit(row: dict) -> None:
        print("stats", format_row(row), flush=True)
        if writer is not None:
            writer.writerow(row)
            csv_fh.flush()  # a killed soak run still leaves a usable CSV

    t_start = time.monotonic()
    # A trailing interval shorter than half a period is still printed but not
    # counted in fps min/mean/max.
    stats = SoakStats(now=t_start, min_interval=max(0.5, args.stats_every / 2))
    g = LatestFrameGrabber(src)
    end = t_start
    try:
        with g:
            print("negotiated:", src.describe(), flush=True)
            try:
                _loop(args, g, src, stats, meter, emit, t_start)
            finally:
                # Final partial interval, taken while the grabber is still
                # running so its shutdown time is not reported as a frame gap.
                end = time.monotonic()
                if args.stats_every > 0 and end - stats.interval_start >= 0.5:
                    emit(stats.snapshot(g, src, end))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr, flush=True)
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        if csv_fh is not None:
            csv_fh.close()
    s = stats.summary(end)
    print(f"done: {meter.fps:.1f} fps (rolling), read={g.frames_read} consumed={g.frames_consumed} "
          f"dropped={g.dropped} read_failures={g.read_failures}", flush=True)
    print("summary", format_row(s), flush=True)


def _loop(args, g: LatestFrameGrabber, src, stats: SoakStats, meter: RateMeter, emit, t_start: float) -> None:
    """Consume newest frames until --seconds, EOF, or q/ESC."""
    next_stats = t_start + args.stats_every
    last = 0
    while True:
        now = time.monotonic()
        if args.stats_every > 0 and now >= next_stats:
            emit(stats.snapshot(g, src, now))
            next_stats += args.stats_every
            if next_stats <= now:  # fell behind (e.g. long stall): realign
                next_stats = now + args.stats_every
        if args.seconds and now - t_start > args.seconds:
            return
        timeout = 1.0
        if args.stats_every > 0:
            timeout = min(timeout, max(0.01, next_stats - now))
        f = g.get(after_seq=last, timeout=timeout)
        if f is None:
            if g.eof:
                return
            continue
        last = f.seq
        meter.tick()
        stats.on_frame(f.ts)
        w, h = f.shape
        scale = args.preview_width / w
        small = cv2.resize(f.image, (args.preview_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        age_ms = (time.monotonic() - f.ts) * 1000
        label = f"{w}x{h} | {meter.fps:.1f} fps | age {age_ms:.0f} ms | dropped {g.dropped}"
        cv2.putText(small, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if args.no_display:
            continue
        cv2.imshow("worldcam", small)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            return


if __name__ == "__main__":
    main()
