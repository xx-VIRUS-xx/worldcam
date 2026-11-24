"""Live preview with FPS and frame age overlay.

  worldcam-preview --source 0
  worldcam-preview --source http://192.168.1.5:8080/video
  worldcam-preview --source synthetic --no-display --seconds 3   # headless check

Quit with q or ESC.
"""
from __future__ import annotations

import argparse
import time

import cv2

from ..capture import LatestFrameGrabber
from ..metrics import RateMeter
from ..sources import make_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview-width", type=int, default=640, help="display/detection copy width")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until quit")
    args = ap.parse_args()

    src = make_source(args.source, args.width, args.height, args.fps)
    meter = RateMeter()
    t_start = time.monotonic()
    with LatestFrameGrabber(src) as g:
        print("negotiated:", src.describe())
        last = 0
        while True:
            if args.seconds and time.monotonic() - t_start > args.seconds:
                break
            f = g.get(after_seq=last, timeout=1.0)
            if f is None:
                if g.eof:
                    break
                continue
            last = f.seq
            meter.tick()
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
                break
    cv2.destroyAllWindows()
    print(f"done: {meter.fps:.1f} fps (rolling), dropped={g.dropped}")


if __name__ == "__main__":
    main()
