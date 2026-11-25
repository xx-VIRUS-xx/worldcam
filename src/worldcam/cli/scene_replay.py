"""Draw zones on a saved frame so the owner can check them. No camera access.

  python -m worldcam.cli.scene_replay --overlay frame.jpg --zones zones.json [--camera default] [--out out.jpg]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ..perception.zones import ZoneConfigError, load_zones

PALETTE = [(0, 200, 255), (80, 220, 80), (255, 140, 60), (220, 80, 220), (60, 120, 255), (200, 200, 60)]


def draw_zones(image: np.ndarray, zones) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    layer = image.copy()
    for i, z in enumerate(zones):
        color = PALETTE[i % len(PALETTE)]
        pts = np.array([[int(x * w), int(y * h)] for x, y in z.polygon], np.int32)
        cv2.fillPoly(layer, [pts], color)
        cv2.polylines(out, [pts], True, color, 2)
        cv2.putText(out, z.name, (int(pts[:, 0].min()) + 4, int(pts[:, 1].min()) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.25, out, 0.75, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlay", required=True, help="saved frame (image file)")
    ap.add_argument("--zones", required=True, help="zones.json")
    ap.add_argument("--camera", default=None, help="only draw this camera's zones")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    img = cv2.imread(a.overlay)
    if img is None:
        print(f"cannot read image {a.overlay}", file=sys.stderr)
        return 2
    try:
        zones = load_zones(a.zones)
    except (OSError, ZoneConfigError) as e:
        print(f"zones error: {e}", file=sys.stderr)
        return 2
    if a.camera:
        zones = [z for z in zones if z.camera == a.camera]
    p = Path(a.overlay)
    out = Path(a.out) if a.out else p.with_name(p.stem + "_zones.jpg")
    cv2.imwrite(str(out), draw_zones(img, zones))
    print(f"wrote {out} ({len(zones)} zones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
