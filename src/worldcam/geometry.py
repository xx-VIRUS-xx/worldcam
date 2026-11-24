"""Resolution-independent geometry. All boxes and polygons are NORMALIZED
(0..1, x right, y down), so a box found on a 640px substream can crop the
1080p mainstream without any coordinate bookkeeping."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Box = tuple[float, float, float, float]   # x1, y1, x2, y2 normalized
Point = tuple[float, float]


def box_center(box: Box) -> Point:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_foot(box: Box) -> Point:
    """Bottom-center: where a person stands. Better than the center for zones."""
    return (box[0] + box[2]) / 2, box[3]


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    """Ray casting. Points exactly on an edge may go either way."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def crop_norm(image: np.ndarray, box: Box, pad: float = 0.1) -> Optional[np.ndarray]:
    """Crop a normalized box (padded by `pad` of its own size) from an image of
    any resolution. Returns a copy, or None if the result is empty."""
    h, w = image.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0.0, box[0] - pad * bw)
    y1 = max(0.0, box[1] - pad * bh)
    x2 = min(1.0, box[2] + pad * bw)
    y2 = min(1.0, box[3] + pad * bh)
    px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(round(x2 * w)), int(round(y2 * h))
    if px2 <= px1 or py2 <= py1:
        return None
    return image[py1:py2, px1:px2].copy()
