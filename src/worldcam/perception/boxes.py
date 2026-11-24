"""Small normalized-box helpers shared by the perception stages."""
from __future__ import annotations

from typing import Optional, Sequence

from ..geometry import Box


def clamp_box(box: Box) -> Box:
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def ioa(box: Box, region: Box) -> float:
    """Intersection over the area of `box` (how much of box lies in region)."""
    a = box_area(box)
    if a <= 0:
        return 0.0
    ix = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    return ix * iy / a


def overlaps_any(box: Box, regions: Optional[Sequence[Box]], min_ioa: float = 0.1) -> bool:
    """True if `regions` is None (whole frame observed) or at least `min_ioa`
    of the box lies inside one region. The detector uses this to drop
    detections outside motion regions, and the tracker uses the SAME rule to
    decide which tracks were observed this frame (so a track outside every
    region is not counted as a miss)."""
    if regions is None:
        return True
    return any(ioa(box, r) >= min_ioa for r in regions)


def union_box(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
