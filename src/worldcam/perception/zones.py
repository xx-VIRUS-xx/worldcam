"""Zones: named polygons per camera, and a resolver that assigns tracks to them.

Assignment uses the box FOOT point (where a person stands), and a hysteresis
margin so a track standing on a boundary does not flicker between zones:
  - a track ENTERS a zone only when its foot is strictly inside the polygon;
  - it LEAVES only when the foot is further than `margin` outside the polygon.
All coordinates are normalized (0..1).
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union

from ..geometry import Point, box_foot, point_in_polygon
from ..pipeline import Track, Zone


class ZoneConfigError(ValueError):
    pass


def parse_zones(data: object) -> list[Zone]:
    """Validate `{"camera": {"zone": [[x, y], ...]}}` and return Zones."""
    if not isinstance(data, dict) or not data:
        raise ZoneConfigError("zones config must be a non-empty object {camera: {zone: [[x,y],...]}}")
    out: list[Zone] = []
    for cam, zs in data.items():
        if not isinstance(cam, str) or not cam:
            raise ZoneConfigError("camera names must be non-empty strings")
        if not isinstance(zs, dict):
            raise ZoneConfigError(f"camera {cam!r}: expected an object of zones")
        for name, pts in zs.items():
            where = f"{cam}/{name}"
            if not isinstance(name, str) or not name.strip():
                raise ZoneConfigError(f"camera {cam!r}: zone names must be non-empty strings")
            if not isinstance(pts, list) or len(pts) < 3:
                raise ZoneConfigError(f"{where}: polygon needs at least 3 points")
            poly: list[Point] = []
            for p in pts:
                if (not isinstance(p, (list, tuple)) or len(p) != 2
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in p)):
                    raise ZoneConfigError(f"{where}: points must be [x, y] numbers")
                x, y = float(p[0]), float(p[1])
                if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ZoneConfigError(f"{where}: point {p} outside normalized range 0..1")
                poly.append((x, y))
            if _area(poly) < 1e-6:
                raise ZoneConfigError(f"{where}: polygon is degenerate (zero area)")
            out.append(Zone(name=name, polygon=tuple(poly), camera=cam))
    return out


def load_zones(path: Union[str, Path]) -> list[Zone]:
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ZoneConfigError(f"{p}: invalid JSON: {e}") from e
    return parse_zones(data)


def _area(poly: Sequence[Point]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def distance_to_polygon(pt: Point, poly: Sequence[Point]) -> float:
    """0 if inside, else distance to the nearest edge."""
    if point_in_polygon(pt, poly):
        return 0.0
    best = float("inf")
    px, py = pt
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


class PolygonZoneResolver:
    """ZoneResolver. One instance per camera (`camera=None` = consider all zones)."""

    def __init__(self, camera: Optional[str] = None, margin: float = 0.03, max_tracks: int = 512):
        self.camera = camera
        self.margin = margin
        self.max_tracks = max_tracks
        self._current: "OrderedDict[int, tuple[str, ...]]" = OrderedDict()

    def assign(self, track: Track, zones: Sequence[Zone]) -> tuple[str, ...]:
        foot = box_foot(track.box)
        prev = self._current.get(track.track_id, ())
        found: list[str] = []
        for z in zones:
            if self.camera is not None and z.camera != self.camera:
                continue
            if z.name in found:
                continue
            d = distance_to_polygon(foot, z.polygon)
            if d == 0.0 or (z.name in prev and d <= self.margin):
                found.append(z.name)
        res = tuple(found)
        self._current[track.track_id] = res
        self._current.move_to_end(track.track_id)
        while len(self._current) > self.max_tracks:
            self._current.popitem(last=False)
        return res

    def release(self, track_id: int) -> None:
        """Forget an ended track."""
        self._current.pop(track_id, None)
