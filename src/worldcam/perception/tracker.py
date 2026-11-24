"""IoU tracker with lifecycle and best-snapshot selection.

Why our own small IoU/Hungarian tracker instead of ultralytics
`model.track(persist=True)` (ByteTrack/BoT-SORT):
- The motion gate skips the detector on still frames and the detector may only
  look at motion regions. model.track() assumes it sees every frame in full;
  skipped frames age its tracks into "lost" and it can't tell "not observed"
  from "missed". Here update() takes `observed_regions`, and a track outside
  every observed region is simply not counted as a miss.
- It keeps Detector and Tracker separate (the stage protocols), so the
  tracker is testable with scripted detections and no weights.
- At 10-30 fps indoors with few objects, IoU + Hungarian (scipy, greedy
  fallback) is enough; ByteTrack's low-score second pass and Kalman motion
  model matter mostly for crowds and fast motion.

Lifecycle (all times are Frame.ts seconds):
  NEW         -> ACTIVE after `n_init` matched frames
  ACTIVE      -> STATIONARY when the foot point has stayed within `still_eps`
                 for `still_after_s`; back to ACTIVE when it moves more
  any         -> LOST after `lost_misses` consecutive observed misses
  LOST        -> ACTIVE/STATIONARY when matched again
  LOST        -> ENDED `end_after_s` after it became LOST; also any track not
                 matched for `max_unseen_s` (safety net). Emitted in
                 TrackUpdate.ended exactly once, then forgotten.
  NEW tracks that die before confirmation are dropped silently (noise), and
  counted in `stats["dropped_tentative"]`, unless emit_unconfirmed=True.

Best snapshot: score = conf x box_area x sharpness, sharpness = Laplacian
variance of the (sub-frame) box crop resized to a fixed height, squashed to
0..1 (v / (v + sharp_ref)). The score is computed on the cheap sub frame; only
when it beats the current best by `best_margin` is the crop cut, from the
FULL-RES main frame through `crop_fn`. If no crop_fn is given (no main
available) the crop comes from the sub frame and the reasons say so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from ..geometry import Box, box_foot, crop_norm, iou
from ..pipeline import BestSnapshot, Detection, Track, TrackState, TrackUpdate
from ..types import Frame
from .boxes import box_area, overlaps_any

try:  # scipy ships with ultralytics; greedy fallback otherwise
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

# crop_fn(box) -> (image or None, reasons describing where it came from)
CropFn = Callable[[Box], tuple[Optional[np.ndarray], tuple[str, ...]]]


def sharpness(image: np.ndarray, box: Box, norm_h: int = 96) -> float:
    """Laplacian variance of the box crop, resized to a fixed height so boxes
    of different sizes compare fairly. Raw variance (not squashed)."""
    c = crop_norm(image, box, pad=0.0)
    if c is None or c.size == 0:
        return 0.0
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    h, w = g.shape[:2]
    if h != norm_h:
        g = cv2.resize(g, (max(1, int(round(w * norm_h / h))), norm_h), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


@dataclass
class _T:
    track_id: int
    label: str
    box: Box
    conf: float
    first_ts: float
    last_ts: float
    state: TrackState = TrackState.NEW
    hits: int = 1
    misses: int = 0
    anchor: tuple[float, float] = (0.0, 0.0)
    anchor_ts: float = 0.0
    stationary_since: Optional[float] = None
    lost_since: Optional[float] = None
    best: Optional[BestSnapshot] = None
    best_parts: dict = field(default_factory=dict)  # conf/area/sharp of the best, for "x_so_far" tags

    def freeze(self, state: Optional[TrackState] = None) -> Track:
        return Track(
            track_id=self.track_id, label=self.label, state=state or self.state,
            box=self.box, first_ts=self.first_ts, last_ts=self.last_ts,
            stationary_since=self.stationary_since, best=self.best,
        )


class IouTracker:
    """Tracker implementation (see pipeline.Tracker)."""

    def __init__(
        self,
        iou_match: float = 0.3,
        n_init: int = 3,
        lost_misses: int = 2,
        end_after_s: float = 3.0,
        max_unseen_s: float = 120.0,
        still_eps: float = 0.03,
        still_after_s: float = 10.0,
        min_region_ioa: float = 0.1,
        sharp_ref: float = 100.0,
        best_margin: float = 1.05,
        best_labels: Optional[Sequence[str]] = None,
        crop_pad: float = 0.1,
        emit_unconfirmed: bool = False,
    ):
        self.iou_match = iou_match
        self.n_init = n_init
        self.lost_misses = lost_misses
        self.end_after_s = end_after_s
        self.max_unseen_s = max_unseen_s
        self.still_eps = still_eps
        self.still_after_s = still_after_s
        self.min_region_ioa = min_region_ioa
        self.sharp_ref = sharp_ref
        self.best_margin = best_margin
        self.best_labels = None if best_labels is None else set(best_labels)
        self.crop_pad = crop_pad
        self.emit_unconfirmed = emit_unconfirmed
        self._tracks: dict[int, _T] = {}
        self._next_id = 1
        self.stats = {"created": 0, "confirmed": 0, "ended": 0, "dropped_tentative": 0,
                      "best_updates": 0, "best_fallback_sub": 0}

    # ---- matching ----------------------------------------------------
    def _match(self, tracks: list[_T], dets: Sequence[Detection]) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not tracks or not dets:
            return [], set(range(len(tracks))), set(range(len(dets)))
        cost = np.ones((len(tracks), len(dets)))
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if d.label == t.label:
                    cost[i, j] = 1.0 - iou(t.box, d.box)
        pairs: list[tuple[int, int]] = []
        if _lsa is not None:
            rows, cols = _lsa(cost)
            cand = list(zip(rows.tolist(), cols.tolist()))
        else:
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            used_r, used_c, cand = set(), set(), []
            for r, c in order.tolist():
                if r not in used_r and c not in used_c:
                    used_r.add(r); used_c.add(c); cand.append((r, c))
        for r, c in cand:
            if cost[r, c] <= 1.0 - self.iou_match:
                pairs.append((r, c))
        mt = {r for r, _ in pairs}
        md = {c for _, c in pairs}
        return pairs, set(range(len(tracks))) - mt, set(range(len(dets))) - md

    # ---- best snapshot -----------------------------------------------
    def _consider_best(self, t: _T, d: Detection, frame: Frame, crop_fn: Optional[CropFn]) -> None:
        if self.best_labels is not None and t.label not in self.best_labels:
            return
        area = box_area(d.box)
        sv = sharpness(frame.image, d.box)
        sharp = sv / (sv + self.sharp_ref)
        score = d.conf * area * sharp
        if score <= 0:
            return
        if t.best is not None and score < t.best.score * self.best_margin:
            return
        prev = t.best_parts
        tags = [f"conf={d.conf:.2f}", f"area={area:.3f}", f"sharp={sharp:.2f}"]
        if t.best is None:
            tags.append("first")
        else:
            for k, v in (("conf", d.conf), ("area", area), ("sharp", sharp)):
                if v > prev.get(k, 0.0):
                    tags.append({"conf": "most_confident", "area": "largest", "sharp": "sharpest"}[k])
        if crop_fn is not None:
            img, src = crop_fn(d.box)
        else:
            img = crop_norm(frame.image, d.box, self.crop_pad)
            src = ("src=sub", "main_unavailable")
        if any(s.startswith("src=sub") for s in src):
            self.stats["best_fallback_sub"] += 1
        bw, bh = d.box[2] - d.box[0], d.box[3] - d.box[1]
        p = self.crop_pad
        crop_box = (max(0.0, d.box[0] - p * bw), max(0.0, d.box[1] - p * bh),
                    min(1.0, d.box[2] + p * bw), min(1.0, d.box[3] + p * bh))
        t.best = BestSnapshot(seq=frame.seq, ts=frame.ts, box=d.box, score=score,
                              crop=img, reasons=tuple(tags) + tuple(src), crop_box=crop_box)
        t.best_parts = {"conf": d.conf, "area": area, "sharp": sharp}
        self.stats["best_updates"] += 1

    # ---- main entry --------------------------------------------------
    def update(
        self,
        frame: Frame,
        dets: Sequence[Detection],
        observed_regions: Optional[Sequence[Box]] = None,
        crop_fn: Optional[CropFn] = None,
    ) -> TrackUpdate:
        """frame: the SUB frame the detections were made on.
        observed_regions: where the detector looked this frame. None = whole
            frame; [] = detector skipped (no track can be missed).
        crop_fn: cuts a normalized box from the full-res MAIN frame matching
            `frame` (see PerceptionLoop). None = crop from `frame`."""
        ts = frame.ts
        tracks = list(self._tracks.values())
        pairs, un_t, un_d = self._match(tracks, dets)

        for ti, di in pairs:
            t, d = tracks[ti], dets[di]
            t.box, t.conf, t.last_ts = d.box, d.conf, ts
            t.hits += 1
            t.misses = 0
            foot = box_foot(d.box)
            if t.state == TrackState.NEW:
                if t.hits >= self.n_init:
                    t.state = TrackState.ACTIVE
                    self.stats["confirmed"] += 1
                t.anchor, t.anchor_ts = foot, ts
            elif t.state == TrackState.LOST:
                t.state, t.lost_since = TrackState.ACTIVE, None
            if t.state in (TrackState.ACTIVE, TrackState.STATIONARY):
                if math.dist(foot, t.anchor) > self.still_eps:
                    t.anchor, t.anchor_ts = foot, ts
                    t.state, t.stationary_since = TrackState.ACTIVE, None
                elif t.state == TrackState.ACTIVE and ts - t.anchor_ts >= self.still_after_s:
                    t.state, t.stationary_since = TrackState.STATIONARY, t.anchor_ts
            self._consider_best(t, d, frame, crop_fn)

        for ti in un_t:
            t = tracks[ti]
            if overlaps_any(t.box, observed_regions, self.min_region_ioa):
                t.misses += 1
                if t.state == TrackState.NEW:
                    # tentative tracks die on the first miss
                    t.state = TrackState.ENDED
                elif t.misses >= self.lost_misses and t.state != TrackState.LOST:
                    t.state, t.lost_since = TrackState.LOST, ts

        for di in un_d:
            d = dets[di]
            t = _T(self._next_id, d.label, d.box, d.conf, ts, ts,
                   anchor=box_foot(d.box), anchor_ts=ts)
            if self.n_init <= 1:
                t.state = TrackState.ACTIVE
                self.stats["confirmed"] += 1
            self._next_id += 1
            self._tracks[t.track_id] = t
            self.stats["created"] += 1
            self._consider_best(t, d, frame, crop_fn)

        ended: list[Track] = []
        for tid in list(self._tracks):
            t = self._tracks[tid]
            unseen = ts - t.last_ts
            done = (t.state == TrackState.ENDED
                    or (t.state == TrackState.LOST and ts - (t.lost_since or ts) >= self.end_after_s)
                    or unseen >= self.max_unseen_s)
            if not done:
                continue
            del self._tracks[tid]
            confirmed = t.hits >= self.n_init
            if not confirmed and not self.emit_unconfirmed:
                self.stats["dropped_tentative"] += 1
                continue
            self.stats["ended"] += 1
            ended.append(t.freeze(TrackState.ENDED))

        active = [t.freeze() for t in self._tracks.values()]
        return TrackUpdate(active=active, ended=ended)

    @property
    def live(self) -> int:
        return len(self._tracks)
