"""Stage contracts for everything AFTER capture. Interfaces only; nothing here
is implemented yet. Each stage consumes the previous stage's output type, so
stages can be built and tested independently from recorded clips.

All boxes/polygons are NORMALIZED (0..1); see geometry.py.

Flow (patterns borrowed from NVR practice, the last two stages are the product):

  DualGrabber(sub, main)
    -> MotionGate        cheap frame-diff; no motion => detector does not run
    -> Detector          on the substream, only in motion regions
    -> Tracker           persistent track ids + lifecycle + best snapshot
    -> ZoneResolver      tracks -> named zones ("desk", "door")
    -> IdentityModel     once per track on its best snapshot; "unknown" is valid
    -> EventGate         decides when the slow VLM is worth calling
    -> VisionLLM         constrained JSON hypothesis from best-snapshot crops
    -> ActivityFuser     VLM hypothesis + hard signals -> activity + confidence
    -> WorldState        live entity model with TTLs
    -> Memory            append-only structured events + summaries + query
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence

import numpy as np

from .geometry import Box, Point
from .types import Frame


# ---- perception ----------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    label: str                      # "person", "laptop", "cat", ...
    conf: float
    box: Box                        # normalized


class TrackState(str, Enum):
    NEW = "new"                     # seen < N frames, not yet trusted
    ACTIVE = "active"               # moving / confirmed
    STATIONARY = "stationary"       # not moved for T seconds; re-verify rarely
    LOST = "lost"                   # missed recently, may come back (occlusion)
    ENDED = "ended"                 # gone for good; emitted once


@dataclass(frozen=True)
class BestSnapshot:
    """The single frame a track is judged on. Chosen by score (detector conf,
    box size, sharpness, face visible). Identity and VLM use THIS, not every frame."""
    seq: int
    ts: float
    box: Box
    score: float
    crop: Optional[np.ndarray] = None   # full-res crop taken from the mainstream
    reasons: tuple[str, ...] = ()       # why it won: ("largest", "face_visible")


@dataclass(frozen=True)
class Track:
    track_id: int
    label: str
    state: TrackState
    box: Box
    first_ts: float
    last_ts: float
    zones: tuple[str, ...] = ()
    stationary_since: Optional[float] = None
    best: Optional[BestSnapshot] = None


@dataclass(frozen=True)
class TrackUpdate:
    active: Sequence[Track]
    ended: Sequence[Track]          # each track appears here exactly once


@dataclass(frozen=True)
class Zone:
    name: str                       # "desk", "door", "bed"
    polygon: tuple[Point, ...]      # normalized
    camera: str = "default"


@dataclass(frozen=True)
class Identity:
    track_id: int
    name: str                       # "unknown" is a first-class answer
    conf: float


@dataclass(frozen=True)
class FaceObservation:
    """One face found in a track's best snapshot (identity/faces.py output).
    Added for the ambiguity engine: Identity alone can't carry the embedding
    the graph needs for clustering and matching."""
    track_id: int
    ts: float
    box: Box                        # normalized, relative to the full frame
    embedding: tuple[float, ...]    # L2-normalized, 512-d (ArcFace)
    quality: float                  # 0..1: detector score x size x sharpness x frontalness
    crop_path: Optional[str] = None # saved face crop (local), sent to the owner when asked


@dataclass(frozen=True)
class ActivityHypothesis:
    """VLM output. A hypothesis, never a fact."""
    activity: str                   # "coding", "eating", "unknown", ...
    objects: tuple[str, ...]
    confidence: float
    raw: dict = field(default_factory=dict)


# ---- world / memory ------------------------------------------------------

@dataclass
class Entity:
    entity_id: str                  # "prabhat", "unknown_3", "cat_1"
    kind: str                       # person | pet | object
    zone: Optional[str] = None
    activity: Optional[str] = None
    activity_conf: float = 0.0
    since_ts: float = 0.0           # when the current (zone, activity) began
    last_seen_ts: float = 0.0
    track_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Event:
    """Structured, append-only unit of memory. The VLM never writes free text
    into memory directly; text summaries are derived later from these."""
    ts_wall: float
    kind: str                       # entered | left | zone_changed | activity_changed
    entity_id: str
    zone: Optional[str] = None
    activity: Optional[str] = None
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)  # track ids, snapshot refs, model ids


# ---- stage protocols -----------------------------------------------------

class MotionGate(Protocol):
    def regions(self, frame: Frame) -> Sequence[Box]:
        """Empty => skip the detector entirely for this frame."""


class Detector(Protocol):
    def detect(self, frame: Frame, regions: Optional[Sequence[Box]] = None) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def update(self, frame: Frame, dets: Sequence[Detection]) -> TrackUpdate:
        """Owns lifecycle (NEW/ACTIVE/STATIONARY/LOST/ENDED) and best-snapshot selection."""


class ZoneResolver(Protocol):
    def assign(self, track: Track, zones: Sequence[Zone]) -> tuple[str, ...]:
        """Use the box's foot point (geometry.box_foot), not its center."""


class IdentityModel(Protocol):
    def identify(self, track: Track) -> Identity:
        """Per track on track.best; cache by track_id, re-verify rarely."""


class EventGate(Protocol):
    def should_fire(self, update: TrackUpdate, ids: Sequence[Identity], world: dict) -> bool:
        """Fire on state changes only (entered, new zone, stationary->active,
        dwell threshold), never on a timer per frame. Enforce a per-hour VLM budget."""


class VisionLLM(Protocol):
    def describe(self, tracks: Sequence[Track], context: dict) -> ActivityHypothesis:
        """Input: best-snapshot crops + structured context. Output: constrained JSON."""


class ActivityFuser(Protocol):
    def fuse(self, hyp: ActivityHypothesis, track: Track, objects_in_zone: Sequence[str],
             dwell_s: float) -> tuple[str, float]:
        """Combine the VLM guess with hard signals (zone, objects, dwell) into
        (activity, confidence). 'coding' needs a person in the desk zone with a
        laptop for a minimum dwell, not just one caption."""


class WorldState(Protocol):
    def apply(self, event: Event) -> None: ...
    def snapshot(self) -> dict[str, Entity]: ...


class Memory(Protocol):
    def append(self, event: Event) -> None: ...
    def query(self, text: str, limit: int = 20) -> Sequence[Event]: ...
