"""Detector post-filters: cut YOLO false positives before they reach the tracker,
the timeline and the doubt engine.

Seen in real use: a pink blanket, a face close-up and a raised hand were all
reported as "cat". Animals are rare in a living room, so they need a higher
confidence than people, and an animal box that sits inside a person box is
almost always a body part or clothing, not an animal.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..pipeline import Detection
from ..types import Frame
from .boxes import ioa

ANIMALS = frozenset({"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"})


class FilteredDetector:
    """Wraps any Detector. `class_conf` raises the confidence bar per label;
    labels in `suppress_inside_person` are dropped when >= `person_overlap` of
    their box lies inside a detected person."""

    def __init__(self, inner, class_conf: Optional[Mapping[str, float]] = None, *,
                 suppress_inside_person: Sequence[str] = tuple(ANIMALS), person_overlap: float = 0.5):
        self.inner = inner
        self.class_conf = dict(class_conf or {})
        self.suppress = frozenset(suppress_inside_person)
        self.person_overlap = person_overlap
        self.dropped = {"low_conf": 0, "inside_person": 0}

    def detect(self, frame: Frame, regions=None) -> Sequence[Detection]:
        dets = list(self.inner.detect(frame, regions))
        people = [d.box for d in dets if d.label == "person"]
        out = []
        for d in dets:
            if d.conf < self.class_conf.get(d.label, 0.0):
                self.dropped["low_conf"] += 1
                continue
            if d.label in self.suppress and any(ioa(d.box, p) >= self.person_overlap for p in people):
                self.dropped["inside_person"] += 1
                continue
            out.append(d)
        return out

    def __getattr__(self, name):          # keep the wrapped detector's attributes (device, weights, ...)
        return getattr(self.inner, name)
