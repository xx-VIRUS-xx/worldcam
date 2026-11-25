"""RuleActivityFuser: VLM hypothesis + hard signals -> (activity, confidence).

A caption alone is never enough for activities that have a rule. Rules are DATA
(a JSON-able dict, see DEFAULT_RULES / load_rules):

  "<activity>": {
     "requires":    [cond, ...],        # ALL must hold
     "min_vlm_conf": 0.5,                # VLM must agree at least this strongly
     "on_fail":     "reject" | "cap",   # reject -> "unknown"; cap -> keep, lower confidence
     "penalty":     0.5,                # confidence multiplier on failure
     "bonus":       0.1,                # added when all requirements hold
     "max_conf":    0.7,                # hard ceiling (e.g. unvalidated detectors)
     "contradicts": [{"if": cond, "penalty": 0.5}, ...]
  }
  cond keys (all keys in one cond must hold): zone_any, zone_none, objects_any,
  objects_all, signal_any ({"zones": [...], "objects": [...]}), min_dwell_s,
  max_dwell_s, posture ("horizontal"|"upright"), state ("stationary"|"active"|...).

"objects" are the HARD objects (detector objects in the zone), not what the VLM
claims to see. The reasons for every decision are recorded in FusionResult.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from ..brain.api import BrainEvent, new_event
from ..pipeline import ActivityHypothesis, Track
from .eventgate import posture as box_posture

DEFAULT_RULES: dict[str, dict] = {
    "coding": {
        "requires": [{"zone_any": ["desk"]}, {"objects_any": ["laptop"]}, {"min_dwell_s": 120}],
        "on_fail": "reject", "penalty": 0.5,
    },
    "working_at_desk": {
        "requires": [{"zone_any": ["desk"]}, {"min_dwell_s": 60}],
        "on_fail": "cap", "penalty": 0.6,
    },
    "reading": {"requires": [{"min_dwell_s": 30}], "on_fail": "cap", "penalty": 0.7,
                "contradicts": [{"if": {"posture": "horizontal", "zone_none": ["bed", "sofa"]}, "penalty": 0.7}]},
    "taking_medication": {
        # NOT a validated detector: needs a medication object/zone signal AND a confident VLM.
        "requires": [{"signal_any": {"objects": ["pill_bottle", "medication"], "zones": ["medication"]}}],
        "min_vlm_conf": 0.6, "on_fail": "reject", "penalty": 0.4, "max_conf": 0.75,
    },
    "fall_suspected": {
        # NOT a validated detector: needs a horizontal box AND a VLM that agrees; always a lead.
        "requires": [{"posture": "horizontal"}],
        "min_vlm_conf": 0.5, "on_fail": "reject", "penalty": 0.4, "max_conf": 0.7,
    },
    "lying_down": {
        "requires": [{"posture": "horizontal"}], "on_fail": "reject", "penalty": 0.5,
    },
    "sitting": {"contradicts": [{"if": {"posture": "horizontal"}, "penalty": 0.5}]},
    "standing": {"contradicts": [{"if": {"posture": "horizontal"}, "penalty": 0.5}]},
    "walking": {"contradicts": [{"if": {"state": "stationary"}, "penalty": 0.5}]},
    "no_movement": {"requires": [{"state": "stationary"}, {"min_dwell_s": 60}],
                    "on_fail": "reject", "penalty": 0.5},
    "loitering": {"requires": [{"min_dwell_s": 60}], "on_fail": "reject", "penalty": 0.5},
    "person_at_door": {"requires": [{"zone_any": ["door"]}], "on_fail": "cap", "penalty": 0.6},
    "delivering_package": {"requires": [{"zone_any": ["door"]}], "on_fail": "cap", "penalty": 0.5},
}

INFERRED_CAP = 0.6   # contract: agent guesses are `inferred` with confidence <= 0.6


def load_rules(path: Union[str, Path]) -> dict[str, dict]:
    rules = json.loads(Path(path).read_text())
    if not isinstance(rules, dict) or not all(isinstance(v, dict) for v in rules.values()):
        raise ValueError("rules file must be {activity: rule-object}")
    return rules


@dataclass(frozen=True)
class FusionResult:
    activity: str
    confidence: float
    vlm_activity: str
    vlm_confidence: float
    corroborated: bool                    # all hard requirements of a rule held
    reasons: tuple[str, ...] = ()


def _cond_ok(cond: dict, zones: Sequence[str], objects: Sequence[str], dwell: float,
             post: str, state: str) -> bool:
    for k, v in cond.items():
        if k == "zone_any":
            ok = any(z in zones for z in v)
        elif k == "zone_none":
            ok = not any(z in zones for z in v)
        elif k == "objects_any":
            ok = any(o in objects for o in v)
        elif k == "objects_all":
            ok = all(o in objects for o in v)
        elif k == "signal_any":
            ok = any(o in objects for o in v.get("objects", ())) or any(z in zones for z in v.get("zones", ()))
        elif k == "min_dwell_s":
            ok = dwell >= v
        elif k == "max_dwell_s":
            ok = dwell <= v
        elif k == "posture":
            ok = post == v
        elif k == "state":
            ok = state == v
        else:
            raise ValueError(f"unknown rule condition {k!r}")
        if not ok:
            return False
    return True


class RuleActivityFuser:
    def __init__(self, rules: Optional[dict[str, dict]] = None, frame_aspect: float = 16 / 9,
                 max_tracks: int = 256):
        self.rules = DEFAULT_RULES if rules is None else rules
        self.frame_aspect = frame_aspect
        self.last: Optional[FusionResult] = None
        self._last_activity: "OrderedDict[Any, str]" = OrderedDict()
        self.max_tracks = max_tracks

    def fuse_detailed(self, hyp: ActivityHypothesis, track: Track, objects_in_zone: Sequence[str],
                      dwell_s: float) -> FusionResult:
        act, conf = hyp.activity, max(0.0, min(1.0, hyp.confidence))
        reasons: list[str] = [f"vlm:{act}@{conf:.2f}"]
        rule = self.rules.get(act)
        corroborated = False
        if act == "unknown" or rule is None:
            if rule is None and act != "unknown":
                reasons.append("no_rule:vlm_only")
            res = FusionResult(act, conf, act, conf, False, tuple(reasons))
            self.last = res
            return res
        zones = track.zones
        post = box_posture(track.box, self.frame_aspect)
        state = getattr(track.state, "value", str(track.state))
        ok_all = True
        for cond in rule.get("requires", ()):
            if _cond_ok(cond, zones, objects_in_zone, dwell_s, post, state):
                reasons.append(f"ok:{json.dumps(cond, sort_keys=True)}")
            else:
                ok_all = False
                reasons.append(f"missing:{json.dumps(cond, sort_keys=True)}")
        if conf < rule.get("min_vlm_conf", 0.0):
            ok_all = False
            reasons.append(f"vlm_conf_below:{rule['min_vlm_conf']}")
        pen = rule.get("penalty", 0.5)
        if ok_all:
            corroborated = bool(rule.get("requires"))
            conf = min(1.0, conf + (rule.get("bonus", 0.1) if corroborated else 0.0))
        elif rule.get("on_fail", "cap") == "reject":
            conf = min(conf * pen, 0.3)
            reasons.append(f"rejected:{act}->unknown")
            act = "unknown"
        else:
            conf *= pen
            reasons.append(f"capped:x{pen}")
        if act != "unknown":
            for c in rule.get("contradicts", ()):
                if _cond_ok(c["if"], zones, objects_in_zone, dwell_s, post, state):
                    conf *= c.get("penalty", 0.5)
                    corroborated = False
                    reasons.append(f"contradiction:{json.dumps(c['if'], sort_keys=True)}")
            if "max_conf" in rule and conf > rule["max_conf"]:
                conf = rule["max_conf"]
                reasons.append(f"max_conf:{rule['max_conf']}")
        res = FusionResult(act, round(max(0.0, min(1.0, conf)), 4), hyp.activity, hyp.confidence,
                           corroborated, tuple(reasons))
        self.last = res
        return res

    # ActivityFuser protocol
    def fuse(self, hyp: ActivityHypothesis, track: Track, objects_in_zone: Sequence[str],
             dwell_s: float) -> tuple[str, float]:
        r = self.fuse_detailed(hyp, track, objects_in_zone, dwell_s)
        return r.activity, r.confidence

    # -- events ---------------------------------------------------------------
    def to_event(self, res: FusionResult, track: Track, *, camera: str = "default",
                 subject_id: Optional[str] = None, subject_label: Optional[str] = None,
                 profiles: tuple[str, ...] = (), ts: Optional[float] = None,
                 snapshot_paths: tuple[str, ...] = ()) -> BrainEvent:
        """source='camera' only if hard signals corroborated the VLM; otherwise
        'inferred' with confidence capped at 0.6."""
        src = "camera" if res.corroborated else "inferred"
        conf = res.confidence if res.corroborated else min(res.confidence, INFERRED_CAP)
        kw = {"ts": ts} if ts is not None else {}
        return new_event(
            "activity_changed", subject_id=subject_id, subject_label=subject_label,
            zone=track.zones[0] if track.zones else None, camera=camera, activity=res.activity,
            confidence=conf, source=src, snapshot_paths=snapshot_paths, profiles=profiles,
            detail={"track_id": track.track_id, "vlm_activity": res.vlm_activity,
                    "vlm_confidence": res.vlm_confidence, "reasons": list(res.reasons)}, **kw)

    def event_if_changed(self, res: FusionResult, track: Track, **kw) -> Optional[BrainEvent]:
        """Emit only when the fused activity for this track differs from the last one.
        'unknown' never produces an event."""
        key = track.track_id
        prev = self._last_activity.get(key)
        self._last_activity[key] = res.activity
        self._last_activity.move_to_end(key)
        while len(self._last_activity) > self.max_tracks:
            self._last_activity.popitem(last=False)
        if res.activity == "unknown" or res.activity == prev:
            return None
        return self.to_event(res, track, **kw)

    def release(self, track_id: int) -> None:
        self._last_activity.pop(track_id, None)
