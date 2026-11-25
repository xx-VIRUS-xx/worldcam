"""Profiles are data. Load, validate, merge and evaluate them.

A profile JSON has the fields of `agent.api.Profile`. Alert rule:
  {"id": str, "when": {...}, "to_permission": "receive_*", "priority": 0-3,
   "cooldown_min": int, "text": "template with {zone} {time} {routine} {minutes} {subject}"}
`when` keys (all optional, all must match):
  event_kind (str|list), subject ("unknown"|"known"|"any"), zone (str|list, substring, case-insens.),
  activity (str|list), label (str|list), min_confidence (float),
  night (bool; uses `hours`, default 22-06), hours ([start, end) local; for event rules the event
  time must fall inside, for state rules "now" must),
  routine_state ("overdue" | "due": state rule over RoutineStore.status),
  no_movement_minutes (int: state rule, nothing seen for that long while the system has events).
Check-in: {"id", "every_hours", "jitter" 0-1, "prompt", "quiet" (silent message),
  "hours": [start, end) window, "audience": "permission" | "self", "to_permission"}.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from ..brain.api import BrainEvent
from ..users.api import PERMISSIONS, ROLES
from .api import Profile

BUILTIN_DIR = Path(__file__).with_name("profiles")
WHEN_KEYS = {"event_kind", "subject", "zone", "activity", "label", "min_confidence", "night",
             "hours", "routine_state", "no_movement_minutes"}
STATE_KEYS = {"routine_state", "no_movement_minutes"}
PROFILE_FIELDS = {"name", "description", "goals", "notice_activities", "notice_labels", "ask_topics",
                  "alert_rules", "checkins", "tone", "allowed_roles", "retention_days",
                  "send_images_to_cloud", "enabled"}
DEFAULT_NIGHT = (22, 6)


class ProfileError(ValueError):
    pass


def _tuple_of_str(d: dict, key: str, errs: list) -> tuple:
    v = d.get(key, ())
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
        errs.append(f"{key} must be a list of strings")
        return ()
    return tuple(v)


def _hours(v, where: str, errs: list) -> None:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(h, int) and 0 <= h <= 24 for h in v)):
        errs.append(f"{where}: hours must be [start, end] ints 0-24")


def validate_profile_dict(d: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["profile must be an object"]
    for k in d:
        if k not in PROFILE_FIELDS:
            errs.append(f"unknown field {k!r}")
    if not isinstance(d.get("name"), str) or not d.get("name"):
        errs.append("name is required")
    for k in ("goals", "notice_activities", "notice_labels", "ask_topics", "allowed_roles"):
        _tuple_of_str(d, k, errs)
    for r in d.get("allowed_roles", ()):
        if r not in ROLES:
            errs.append(f"allowed_roles: bad role {r!r}")
    if not isinstance(d.get("retention_days", 30), int) or d.get("retention_days", 30) < 0:
        errs.append("retention_days must be a non-negative int")
    if not isinstance(d.get("send_images_to_cloud", False), bool):
        errs.append("send_images_to_cloud must be a bool")
    for i, r in enumerate(d.get("alert_rules", ())):
        w = f"alert_rules[{i}]"
        if not isinstance(r, dict) or not isinstance(r.get("when"), dict):
            errs.append(f"{w}: needs a 'when' object")
            continue
        for k in r["when"]:
            if k not in WHEN_KEYS:
                errs.append(f"{w}: unknown when-key {k!r}")
        if "hours" in r["when"]:
            _hours(r["when"]["hours"], w, errs)
        if r.get("to_permission") not in PERMISSIONS:
            errs.append(f"{w}: bad to_permission {r.get('to_permission')!r}")
        if not isinstance(r.get("priority", 1), int) or not 0 <= r.get("priority", 1) <= 3:
            errs.append(f"{w}: priority must be 0-3")
        if not isinstance(r.get("text"), str) or not r["text"]:
            errs.append(f"{w}: text is required")
        if r["when"].get("subject", "any") not in ("unknown", "known", "any"):
            errs.append(f"{w}: subject must be unknown|known|any")
    for i, c in enumerate(d.get("checkins", ())):
        w = f"checkins[{i}]"
        if not isinstance(c, dict):
            errs.append(f"{w}: must be an object")
            continue
        if not isinstance(c.get("every_hours"), (int, float)) or c["every_hours"] <= 0:
            errs.append(f"{w}: every_hours must be > 0")
        if not isinstance(c.get("jitter", 0.0), (int, float)) or not 0 <= c.get("jitter", 0.0) <= 1:
            errs.append(f"{w}: jitter must be 0-1")
        if not isinstance(c.get("prompt"), str) or not c["prompt"]:
            errs.append(f"{w}: prompt is required")
        if c.get("audience", "permission") not in ("permission", "self"):
            errs.append(f"{w}: audience must be permission|self")
        if "to_permission" in c and c["to_permission"] not in PERMISSIONS:
            errs.append(f"{w}: bad to_permission")
        if "hours" in c:
            _hours(c["hours"], w, errs)
    return errs


def profile_from_dict(d: dict) -> Profile:
    errs = validate_profile_dict(d)
    if errs:
        raise ProfileError(f"profile {d.get('name') if isinstance(d, dict) else '?'}: " + "; ".join(errs))
    name = d["name"]
    rules = []
    for i, r in enumerate(d.get("alert_rules", ())):
        r = dict(r)
        r["id"] = f"{name}:{r.get('id') or i}"
        r["profile"] = name
        r.setdefault("priority", 1)
        rules.append(r)
    checkins = []
    for i, c in enumerate(d.get("checkins", ())):
        c = dict(c)
        c["id"] = f"{name}:{c.get('id') or i}"
        c["profile"] = name
        c.setdefault("jitter", 0.0)
        c.setdefault("audience", "permission")
        c.setdefault("to_permission", "receive_companion")
        checkins.append(c)
    return Profile(
        name=name, description=d.get("description", ""),
        goals=tuple(d.get("goals", ())), notice_activities=tuple(d.get("notice_activities", ())),
        notice_labels=tuple(d.get("notice_labels", ())), ask_topics=tuple(d.get("ask_topics", ())),
        alert_rules=tuple(rules), checkins=tuple(checkins), tone=d.get("tone", "neutral"),
        allowed_roles=tuple(d.get("allowed_roles", ("owner",))),
        retention_days=d.get("retention_days", 30),
        send_images_to_cloud=d.get("send_images_to_cloud", False), enabled=d.get("enabled", True))


def load_profile(path: Union[str, Path]) -> Profile:
    return profile_from_dict(json.loads(Path(path).read_text()))


def builtin_names() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.json"))


def load_builtin(name: str) -> Profile:
    p = BUILTIN_DIR / f"{name}.json"
    if not p.exists():
        raise ProfileError(f"no built-in profile {name!r} (have {builtin_names()})")
    return load_profile(p)


def load_profiles(specs: Iterable[str]) -> list[Profile]:
    """Each spec is a built-in name or a path to a JSON file."""
    out = []
    for s in specs:
        out.append(load_profile(s) if s.endswith(".json") or "/" in s else load_builtin(s))
    return out


def merge_profiles(profiles: Sequence[Profile]) -> Profile:
    """Union of vocabularies/goals/rules; strictest privacy wins (no cloud images if any
    profile forbids it; shortest retention)."""
    ps = [p for p in profiles if p.enabled]
    if not ps:
        return Profile(name="none", description="", enabled=False)

    def union(attr):
        seen, out = set(), []
        for p in ps:
            for x in getattr(p, attr):
                key = json.dumps(x, sort_keys=True) if isinstance(x, dict) else x
                if key not in seen:
                    seen.add(key)
                    out.append(x)
        return tuple(out)

    return Profile(
        name="+".join(p.name for p in ps), description=" | ".join(p.description for p in ps if p.description),
        goals=union("goals"), notice_activities=union("notice_activities"), notice_labels=union("notice_labels"),
        ask_topics=union("ask_topics"), alert_rules=union("alert_rules"), checkins=union("checkins"),
        tone=ps[0].tone, allowed_roles=union("allowed_roles"),
        retention_days=min(p.retention_days for p in ps),
        send_images_to_cloud=all(p.send_images_to_cloud for p in ps), enabled=True)


# ---- rule evaluation -----------------------------------------------------------

def in_hours(hours: Optional[Sequence[int]], ts: float) -> bool:
    if not hours:
        return True
    s, e = hours
    h = time.localtime(ts).tm_hour
    return (h >= s or h < e) if s > e else (s <= h < e)


def _any(spec, value: Optional[str], substring: bool = False) -> bool:
    if value is None:
        return False
    specs = [spec] if isinstance(spec, str) else list(spec)
    v = value.lower()
    return any((s.lower() in v) if substring else (s.lower() == v) for s in specs)


def is_unknown_subject(e: BrainEvent) -> bool:
    return e.subject_id is None or (e.subject_label or "").lower().startswith("unknown")


def is_state_rule(rule: dict) -> bool:
    return bool(STATE_KEYS & set(rule["when"]))


def rule_matches_event(rule: dict, e: BrainEvent) -> bool:
    w = rule["when"]
    if is_state_rule(rule) or e.kind == "alert":
        return False
    if "event_kind" in w and not _any(w["event_kind"], e.kind):
        return False
    s = w.get("subject", "any")
    unk = is_unknown_subject(e)
    if (s == "unknown" and not unk) or (s == "known" and unk):
        return False
    if "zone" in w and not _any(w["zone"], e.zone, substring=True):
        return False
    if "activity" in w and not _any(w["activity"], e.activity):
        return False
    if "label" in w and not _any(w["label"], e.subject_label):
        return False
    if e.confidence < w.get("min_confidence", 0.0):
        return False
    if w.get("night") and not in_hours(w.get("hours") or DEFAULT_NIGHT, e.ts):
        return False
    if "hours" in w and not w.get("night") and not in_hours(w["hours"], e.ts):
        return False
    return True


class SafeDict(dict):
    def __missing__(self, key):
        return "?"


def render(template: str, **kw) -> str:
    try:
        return template.format_map(SafeDict(**kw))
    except (ValueError, IndexError):
        return template


class ProfileSet:
    """The active profiles and their merged view."""

    def __init__(self, profiles: Sequence[Profile]):
        self.profiles = [p for p in profiles if p.enabled]
        self.merged = merge_profiles(self.profiles)

    def names(self) -> list[str]:
        return [p.name for p in self.profiles]

    def get(self, name: str) -> Optional[Profile]:
        return next((p for p in self.profiles if p.name == name), None)

    @property
    def rules(self) -> tuple:
        return self.merged.alert_rules

    @property
    def checkins(self) -> tuple:
        return self.merged.checkins

    @property
    def goals(self) -> tuple:
        return self.merged.goals

    def cloud_images_ok(self) -> bool:
        return self.merged.send_images_to_cloud
