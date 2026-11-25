"""Deterministic day summaries (+ optional LLM polish). See docs/brain.md.

`build_day_summary` is the source of truth: stats and a short factual text built
only from events/routines. `polish_summary` may rewrite the prose with a
ModelClient (task "summarize"); on any failure or a suspicious answer it returns
the deterministic summary unchanged. Stats are never taken from the model.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, tzinfo
from typing import Optional, Sequence

from .api import BrainEvent, DaySummary, Routine
from .routines import _dt, fmt_minute

PRESENCE_KINDS = ("entered", "seen", "left", "zone_changed", "stationary")


def day_bounds(day: str, tz: Optional[tzinfo] = None) -> tuple[float, float]:
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=tz)
    d1 = (d0 + timedelta(days=1)).replace(tzinfo=tz) if tz else d0 + timedelta(days=1)
    return d0.timestamp(), d1.timestamp()


def _hhmm(ts: float, tz) -> str:
    return _dt(ts, tz).strftime("%H:%M")


def _missed_routines(events: Sequence[BrainEvent], routines: Sequence[Routine], day: str, tz,
                     now: Optional[float] = None) -> list[dict]:
    d0, d1 = day_bounds(day, tz)
    wd = datetime.strptime(day, "%Y-%m-%d").weekday()
    missed = []
    for r in routines:
        if r.kind not in ("daily_at", "weekday_at"):
            continue
        if r.kind == "weekday_at" and wd not in (r.params.get("weekdays") or ()):
            continue
        m = r.params.get("minute_of_day", 0)
        tol = r.params.get("tolerance_min", 30)
        lo, hi = d0 + (m - tol) * 60, d0 + (m + tol) * 60
        hit = any(e.subject_id == r.subject_id and e.kind == r.params.get("event_kind", e.kind)
                  and (not r.params.get("activity") or e.activity == r.params["activity"])
                  and lo <= e.ts <= hi for e in events)
        if not hit and hi <= d1 and (now is None or hi <= now):
            missed.append({"routine_id": r.id, "subject_id": r.subject_id,
                           "description": r.description, "expected": fmt_minute(m)})
    return missed


def build_day_summary(events: Sequence[BrainEvent], routines: Sequence[Routine], day: str,
                      profile: Optional[str] = None, *, tz: Optional[tzinfo] = None,
                      now: Optional[float] = None) -> DaySummary:
    d0, d1 = day_bounds(day, tz)
    evs = sorted((e for e in events if d0 <= e.ts < d1 and (profile is None or profile in e.profiles
                                                              or not e.profiles)), key=lambda e: e.ts)
    pres = [e for e in evs if e.kind in PRESENCE_KINDS]
    visitors: dict[str, str] = {}
    unknown_entries = 0
    for e in evs:
        if e.kind in PRESENCE_KINDS and e.subject_id:
            visitors[e.subject_id] = e.subject_label or e.subject_id
        if e.kind == "entered" and not e.subject_id:
            unknown_entries += 1
    # zone dwell: per subject, time between a presence event and the next zone change / left
    dwell: dict[str, float] = defaultdict(float)
    cur: dict[str, tuple[Optional[str], float]] = {}
    for e in pres:
        k = e.subject_id or f"?{e.detail.get('cluster_id') or e.detail.get('track_id')}"
        if k in cur:
            z, t = cur[k]
            if z:
                dwell[z] += max(0.0, e.ts - t)
        if e.kind == "left":
            cur.pop(k, None)
        else:
            cur[k] = (e.zone, e.ts)
    acts = Counter(e.activity for e in evs if e.kind == "activity_changed" and e.activity)
    alerts = [e for e in evs if e.kind == "alert"]
    statements = [e for e in evs if e.kind == "owner_statement"]
    missed = _missed_routines(evs, [r for r in routines if profile is None or not r.profiles or profile in r.profiles],
                              day, tz, now)
    anomalies = [f"missed routine: {m['description']} (expected ~{m['expected']})" for m in missed]
    anomalies += [f"alert: {e.activity or e.detail.get('reason') or e.subject_label or 'flagged'} at {_hhmm(e.ts, tz)}"
                  for e in alerts]
    stats = {
        "events": len(evs),
        "first_seen": _hhmm(pres[0].ts, tz) if pres else None,
        "last_seen": _hhmm(pres[-1].ts, tz) if pres else None,
        "visitors": sorted(visitors.values()),
        "unknown_visits": unknown_entries,
        "zone_dwell_min": {z: round(s / 60, 1) for z, s in sorted(dwell.items(), key=lambda kv: -kv[1])},
        "activities": dict(acts.most_common()),
        "alerts": len(alerts),
        "owner_statements": len(statements),
        "missed_routines": missed,
        "anomalies": anomalies,
    }
    return DaySummary(day=day, text=_text(day, stats), event_ids=tuple(e.id for e in evs),
                      stats=stats, profile=profile)


def _text(day: str, s: dict) -> str:
    if not s["events"]:
        return f"{day}: no events recorded."
    parts = []
    if s["first_seen"]:
        parts.append(f"Activity seen from {s['first_seen']} to {s['last_seen']}.")
    if s["visitors"]:
        parts.append("Seen: " + ", ".join(s["visitors"]) + ".")
    if s["unknown_visits"]:
        parts.append(f"{s['unknown_visits']} unidentified visit(s).")
    if s["zone_dwell_min"]:
        parts.append("Time by zone: " + ", ".join(f"{z} {m:g} min" for z, m in list(s["zone_dwell_min"].items())[:4]) + ".")
    if s["activities"]:
        parts.append("Activities: " + ", ".join(f"{a.replace('_', ' ')} x{n}" for a, n in s["activities"].items()) + ".")
    if s["owner_statements"]:
        parts.append(f"{s['owner_statements']} owner statement(s) logged.")
    parts += [a[0].upper() + a[1:] + "." for a in s["anomalies"]]
    if not s["anomalies"]:
        parts.append("Nothing unusual.")
    return f"{day}: " + " ".join(parts)


_NUM = re.compile(r"\d+(?::\d+)?")


def polish_summary(summary: DaySummary, client, *, max_tokens: int = 300) -> DaySummary:
    """Ask `client` (task "summarize") to rewrite `summary.text` as friendly prose.
    Falls back to the deterministic text if the call fails, the answer is empty or
    too long, or it mentions numbers/times that are not in the facts."""
    from dataclasses import replace
    from ..models.api import GenRequest
    facts = summary.text
    req = GenRequest(task="summarize", max_tokens=max_tokens,
                     system="Rewrite the facts as 2-4 short, plain sentences. Do not add, infer or omit facts; "
                            "keep every number and time exactly.",
                     prompt=facts)
    try:
        out = (client.generate(req).text or "").strip()
    except Exception:
        return summary
    allowed = set(_NUM.findall(facts))
    if not out or len(out) > 3 * len(facts) + 200 or any(n not in allowed for n in _NUM.findall(out)):
        return summary
    return replace(summary, text=out, stats={**summary.stats, "polished": True, "facts_text": facts})
