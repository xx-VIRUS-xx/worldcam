"""Routine mining + status. Plain statistics, no ML. See docs/brain.md.

Mining: events are grouped per (subject, event kind, activity). For each group
the FIRST matching event of each local day gives one minute-of-day sample.
Samples are clustered (1-D, greedy, window = 2 * tolerance); a cluster becomes a
routine when it spans >= `min_support` DISTINCT days and covers >= `min_coverage`
of the days the subject was observed in the cluster's span. `daily_at` when it
happens (nearly) every day, `weekday_at` when it is regular on a fixed subset of
weekdays. `presence_window` comes from first-arrival / last-departure per day.
Samples are not wrapped across midnight (a 23:50 vs 00:10 habit is two clusters).

Owner-declared routines (source="user") are never modified or deleted by mining,
and an inferred routine duplicating a user one is not created.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, tzinfo
from typing import Iterable, Optional, Sequence

from .api import BrainEvent, Routine, RoutineStatus

MINING_KINDS = ("entered", "activity_changed")
DAY_S = 86400.0


def _dt(ts: float, tz: Optional[tzinfo]) -> datetime:
    return datetime.fromtimestamp(ts, tz)


def minute_of_day(ts: float, tz: Optional[tzinfo] = None) -> int:
    d = _dt(ts, tz)
    return d.hour * 60 + d.minute


def day_start(ts: float, tz: Optional[tzinfo] = None) -> float:
    d = _dt(ts, tz)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def fmt_minute(m: float) -> str:
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


_WD = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _cluster(samples: list[tuple[int, str, int]], window: int) -> list[list[tuple[int, str, int]]]:
    """samples: (minute, day_key, weekday). Greedy 1-D clustering, cluster width <= window."""
    out: list[list] = []
    for s in sorted(samples):
        if out and s[0] - out[-1][0][0] <= window:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def _rid(*parts) -> str:
    return "rt_" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _confidence(n_days: int, coverage: float, std: float, tol: int) -> float:
    regular = max(0.0, 1.0 - std / max(1.0, 2.0 * tol))
    support = n_days / (n_days + 2.0)
    return round(min(0.95, coverage * (0.5 + 0.5 * regular) * (0.6 + 0.4 * support)), 3)


def _label(event_kind: str, activity: Optional[str]) -> str:
    if activity:
        return activity.replace("_", " ")
    return {"entered": "shows up", "activity_changed": "does something"}.get(event_kind, event_kind)


def mine_routines(events: Iterable[BrainEvent], *, tz: Optional[tzinfo] = None, tolerance_min: int = 45,
                  min_support: int = 3, min_coverage: float = 0.6, min_confidence: float = 0.35,
                  kinds: Sequence[str] = MINING_KINDS, profiles_for: Optional[dict] = None) -> list[Routine]:
    """Pure function: events -> inferred routines (not stored)."""
    by_subject: dict[str, list[BrainEvent]] = defaultdict(list)
    for e in events:
        if e.subject_id:
            by_subject[e.subject_id].append(e)
    out: list[Routine] = []
    window = 2 * tolerance_min
    for sid, evs in sorted(by_subject.items()):
        observed_days: dict[str, int] = {}          # day key -> weekday
        for e in evs:
            d = _dt(e.ts, tz)
            observed_days[d.strftime("%Y-%m-%d")] = d.weekday()
        groups: dict[tuple, dict[str, BrainEvent]] = defaultdict(dict)   # key -> day -> first event
        for e in sorted(evs, key=lambda x: x.ts):
            if e.kind not in kinds:
                continue
            day = _dt(e.ts, tz).strftime("%Y-%m-%d")
            groups[(e.kind, e.activity)].setdefault(day, e)
        for (kind, activity), per_day in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
            samples = [(minute_of_day(e.ts, tz), day, _dt(e.ts, tz).weekday()) for day, e in per_day.items()]
            for cl in _cluster(samples, window):
                days = {s[1] for s in cl}
                if len(days) < min_support:
                    continue
                r = _routine_from_cluster(sid, kind, activity, cl, observed_days, per_day, tolerance_min,
                                          min_coverage, tz)
                if r and r.confidence >= min_confidence:
                    if profiles_for:
                        r = _with_profiles(r, profiles_for)
                    out.append(r)
        pw = _presence_window(sid, evs, observed_days, tolerance_min, min_support, min_coverage, tz)
        if pw and pw.confidence >= min_confidence:
            out.append(_with_profiles(pw, profiles_for) if profiles_for else pw)
    return out


def _with_profiles(r: Routine, profiles_for: dict) -> Routine:
    from dataclasses import replace
    prof = profiles_for.get(r.params.get("activity") or r.params.get("event_kind"))
    return replace(r, profiles=tuple(prof)) if prof else r


def _routine_from_cluster(sid, kind, activity, cl, observed_days, per_day, tol, min_cov, tz):
    mins = [s[0] for s in cl]
    days = sorted({s[1] for s in cl})
    wds = {s[2] for s in cl}
    center = int(round(statistics.median(mins)))
    std = statistics.pstdev(mins) if len(mins) > 1 else 0.0
    t = int(min(tol, max(15, math.ceil(2 * std)))) if std else 15
    t = max(t, int(max(abs(m - center) for m in mins)))
    t = min(t, tol)
    gdays = sorted(per_day)
    span_days = [d for d in observed_days if gdays[0] <= d <= gdays[-1]]
    cov_all = len(days) / max(1, len(span_days))
    in_set = [d for d in span_days if observed_days[d] in wds]
    cov_wd = len(days) / max(1, len(in_set))
    weekdays = None
    if len(wds) < 7 and len(days) >= 2 * len(wds) and cov_wd >= min_cov and cov_wd - cov_all >= 0.15:
        rk, coverage, weekdays = "weekday_at", cov_wd, sorted(wds)
    elif cov_all >= min_cov:
        rk, coverage = "daily_at", cov_all
    else:
        return None
    params = {"event_kind": kind, "activity": activity, "minute_of_day": center, "tolerance_min": t}
    if weekdays is not None:
        params["weekdays"] = weekdays
    when = f"around {fmt_minute(center)}"
    where = "" if weekdays is None else " on " + "/".join(_WD[w] for w in weekdays)
    verb = _label(kind, activity)
    desc = (f"usually {verb} {when}{where}" if kind == "activity_changed"
            else f"usually {verb} {when}{where}")
    first = per_day[days[0]].ts
    last = per_day[days[-1]].ts
    return Routine(id=_rid(sid, rk, kind, activity, center // max(1, 2 * tol)), subject_id=sid, kind=rk,
                   description=desc + f" (+-{t} min)", params=params, support=len(days),
                   confidence=_confidence(len(days), coverage, std, t), first_seen=first, last_seen=last,
                   source="inferred")


def _presence_window(sid, evs, observed_days, tol, min_support, min_cov, tz) -> Optional[Routine]:
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for e in sorted(evs, key=lambda x: x.ts):
        if e.kind not in ("entered", "seen", "left"):
            continue
        day = _dt(e.ts, tz).strftime("%Y-%m-%d")
        if e.kind in ("entered", "seen"):
            first.setdefault(day, e.ts)
        if e.kind in ("left", "seen"):
            last[day] = e.ts
    days = sorted(d for d in first if d in last and last[d] - first[d] >= 1800)
    if len(days) < min_support:
        return None
    starts = [minute_of_day(first[d], tz) for d in days]
    ends = [minute_of_day(last[d], tz) for d in days]
    ss, es = statistics.pstdev(starts), statistics.pstdev(ends)
    window = 2 * tol
    if max(starts) - min(starts) > window or max(ends) - min(ends) > window:
        return None
    span = [d for d in observed_days if days[0] <= d <= days[-1]]
    coverage = len(days) / max(1, len(span))
    if coverage < min_cov:
        return None
    sm, em = int(round(statistics.median(starts))), int(round(statistics.median(ends)))
    t = int(min(tol, max(15, math.ceil(2 * max(ss, es)))))
    return Routine(id=_rid(sid, "presence_window"), subject_id=sid, kind="presence_window",
                   description=f"usually present {fmt_minute(sm)}-{fmt_minute(em)} (+-{t} min)",
                   params={"start_minute": sm, "end_minute": em, "tolerance_min": t},
                   support=len(days), confidence=_confidence(len(days), coverage, max(ss, es), t),
                   first_seen=first[days[0]], last_seen=last[days[-1]], source="inferred")


def _duplicates_user(r: Routine, user_rts: list[Routine]) -> bool:
    for u in user_rts:
        if u.subject_id != r.subject_id or u.kind != r.kind:
            continue
        p, q = r.params, u.params
        if r.kind == "presence_window":
            return True
        if (p.get("event_kind") == q.get("event_kind") and p.get("activity") == q.get("activity")
                and abs(p.get("minute_of_day", 0) - q.get("minute_of_day", 10**6))
                <= p.get("tolerance_min", 0) + q.get("tolerance_min", 0)):
            return True
    return False


def apply_mining(events_store, routine_store=None, *, since: Optional[float] = None, now: Optional[float] = None,
                 lookback_days: int = 60, **kw) -> list[Routine]:
    """Mine the store's events and write inferred routines. Stale inferred
    routines of the mined subjects are removed; source="user" ones are untouched."""
    import time as _t
    routine_store = routine_store or events_store
    now = _t.time() if now is None else now
    since = now - lookback_days * DAY_S if since is None else since
    evs = events_store.query(since=since, until=now + 1, limit=10**7, newest_first=False)
    mined = mine_routines(evs, **kw)
    existing = routine_store.list_routines()
    user_rts = [r for r in existing if r.source == "user"]
    mined = [r for r in mined if not _duplicates_user(r, user_rts)]
    keep = {r.id for r in mined}
    subjects = {e.subject_id for e in evs if e.subject_id}
    for r in existing:
        if r.source != "user" and r.subject_id in subjects and r.id not in keep:
            routine_store.delete_routine(r.id)
    for r in mined:
        routine_store.upsert_routine(r)
    return mined


# ------------------------------------------------------------------- status
def _matches(rt: Routine, e: BrainEvent) -> bool:
    p = rt.params
    if p.get("event_kind") and e.kind != p["event_kind"]:
        return False
    if p.get("activity") and e.activity != p["activity"]:
        return False
    if p.get("zone") and e.zone != p["zone"]:
        return False
    return True


def routine_status(events, routines: Sequence[Routine], now: float, tz: Optional[tzinfo] = None) -> list[RoutineStatus]:
    """on_time (done, or nothing owed yet), due (inside the window, not done),
    overdue (window passed, not done), not_applicable (other weekday / outside window)."""
    day0 = day_start(now, tz)
    out = []
    for rt in routines:
        p = rt.params
        if rt.kind in ("daily_at", "weekday_at"):
            if rt.kind == "weekday_at" and _dt(now, tz).weekday() not in (p.get("weekdays") or ()):
                out.append(RoutineStatus(rt, "not_applicable"))
                continue
            tol = p.get("tolerance_min", 30) * 60
            exp = day0 + p.get("minute_of_day", 0) * 60
            # DST-safe: recompute from local wall clock
            exp = _dt(day0, tz).replace(hour=p.get("minute_of_day", 0) // 60,
                                        minute=p.get("minute_of_day", 0) % 60).timestamp()
            done = [e for e in events.query(subject_id=rt.subject_id, kind=p.get("event_kind"),
                                            since=exp - tol, until=now + 1, limit=200)
                    if _matches(rt, e)]
            last = max((e.ts for e in done), default=None)
            if last is not None:
                state = "on_time"
            elif now < exp - tol:
                state = "on_time"
            elif now <= exp + tol:
                state = "due"
            else:
                state = "overdue"
            out.append(RoutineStatus(rt, state, expected_at=exp, last_done=last))
        elif rt.kind == "presence_window":
            lo = day0 + (p["start_minute"] - p["tolerance_min"]) * 60
            hi = day0 + (p["end_minute"] + p["tolerance_min"]) * 60
            start = day0 + p["start_minute"] * 60
            if not (lo <= now <= hi):
                out.append(RoutineStatus(rt, "not_applicable", expected_at=start))
                continue
            evs = events.query(subject_id=rt.subject_id, since=day0, until=now + 1, limit=1)
            present = bool(evs) and evs[0].kind != "left"
            if present:
                state = "on_time"
            elif now <= start + p["tolerance_min"] * 60:
                state = "due"
            else:
                state = "overdue"
            out.append(RoutineStatus(rt, state, expected_at=start, last_done=evs[0].ts if evs else None))
        elif rt.kind == "interval":
            every = float(p.get("interval_min", 0)) * 60
            ev = [e for e in events.query(subject_id=rt.subject_id, kind=p.get("event_kind"),
                                          since=now - 30 * DAY_S, until=now + 1, limit=200) if _matches(rt, e)]
            last = ev[0].ts if ev else None
            if every <= 0:
                out.append(RoutineStatus(rt, "not_applicable"))
            else:
                age = now - last if last is not None else math.inf
                state = "on_time" if age <= every else ("due" if age <= every * 1.25 else "overdue")
                out.append(RoutineStatus(rt, state, expected_at=(last + every) if last else None, last_done=last))
        else:
            out.append(RoutineStatus(rt, "not_applicable"))
    return out
