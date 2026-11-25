"""Owner-only admin for the brain's timeline.

    python -m worldcam.cli.brain_admin [--sqlite PATH] stats|timeline|routines|mine|summary|purge|forget

Default backend is Neo4j (.env); `--sqlite PATH` uses a local file instead.
`purge`/`forget` print the snapshot paths and delete them only with --delete-files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _store(a):
    if a.sqlite:
        from worldcam.brain.store import SqliteEventStore
        return SqliteEventStore(a.sqlite)
    from worldcam.brain.store import Neo4jEventStore
    return Neo4jEventStore.from_env()


def _ts(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"


def _since(a) -> float | None:
    if a.since is None:
        return None
    s = a.since
    if s[-1:] in "hdm" and s[:-1].replace(".", "").isdigit():
        return time.time() - float(s[:-1]) * {"h": 3600, "d": 86400, "m": 60}[s[-1]]
    return time.mktime(time.strptime(s, "%Y-%m-%d" if len(s) == 10 else "%Y-%m-%d %H:%M"))


def cmd_stats(st, a):
    out = {"events": st.count()}
    kinds = ("entered", "left", "zone_changed", "stationary", "seen", "activity_changed", "alert",
             "owner_statement", "observation")
    out["by_kind"] = {k: n for k in kinds if (n := st.count(kind=k))}
    rts = st.list_routines()
    out["routines"] = {"inferred": sum(r.source != "user" for r in rts), "user": sum(r.source == "user" for r in rts)}
    old = st.query(limit=1, newest_first=False)
    new = st.query(limit=1)
    out["span"] = [_ts(old[0].ts), _ts(new[0].ts)] if old else None
    print(json.dumps(out, indent=2))


def cmd_timeline(st, a):
    for e in st.query(since=_since(a), subject_id=a.subject, kind=a.kind, limit=a.limit):
        print(f"{_ts(e.ts)}  {e.kind:<16} {e.subject_label or e.subject_id or '?':<18} "
              f"{e.zone or '-':<10} {e.activity or ''} {'[snap]' if e.snapshot_paths else ''}")


def cmd_routines(st, a):
    now = time.time()
    state = {s.routine.id: s for s in st.status(now, a.subject)}
    for r in st.list_routines(a.subject):
        s = state.get(r.id)
        print(f"{r.id}  [{r.source}] {r.subject_id}: {r.description}  "
              f"support={r.support} conf={r.confidence:.2f}  status={s.state if s else '?'}")


def cmd_mine(st, a):
    from worldcam.brain.routines import apply_mining
    for r in apply_mining(st, st, lookback_days=a.days):
        print(f"{r.id}  {r.subject_id}: {r.description} support={r.support} conf={r.confidence:.2f}")


def cmd_summary(st, a):
    from worldcam.brain.summary import build_day_summary, day_bounds
    day = a.day or time.strftime("%Y-%m-%d")
    d0, d1 = day_bounds(day)
    s = build_day_summary(st.query(since=d0, until=d1, limit=10**6, newest_first=False),
                          st.list_routines(), day, a.profile)
    print(s.text)
    if a.json:
        print(json.dumps(s.stats, indent=2, default=str))


def cmd_purge(st, a):
    from worldcam.brain.retention import RetentionPolicy
    days = dict(kv.split("=") for kv in a.profile_days or [])
    pol = RetentionPolicy({k: float(v) for k, v in days.items()}, default_days=a.days)
    paths = pol.apply(st, time.time(), delete=a.delete_files)
    print(f"purged; {len(paths)} snapshot file(s) {'deleted' if a.delete_files else 'to delete'}")
    for p in paths:
        print(" ", p)


def cmd_forget(st, a):
    from worldcam.brain.retention import forget_subject
    if not a.yes:
        sys.exit("refusing without --yes")
    paths = forget_subject(st, a.subject, delete=a.delete_files)
    print(f"forgot {a.subject}; {len(paths)} snapshot file(s)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="worldcam-brain")
    ap.add_argument("--sqlite", help="use this SQLite file instead of Neo4j")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("timeline")
    p.add_argument("--subject")
    p.add_argument("--kind")
    p.add_argument("--since", help="24h, 7d, 2026-11-25 or '2026-11-25 08:00'")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_timeline)
    p = sub.add_parser("routines")
    p.add_argument("--subject")
    p.set_defaults(fn=cmd_routines)
    p = sub.add_parser("mine")
    p.add_argument("--days", type=int, default=60)
    p.set_defaults(fn=cmd_mine)
    p = sub.add_parser("summary")
    p.add_argument("--day", help="YYYY-MM-DD (default today)")
    p.add_argument("--profile")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("purge")
    p.add_argument("--days", type=float, required=True, help="default retention in days")
    p.add_argument("--profile-days", nargs="*", metavar="PROFILE=DAYS")
    p.add_argument("--delete-files", action="store_true")
    p.set_defaults(fn=cmd_purge)
    p = sub.add_parser("forget")
    p.add_argument("subject")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--delete-files", action="store_true")
    p.set_defaults(fn=cmd_forget)
    a = ap.parse_args(argv)
    st = _store(a)
    try:
        a.fn(st, a)
    finally:
        getattr(st, "close", lambda: None)()


if __name__ == "__main__":
    main()
