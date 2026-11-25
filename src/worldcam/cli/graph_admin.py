"""Owner-only admin for the worldcam Neo4j graph.

    python -m worldcam.cli.graph_admin init|status|people|show|doubts|forget|export|wipe

Connection settings come from .env (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD).
`forget` is the only way to delete a person (the worker has no delete tool).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _fmt_ts(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def _store(init_schema: bool = False):
    from worldcam.graph.store import Neo4jGraphStore
    return Neo4jGraphStore(init_schema=init_schema)


def _resolve_person(store, ref: str):
    p = store.get_person(ref)
    if p:
        return p
    hits = store.search_people(ref, limit=5)
    exact = [h for h in hits if (h.name or "").lower() == ref.lower()]
    if len(exact) == 1 or len(hits) == 1:
        return (exact or hits)[0]
    if hits:
        sys.exit(f"ambiguous: {', '.join(f'{h.id} ({h.name})' for h in hits)}")
    sys.exit(f"no person matches {ref!r}")


def cmd_init(store, a):
    from worldcam.graph.schema import ensure_schema
    stmts = ensure_schema(store._driver, store.database, store.prefix, store.dim)
    print(f"schema ok ({len(stmts)} constraints/indexes)")


def cmd_status(store, a):
    from worldcam.graph.config import Neo4jConfig
    with store._driver.session(database=store.database) as s:
        comp = s.run("CALL dbms.components() YIELD versions, edition RETURN versions[0] AS v, edition").single()
        idx = s.run("SHOW INDEXES YIELD name, type, state RETURN name, type, state ORDER BY name").data()
    print(f"neo4j {comp['v']} ({comp['edition']}) at {Neo4jConfig.from_env().uri}, db={store.database}")
    for label, n in store.stats().items():
        print(f"  {label:<12} {n}")
    print("indexes:")
    for i in idx:
        print(f"  {i['name']:<28} {i['type']:<8} {i['state']}")


def cmd_people(store, a):
    for p in store.list_people(limit=a.limit):
        tag = f"{p.entity_type}/{p.label}" if p.label else p.entity_type
        print(f"{p.id}  {tag:<16} {p.kind:<10} {p.name or '(unnamed)':<24} last seen {_fmt_ts(p.last_seen)}"
              + (f"  aka {', '.join(p.aliases)}" if p.aliases else ""))


def cmd_show(store, a):
    p = _resolve_person(store, a.person)
    ctx = store.person_context(p.id)
    print(f"{p.id}  {p.name or '(unnamed)'}  type={p.entity_type}  label={p.label or '-'}  kind={p.kind}  aliases={list(p.aliases)}")
    print(f"first seen {_fmt_ts(p.first_seen)}, last seen {_fmt_ts(p.last_seen)}")
    if p.persona_summary:
        print(f"persona ({_fmt_ts(p.persona_updated)}): {p.persona_summary}")
    print(f"face clusters: {', '.join(ctx['clusters']) or '-'}")
    print("facts:")
    for f in ctx["facts"]:
        print(f"  {f.key} = {f.value}  [{f.source}, {f.confidence:.2f}, {_fmt_ts(f.ts)}]")
    print("relations:")
    for rel, other, d in ctx["relations"]:
        o = store.get_person(other)
        print(f"  {'->' if d == 'out' else '<-'} {rel} {other} ({o.name if o else '?'})")
    print("recent sightings:")
    for s in ctx["recent_sightings"]:
        print(f"  {_fmt_ts(s.ts)} {s.camera} track={s.track_id} zones={list(s.zones)} {s.activity or ''}")


def cmd_doubts(store, a):
    for d in store.list_doubts(status=a.status, limit=a.limit):
        print(f"{d.id}  {d.kind:<14} {d.status:<9} rounds={d.rounds} created {_fmt_ts(d.created)}"
              f"  about={list(d.about_ids)}  q={d.question or '-'}")


def cmd_forget(store, a):
    p = store.get_person(a.person_id)
    if p is None:
        sys.exit(f"no person {a.person_id}")
    if not a.yes:
        ans = input(f"Hard delete {p.id} ({p.name or 'unnamed'}) and all their faces, sightings, "
                    f"facts and snapshots? Type the id to confirm: ")
        if ans.strip() != p.id:
            sys.exit("aborted")
    paths = store.forget_person(p.id)
    removed = missing = 0
    for sp in paths:
        try:
            Path(sp).unlink()
            removed += 1
        except FileNotFoundError:
            missing += 1
    print(f"forgot {p.id}: {len(paths)} snapshot paths ({removed} files deleted, {missing} already gone)")


def cmd_export(store, a):
    data = store.export()
    data["exported_at"] = time.time()
    Path(a.file).write_text(json.dumps(data, indent=1, default=str))
    n = sum(len(v) for v in data["nodes"].values())
    print(f"wrote {n} nodes, {len(data['edges'])} edges to {a.file}")


def cmd_wipe(store, a):
    if not a.yes_really:
        sys.exit("refusing: pass --yes-really to delete every worldcam node")
    print(f"deleted {store.wipe()} nodes (schema kept)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m worldcam.cli.graph_admin", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create constraints and vector indexes (idempotent)")
    sub.add_parser("status", help="version, node counts, index states")
    sp = sub.add_parser("people", help="list people")
    sp.add_argument("--limit", type=int, default=200)
    sp = sub.add_parser("show", help="one person's context")
    sp.add_argument("person", help="person id or name")
    sp = sub.add_parser("doubts", help="list doubts")
    sp.add_argument("--status", choices=["open", "asked", "answered", "resolved", "dismissed"])
    sp.add_argument("--limit", type=int, default=50)
    sp = sub.add_parser("forget", help="hard delete a person and their snapshot files")
    sp.add_argument("person_id")
    sp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sp = sub.add_parser("export", help="dump the graph to JSON")
    sp.add_argument("file")
    sp = sub.add_parser("wipe", help="delete ALL worldcam data (schema kept)")
    sp.add_argument("--yes-really", action="store_true")
    a = ap.parse_args(argv)

    store = _store(init_schema=False)
    try:
        {"init": cmd_init, "status": cmd_status, "people": cmd_people, "show": cmd_show,
         "doubts": cmd_doubts, "forget": cmd_forget, "export": cmd_export, "wipe": cmd_wipe}[a.cmd](store, a)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
