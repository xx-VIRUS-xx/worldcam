"""Event/routine stores. See docs/brain.md.

Neo4jEventStore: nodes `:Event` / `:Routine` (label-prefixable for tests),
linked `(:Event)-[:ABOUT]->(:Person)` when a Person with that id exists.
SqliteEventStore: same contract on a local file; the recommended store for a
high-volume timeline (see docs/brain.md for measurements).
Both implement EventStore AND RoutineStore; `status()` delegates to routines.py.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from typing import Optional

from .api import EVENT_KINDS, EVENT_SOURCES, BrainEvent, Routine, RoutineStatus

_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,30}$")
_FILTERS = ("since", "until", "subject_id", "kind", "zone", "profile")


def _check_event(e: BrainEvent) -> None:
    if e.kind not in EVENT_KINDS or e.source not in EVENT_SOURCES:
        raise ValueError("bad event kind/source")


def _ev_row(e: BrainEvent) -> dict:
    return dict(id=e.id, ts=float(e.ts), kind=e.kind, subject_id=e.subject_id,
                subject_label=e.subject_label, zone=e.zone, camera=e.camera,
                activity=e.activity, confidence=float(e.confidence), source=e.source,
                snapshot_paths=list(e.snapshot_paths), profiles=list(e.profiles),
                detail=json.dumps(e.detail, default=str, sort_keys=True))


def _ev_from(d: dict) -> BrainEvent:
    return BrainEvent(id=d["id"], ts=d["ts"], kind=d["kind"], subject_id=d.get("subject_id"),
                      subject_label=d.get("subject_label"), zone=d.get("zone"),
                      camera=d.get("camera") or "default", activity=d.get("activity"),
                      confidence=d.get("confidence") or 0.0, source=d.get("source") or "camera",
                      snapshot_paths=tuple(d.get("snapshot_paths") or ()),
                      profiles=tuple(d.get("profiles") or ()),
                      detail=json.loads(d.get("detail") or "{}"))


def _rt_row(r: Routine) -> dict:
    return dict(id=r.id, subject_id=r.subject_id, kind=r.kind, description=r.description,
                params=json.dumps(r.params, default=str, sort_keys=True), support=int(r.support),
                confidence=float(r.confidence), first_seen=float(r.first_seen),
                last_seen=float(r.last_seen), source=r.source, profiles=list(r.profiles))


def _rt_from(d: dict) -> Routine:
    return Routine(id=d["id"], subject_id=d["subject_id"], kind=d["kind"],
                   description=d.get("description") or "", params=json.loads(d.get("params") or "{}"),
                   support=d.get("support") or 0, confidence=d.get("confidence") or 0.0,
                   first_seen=d.get("first_seen") or 0.0, last_seen=d.get("last_seen") or 0.0,
                   source=d.get("source") or "inferred", profiles=tuple(d.get("profiles") or ()))


def _paths(rows) -> list[str]:
    return sorted({p for r in rows for p in (r or ())})


class _RoutineStatusMixin:
    def status(self, now: float, subject_id: Optional[str] = None) -> list[RoutineStatus]:
        from .routines import routine_status
        return routine_status(self, self.list_routines(subject_id), now)


# --------------------------------------------------------------------- Neo4j
class Neo4jEventStore(_RoutineStatusMixin):
    def __init__(self, uri=None, user=None, password=None, *, database=None, label_prefix: str = "",
                 person_prefix: str = "", init_schema: bool = True, driver=None,
                 connection_timeout_s: float = 5.0):
        if label_prefix and not _PREFIX_RE.match(label_prefix):
            raise ValueError(f"bad label prefix {label_prefix!r}")
        from neo4j import GraphDatabase
        cfg = None
        if driver is None:
            from ..graph.config import Neo4jConfig
            cfg = Neo4jConfig.from_env()
        self.database = database or (cfg.database if cfg else "neo4j")
        self.prefix = label_prefix
        self._ev, self._rt = f"{label_prefix}Event", f"{label_prefix}Routine"
        self._person = f"{person_prefix}Person"
        self._owns = driver is None
        self._driver = driver or GraphDatabase.driver(
            uri or cfg.uri, auth=(user or cfg.user, password if password is not None else cfg.password),
            connection_timeout=connection_timeout_s)
        if init_schema:
            self.ensure_schema()

    @classmethod
    def from_env(cls, env_file=None, **kw) -> "Neo4jEventStore":
        from ..graph.config import Neo4jConfig
        cfg = Neo4jConfig.from_env(env_file)
        return cls(cfg.uri, cfg.user, cfg.password, database=cfg.database, **kw)

    def close(self) -> None:
        if self._owns:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- schema
    def _idx(self, base: str) -> str:
        return f"{self.prefix.lower()}_{base}" if self.prefix else base

    def schema_statements(self) -> list[str]:
        ev, rt = self._ev, self._rt
        s = [f"CREATE CONSTRAINT {self._idx('event_id')} IF NOT EXISTS FOR (n:{ev}) REQUIRE n.id IS UNIQUE",
             f"CREATE CONSTRAINT {self._idx('routine_id')} IF NOT EXISTS FOR (n:{rt}) REQUIRE n.id IS UNIQUE"]
        for prop in ("ts", "subject_id", "kind"):
            s.append(f"CREATE INDEX {self._idx('event_' + prop)} IF NOT EXISTS FOR (n:{ev}) ON (n.{prop})")
        s.append(f"CREATE INDEX {self._idx('routine_subject_id')} IF NOT EXISTS FOR (n:{rt}) ON (n.subject_id)")
        return s

    def ensure_schema(self) -> None:
        with self._driver.session(database=self.database) as s:
            for st in self.schema_statements():
                s.run(st).consume()

    def drop_schema(self) -> None:
        with self._driver.session(database=self.database) as s:
            for n in ("event_id", "routine_id"):
                s.run(f"DROP CONSTRAINT {self._idx(n)} IF EXISTS").consume()
            for n in ("event_ts", "event_subject_id", "event_kind", "routine_subject_id"):
                s.run(f"DROP INDEX {self._idx(n)} IF EXISTS").consume()

    def wipe(self) -> None:
        """Delete every node carrying THIS store's labels (test hygiene)."""
        with self._driver.session(database=self.database) as s:
            for lab in (self._ev, self._rt):
                s.run(f"MATCH (n:{lab}) DETACH DELETE n").consume()

    # -- plumbing
    def _w(self, cypher, **p):
        def fn(tx):
            return [r.data() for r in tx.run(cypher, p)]
        with self._driver.session(database=self.database) as s:
            return s.execute_write(fn)

    def _r(self, cypher, **p):
        def fn(tx):
            return [r.data() for r in tx.run(cypher, p)]
        with self._driver.session(database=self.database) as s:
            return s.execute_read(fn)

    def _where(self, f: dict) -> tuple[str, dict]:
        c, p = [], {}
        if f.get("since") is not None:
            c.append("e.ts >= $since")
            p["since"] = float(f["since"])
        if f.get("until") is not None:
            c.append("e.ts < $until")
            p["until"] = float(f["until"])
        for k in ("subject_id", "kind", "zone"):
            if f.get(k) is not None:
                c.append(f"e.{k} = ${k}")
                p[k] = f[k]
        if f.get("profile") is not None:
            c.append("$profile IN e.profiles")
            p["profile"] = f["profile"]
        return (" WHERE " + " AND ".join(c)) if c else "", p

    # -- EventStore
    def add_event(self, e: BrainEvent) -> str:
        _check_event(e)
        self._w(f"""MERGE (e:{self._ev} {{id: $id}}) SET e += $props
                    WITH e OPTIONAL MATCH (p:{self._person} {{id: $subject_id}})
                    FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
                             MERGE (e)-[:ABOUT]->(p))""",
                id=e.id, props={k: v for k, v in _ev_row(e).items() if v is not None and k != "id"},
                subject_id=e.subject_id)
        return e.id

    def get_event(self, event_id: str) -> Optional[BrainEvent]:
        rows = self._r(f"MATCH (e:{self._ev} {{id: $id}}) RETURN properties(e) AS e", id=event_id)
        return _ev_from(rows[0]["e"]) if rows else None

    def query(self, *, since=None, until=None, subject_id=None, kind=None, zone=None,
              profile=None, limit: int = 100, newest_first: bool = True) -> list[BrainEvent]:
        w, p = self._where(dict(since=since, until=until, subject_id=subject_id, kind=kind,
                                zone=zone, profile=profile))
        order = "DESC" if newest_first else "ASC"
        rows = self._r(f"MATCH (e:{self._ev}){w} RETURN properties(e) AS e "
                       f"ORDER BY e.ts {order}, e.id {order} LIMIT $limit", limit=max(0, int(limit)), **p)
        return [_ev_from(r["e"]) for r in rows]

    def last_event(self, subject_id: str, kind: Optional[str] = None) -> Optional[BrainEvent]:
        r = self.query(subject_id=subject_id, kind=kind, limit=1)
        return r[0] if r else None

    def count(self, **filters) -> int:
        bad = set(filters) - set(_FILTERS)
        if bad:
            raise TypeError(f"unknown filters {sorted(bad)}")
        w, p = self._where(filters)
        return self._r(f"MATCH (e:{self._ev}){w} RETURN count(e) AS n", **p)[0]["n"]

    def _delete(self, where: str, **p) -> list[str]:
        rows = self._w(f"MATCH (e:{self._ev}) WHERE {where} "
                       f"WITH e, e.snapshot_paths AS sp DETACH DELETE e RETURN sp", **p)
        return _paths(r["sp"] for r in rows)

    def purge_before(self, ts: float) -> list[str]:
        return self._delete("e.ts < $ts", ts=float(ts))

    def delete_event(self, event_id: str) -> list[str]:
        """Extra (not in the EventStore protocol): delete one event, return its snapshot paths."""
        return self._delete("e.id = $id", id=event_id)

    def forget_subject(self, subject_id: str) -> list[str]:
        paths = self._delete("e.subject_id = $s", s=subject_id)
        self._w(f"MATCH (r:{self._rt} {{subject_id: $s}}) DETACH DELETE r", s=subject_id)
        return paths

    # -- RoutineStore
    def upsert_routine(self, r: Routine) -> str:
        self._w(f"""MERGE (n:{self._rt} {{id: $id}}) SET n += $props
                    WITH n OPTIONAL MATCH (p:{self._person} {{id: $s}})
                    FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (n)-[:ABOUT]->(p))""",
                id=r.id, props={k: v for k, v in _rt_row(r).items() if k != "id"}, s=r.subject_id)
        return r.id

    def list_routines(self, subject_id=None, profile=None) -> list[Routine]:
        c, p = [], {}
        if subject_id is not None:
            c.append("n.subject_id = $s")
            p["s"] = subject_id
        if profile is not None:
            c.append("$prof IN n.profiles")
            p["prof"] = profile
        w = (" WHERE " + " AND ".join(c)) if c else ""
        rows = self._r(f"MATCH (n:{self._rt}){w} RETURN properties(n) AS n ORDER BY n.id", **p)
        return [_rt_from(r["n"]) for r in rows]

    def delete_routine(self, routine_id: str) -> None:
        self._w(f"MATCH (n:{self._rt} {{id: $id}}) DETACH DELETE n", id=routine_id)


# -------------------------------------------------------------------- SQLite
class SqliteEventStore(_RoutineStatusMixin):
    """Single-file store; thread-safe via one lock. path=":memory:" for tests."""

    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL,
              subject_id TEXT, subject_label TEXT, zone TEXT, camera TEXT, activity TEXT, confidence REAL,
              source TEXT, snapshot_paths TEXT, profiles TEXT, detail TEXT);
            CREATE INDEX IF NOT EXISTS ev_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS ev_subject_ts ON events(subject_id, ts);
            CREATE INDEX IF NOT EXISTS ev_kind_ts ON events(kind, ts);
            CREATE TABLE IF NOT EXISTS routines(id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, kind TEXT,
              description TEXT, params TEXT, support INTEGER, confidence REAL, first_seen REAL,
              last_seen REAL, source TEXT, profiles TEXT);
            CREATE INDEX IF NOT EXISTS rt_subject ON routines(subject_id);""")

    def close(self):
        with self._lock:
            self._db.close()

    @staticmethod
    def _from_row(r) -> BrainEvent:
        d = dict(r)
        d["snapshot_paths"] = json.loads(d["snapshot_paths"] or "[]")
        d["profiles"] = json.loads(d["profiles"] or "[]")
        return _ev_from(d)

    def _where(self, f: dict):
        c, p = [], []
        if f.get("since") is not None:
            c.append("ts >= ?")
            p.append(float(f["since"]))
        if f.get("until") is not None:
            c.append("ts < ?")
            p.append(float(f["until"]))
        for k in ("subject_id", "kind", "zone"):
            if f.get(k) is not None:
                c.append(f"{k} = ?")
                p.append(f[k])
        if f.get("profile") is not None:
            c.append("EXISTS (SELECT 1 FROM json_each(events.profiles) WHERE value = ?)")
            p.append(f["profile"])
        return (" WHERE " + " AND ".join(c)) if c else "", p

    def add_event(self, e: BrainEvent) -> str:
        _check_event(e)
        r = _ev_row(e)
        r["snapshot_paths"] = json.dumps(r["snapshot_paths"])
        r["profiles"] = json.dumps(r["profiles"])
        cols = list(r)
        with self._lock, self._db:
            self._db.execute(f"INSERT OR REPLACE INTO events({','.join(cols)}) "
                             f"VALUES({','.join('?' * len(cols))})", [r[c] for c in cols])
        return e.id

    def get_event(self, event_id):
        with self._lock:
            r = self._db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._from_row(r) if r else None

    def query(self, *, since=None, until=None, subject_id=None, kind=None, zone=None,
              profile=None, limit=100, newest_first=True):
        w, p = self._where(dict(since=since, until=until, subject_id=subject_id, kind=kind,
                                zone=zone, profile=profile))
        o = "DESC" if newest_first else "ASC"
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM events{w} ORDER BY ts {o}, id {o} LIMIT ?",
                                    p + [max(0, int(limit))]).fetchall()
        return [self._from_row(r) for r in rows]

    def last_event(self, subject_id, kind=None):
        r = self.query(subject_id=subject_id, kind=kind, limit=1)
        return r[0] if r else None

    def count(self, **filters):
        bad = set(filters) - set(_FILTERS)
        if bad:
            raise TypeError(f"unknown filters {sorted(bad)}")
        w, p = self._where(filters)
        with self._lock:
            return self._db.execute(f"SELECT count(*) FROM events{w}", p).fetchone()[0]

    def _delete(self, where: str, p) -> list[str]:
        with self._lock, self._db:
            rows = self._db.execute(f"SELECT snapshot_paths FROM events WHERE {where}", p).fetchall()
            self._db.execute(f"DELETE FROM events WHERE {where}", p)
        return _paths(json.loads(r[0] or "[]") for r in rows)

    def purge_before(self, ts):
        return self._delete("ts < ?", [float(ts)])

    def delete_event(self, event_id):
        return self._delete("id = ?", [event_id])

    def forget_subject(self, subject_id):
        paths = self._delete("subject_id = ?", [subject_id])
        with self._lock, self._db:
            self._db.execute("DELETE FROM routines WHERE subject_id=?", (subject_id,))
        return paths

    def upsert_routine(self, r: Routine) -> str:
        d = _rt_row(r)
        d["profiles"] = json.dumps(d["profiles"])
        cols = list(d)
        with self._lock, self._db:
            self._db.execute(f"INSERT OR REPLACE INTO routines({','.join(cols)}) "
                             f"VALUES({','.join('?' * len(cols))})", [d[c] for c in cols])
        return r.id

    def list_routines(self, subject_id=None, profile=None):
        c, p = [], []
        if subject_id is not None:
            c.append("subject_id=?")
            p.append(subject_id)
        w = (" WHERE " + " AND ".join(c)) if c else ""
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM routines{w} ORDER BY id", p).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["profiles"] = json.loads(d["profiles"] or "[]")
            rt = _rt_from(d)
            if profile is None or profile in rt.profiles:
                out.append(rt)
        return out

    def delete_routine(self, routine_id):
        with self._lock, self._db:
            self._db.execute("DELETE FROM routines WHERE id=?", (routine_id,))
