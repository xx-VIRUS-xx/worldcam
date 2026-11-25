"""Shared EventStore/RoutineStore contract, run against every implementation:
InMemoryEventStore, SqliteEventStore, Neo4jEventStore (skipped when bolt is down;
uses label prefix WcTest so real data is never touched)."""
from __future__ import annotations

import socket
from dataclasses import replace
from urllib.parse import urlparse

import pytest

from worldcam.brain.api import InMemoryEventStore, Routine, new_event

TEST_PREFIX = "WcTest"


def _bolt_reachable(uri: str) -> bool:
    u = urlparse(uri)
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", u.port or 7687), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def neo4j_brain():
    pytest.importorskip("neo4j")
    from neo4j import GraphDatabase
    from worldcam.graph.config import Neo4jConfig
    cfg = Neo4jConfig.from_env()
    if not cfg.password or not _bolt_reachable(cfg.uri):
        pytest.skip("Neo4j not reachable or NEO4J_PASSWORD unset")
    drv = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password), connection_timeout=3)
    try:
        drv.verify_connectivity()
    except Exception as e:
        drv.close()
        pytest.skip(f"Neo4j not usable: {type(e).__name__}")
    from worldcam.brain.store import Neo4jEventStore
    s = Neo4jEventStore(driver=drv, database=cfg.database, label_prefix=TEST_PREFIX)
    yield s
    s.wipe()
    s.drop_schema()
    drv.close()


@pytest.fixture(params=["memory", "sqlite", "neo4j"])
def store(request):
    if request.param == "memory":
        yield InMemoryEventStore()
    elif request.param == "sqlite":
        from worldcam.brain.store import SqliteEventStore
        s = SqliteEventStore(":memory:")
        yield s
        s.close()
    else:
        s = request.getfixturevalue("neo4j_brain")
        s.wipe()
        yield s
        s.wipe()


def ev(i, ts, kind="seen", **kw):
    return new_event(kind, id=f"e{i}", ts=ts, **kw)


def test_add_get_roundtrip(store):
    e = ev(1, 1000.5, "entered", subject_id="p1", subject_label="Rahul", zone="door", camera="cam2",
           activity="walking", confidence=0.8, source="inferred", snapshot_paths=("/s/a.jpg", "/s/b.jpg"),
           profiles=("cctv", "medical"), detail={"track_id": 7, "nested": {"a": [1, 2]}})
    assert store.add_event(e) == "e1"
    assert store.get_event("e1") == e
    assert store.get_event("nope") is None
    bare = ev(2, 5.0)
    store.add_event(bare)
    assert store.get_event("e2") == bare


def test_bad_kind_or_source_rejected(store):
    with pytest.raises(ValueError):
        store.add_event(ev(1, 1.0, "bogus"))
    with pytest.raises(ValueError):
        store.add_event(ev(2, 1.0, "seen", source="telepathy"))


def _fill(store):
    store.add_event(ev(1, 100, "entered", subject_id="a", zone="door", profiles=("cctv",), snapshot_paths=("/x/1",)))
    store.add_event(ev(2, 200, "seen", subject_id="a", zone="desk", profiles=("medical", "cctv")))
    store.add_event(ev(3, 300, "left", subject_id="b", zone="door", snapshot_paths=("/x/1", "/x/3")))
    store.add_event(ev(4, 400, "seen", subject_id="b", zone="desk"))


def test_query_filters_order_limit(store):
    _fill(store)
    assert [e.id for e in store.query()] == ["e4", "e3", "e2", "e1"]
    assert [e.id for e in store.query(newest_first=False)] == ["e1", "e2", "e3", "e4"]
    assert [e.id for e in store.query(since=200, until=400)] == ["e3", "e2"]      # since incl, until excl
    assert [e.id for e in store.query(subject_id="a")] == ["e2", "e1"]
    assert [e.id for e in store.query(kind="seen")] == ["e4", "e2"]
    assert [e.id for e in store.query(zone="door", newest_first=False)] == ["e1", "e3"]
    assert [e.id for e in store.query(profile="medical")] == ["e2"]
    assert [e.id for e in store.query(profile="cctv", newest_first=False)] == ["e1", "e2"]
    assert [e.id for e in store.query(limit=2)] == ["e4", "e3"]
    assert store.query(subject_id="zzz") == []
    assert store.query(subject_id="a", kind="seen", zone="desk", since=0, until=1000)[0].id == "e2"


def test_last_event_and_count(store):
    _fill(store)
    assert store.last_event("a").id == "e2"
    assert store.last_event("a", "entered").id == "e1"
    assert store.last_event("a", "left") is None
    assert store.last_event("nobody") is None
    assert store.count() == 4
    assert store.count(kind="seen") == 2
    assert store.count(subject_id="b", since=350) == 1
    assert store.count(profile="cctv") == 2


def test_purge_before_returns_paths(store):
    _fill(store)
    assert store.purge_before(300) == ["/x/1"]        # e1, e2 deleted; boundary event e3 kept
    assert store.count() == 2
    assert store.get_event("e1") is None
    assert store.purge_before(300) == []
    assert store.purge_before(10_000) == ["/x/1", "/x/3"]
    assert store.count() == 0


def test_forget_subject_returns_paths(store):
    _fill(store)
    assert store.forget_subject("b") == ["/x/1", "/x/3"]
    assert store.count() == 2 and store.count(subject_id="b") == 0
    assert store.forget_subject("b") == []
    assert store.forget_subject("a") == ["/x/1"]


def _rt(i, sid="a", source="inferred", profiles=()):
    return Routine(id=f"r{i}", subject_id=sid, kind="daily_at", description="takes meds ~09:00",
                   params={"event_kind": "activity_changed", "activity": "taking_medication",
                           "minute_of_day": 540, "tolerance_min": 45},
                   support=5, confidence=0.8, first_seen=1.0, last_seen=2.0, source=source, profiles=profiles)


def test_routine_crud(store):
    r = _rt(1, profiles=("medical",))
    assert store.upsert_routine(r) == "r1"
    store.upsert_routine(_rt(2, sid="b", source="user"))
    assert {x.id for x in store.list_routines()} == {"r1", "r2"}
    assert store.list_routines("a") == [r]
    assert [x.id for x in store.list_routines(profile="medical")] == ["r1"]
    store.upsert_routine(replace(r, support=9))
    assert store.list_routines("a")[0].support == 9 and len(store.list_routines()) == 2
    store.delete_routine("r1")
    store.delete_routine("r1")
    assert [x.id for x in store.list_routines()] == ["r2"]
