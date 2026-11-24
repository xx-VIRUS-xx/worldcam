"""Shared GraphStore contract suite, run against every implementation.

Collected via tests/test_graph_contract.py (`from contract_graph import *`).
The Neo4j case is skipped when bolt isn't reachable. It never touches real
data: it uses label prefix TEST_PREFIX (labels WcTestPerson, ...), its own
indexes, wipes only those labels between tests and drops them at the end.
"""
from __future__ import annotations

import socket
from urllib.parse import urlparse

import numpy as np
import pytest

from worldcam.graph.api import Doubt, InMemoryGraphStore, Sighting

TEST_PREFIX = "WcTest"
DIM = 512

# InMemoryGraphStore deviations from the contract, reported to the coordinator
# (api.py is not ours to edit). strict=True: the xfail flips red once fixed.
INMEM_BUG = "InMemoryGraphStore bug (reported, see docs/graph.md)"


def _bolt_reachable(uri: str) -> bool:
    u = urlparse(uri)
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", u.port or 7687), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def neo4j_driver():
    pytest.importorskip("neo4j")
    from neo4j import GraphDatabase

    from worldcam.graph.config import Neo4jConfig
    from worldcam.graph.schema import drop_schema, ensure_schema
    cfg = Neo4jConfig.from_env()
    if not cfg.password or not _bolt_reachable(cfg.uri):
        pytest.skip("Neo4j not reachable (docker compose up -d) or NEO4J_PASSWORD unset")
    drv = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password), connection_timeout=3)
    try:
        drv.verify_connectivity()
    except Exception as e:  # auth / handshake failure
        drv.close()
        pytest.skip(f"Neo4j not usable: {type(e).__name__}")
    ensure_schema(drv, cfg.database, TEST_PREFIX)
    yield drv, cfg.database
    from worldcam.graph.store import Neo4jGraphStore
    Neo4jGraphStore(driver=drv, database=cfg.database, label_prefix=TEST_PREFIX, init_schema=False).wipe()
    drop_schema(drv, cfg.database, TEST_PREFIX)
    drv.close()


@pytest.fixture(params=["memory", "neo4j"])
def store(request):
    if request.param == "memory":
        yield InMemoryGraphStore()
        return
    drv, db = request.getfixturevalue("neo4j_driver")
    from worldcam.graph.store import Neo4jGraphStore
    s = Neo4jGraphStore(driver=drv, database=db, label_prefix=TEST_PREFIX, init_schema=False)
    s.wipe()
    yield s
    s.wipe()


def _is_mem(store) -> bool:
    return isinstance(store, InMemoryGraphStore)


def _xfail_mem(store, request, why=INMEM_BUG):
    if _is_mem(store):
        request.applymarker(pytest.mark.xfail(strict=True, reason=why))


# ------------------------------------------------------------------ vectors
def unit(rng):
    v = rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def near(v, rng, eps=0.2):
    """Unit vector with cosine ~ 1/sqrt(1+eps^2) to v (0.98 for eps=0.2)."""
    w = v + eps * unit(rng)
    return w / np.linalg.norm(w)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ------------------------------------------------------------------- people
def test_upsert_get_and_update_person(store):
    pid = store.upsert_person(name="Prabhat", kind="household", aliases=["P"], ts=100.0)
    p = store.get_person(pid)
    assert (p.id, p.name, p.kind, p.aliases, p.first_seen, p.last_seen) == \
        (pid, "Prabhat", "household", ("P",), 100.0, 100.0)
    assert store.upsert_person(pid, aliases=["P", "Prab"], ts=50.0) == pid
    p = store.get_person(pid)
    assert (p.name, p.kind, p.aliases, p.first_seen, p.last_seen) == \
        ("Prabhat", "household", ("P", "Prab"), 100.0, 100.0)
    store.upsert_person(pid, name="Prabhat K", kind="owner", ts=200.0)
    p = store.get_person(pid)
    assert (p.name, p.kind, p.last_seen) == ("Prabhat K", "owner", 200.0)
    explicit = store.upsert_person("p_explicit", ts=1.0)
    assert explicit == "p_explicit" and store.get_person(explicit).kind == "unknown"
    assert store.get_person("p_missing") is None
    with pytest.raises(ValueError):
        store.upsert_person(name="x", kind="alien")


def test_search_people_by_name_and_alias(store):
    a = store.upsert_person(name="Rahul Sharma", ts=1.0)
    b = store.upsert_person(name="Anita", aliases=["Rahul's sister"], ts=2.0)
    store.upsert_person(name="Zed", ts=3.0)
    store.upsert_person(ts=4.0)  # unnamed
    assert {p.id for p in store.search_people("rahul")} == {a, b}
    assert [p.id for p in store.search_people("  ANITA ")] == [b]
    assert store.search_people("nobody") == []
    assert len(store.search_people("a", limit=1)) == 1


def test_persona_summary(store):
    pid = store.upsert_person(name="X")
    store.set_persona_summary(pid, "Neighbour, walks a dog.")
    p = store.get_person(pid)
    assert p.persona_summary == "Neighbour, walks a dog." and p.persona_updated
    with pytest.raises(KeyError):
        store.set_persona_summary("p_missing", "x")


# -------------------------------------------------------------------- merge
def test_merge_people_moves_everything(store, rng):
    keep = store.upsert_person(name="Rahul", aliases=["R"], ts=50.0)
    drop = store.upsert_person(name="Rahul S", aliases=["RS"], ts=10.0)
    store.upsert_person(drop, ts=500.0)
    other = store.upsert_person(name="Anita", ts=1.0)
    _, cid = store.add_face(unit(rng), quality=0.9, ts=1.0, snapshot_path="f1.jpg")
    store.label_face_cluster(cid, drop)
    store.add_sighting(Sighting("s_m1", 20.0, "door", 1, person_id=drop))
    store.add_fact(drop, "works_at", "Acme", source="owner", confidence=0.9, ts=5.0)
    store.relate(drop, "friend", other, source="owner")
    store.relate(other, "colleague", drop, source="owner")

    store.merge_people(keep, drop)

    assert store.get_person(drop) is None
    k = store.get_person(keep)
    assert k.name == "Rahul"
    assert set(k.aliases) == {"R", "RS", "Rahul S"}
    assert (k.first_seen, k.last_seen) == (10.0, 500.0)
    assert store.match_face(store_centroid_probe(store, cid, rng), k=1)[0].person_id == keep
    assert [s.person_id for s in store.recent_sightings(person_id=keep)] == [keep]
    assert store.recent_sightings(person_id=drop) == []
    assert [(f.person_id, f.key) for f in store.facts(keep)] == [(keep, "works_at")]
    assert store.facts(drop) == []
    assert sorted(store.relations(keep)) == [("colleague", other, "in"), ("friend", other, "out")]
    assert sorted(store.relations(other)) == [("colleague", keep, "out"), ("friend", keep, "in")]
    assert cid in store.person_context(keep)["clusters"]


def store_centroid_probe(store, cid, rng):
    """A vector equal to the cluster's single face is its centroid; recover it via match."""
    if _is_mem(store):
        return store.clusters[cid]["centroid"]
    rows = store._read_rows("MATCH (c:FaceCluster {id: $id}) RETURN c.centroid AS v", id=cid)
    return rows[0]["v"]


def test_merge_people_unknown_raises(store):
    a = store.upsert_person(name="A")
    with pytest.raises(KeyError):
        store.merge_people(a, "p_missing")
    with pytest.raises(KeyError):
        store.merge_people("p_missing", a)
    assert store.get_person(a) is not None


def test_merge_person_into_itself_is_rejected(store, request):
    a = store.upsert_person(name="A")
    with pytest.raises(ValueError):
        store.merge_people(a, a)
    assert store.get_person(a) is not None


def test_merge_dedups_relations_and_drops_self_loops(store, request):
    keep = store.upsert_person(name="K")
    drop = store.upsert_person(name="D")
    other = store.upsert_person(name="O")
    store.relate(keep, "friend", other, source="owner")
    store.relate(drop, "friend", other, source="owner")
    store.relate(keep, "partner", drop, source="owner")
    store.merge_people(keep, drop)
    assert sorted(store.relations(keep)) == [("friend", other, "out")]


def test_merge_redirects_doubts(store, request):
    keep = store.upsert_person(name="K")
    drop = store.upsert_person(name="D")
    did = store.create_doubt(Doubt("", "conflict", about_ids=(keep, drop)))
    store.merge_people(keep, drop)
    assert store.get_doubt(did).about_ids == (keep,)


# ------------------------------------------------------------------- forget
def test_forget_person_hard_deletes_and_returns_paths(store, rng):
    pid = store.upsert_person(name="Visitor", ts=1.0)
    keep = store.upsert_person(name="Owner", kind="owner", ts=1.0)
    v = unit(rng)
    _, cid = store.add_face(v, quality=0.9, ts=1.0, snapshot_path="snap/f1.jpg")
    store.add_face(near(v, rng), quality=0.8, ts=2.0, snapshot_path="snap/f2.jpg", cluster_id=cid)
    store.add_face(near(v, rng), quality=0.8, ts=3.0, snapshot_path=None, cluster_id=cid)
    store.label_face_cluster(cid, pid)
    _, other_cid = store.add_face(unit(rng), quality=0.9, ts=1.0, snapshot_path="snap/keep.jpg")
    store.label_face_cluster(other_cid, keep)
    store.add_sighting(Sighting("s_f1", 10.0, "door", 1, person_id=pid, snapshot_path="snap/s1.jpg"))
    store.add_sighting(Sighting("s_f2", 11.0, "door", 2, cluster_id=cid, snapshot_path="snap/f1.jpg"))
    store.add_sighting(Sighting("s_f3", 12.0, "door", 3, person_id=keep, snapshot_path="snap/s3.jpg"))
    store.add_fact(pid, "name_hint", "V", source="observed", confidence=0.5)
    store.add_fact(keep, "role", "owner", source="owner", confidence=1.0)
    store.relate(pid, "friend", keep, source="owner")
    did = store.create_doubt(Doubt("", "unknown_face", about_ids=(cid,), snapshot_paths=("snap/d1.jpg",)))
    did_p = store.create_doubt(Doubt("", "metadata_gap", about_ids=(pid,)))
    did_keep = store.create_doubt(Doubt("", "metadata_gap", about_ids=(keep,)))

    paths = store.forget_person(pid)

    assert paths == ["snap/d1.jpg", "snap/f1.jpg", "snap/f2.jpg", "snap/s1.jpg"]
    assert store.get_person(pid) is None
    assert store.facts(pid) == []
    assert store.relations(keep) == []
    assert {s.id for s in store.recent_sightings()} == {"s_f3"}
    with pytest.raises(KeyError):
        store.cluster_size(cid)
    assert [m.cluster_id for m in store.match_face(v, k=5)] == [other_cid]
    assert [f.key for f in store.facts(keep)] == ["role"]
    assert store.get_doubt(did) is None and store.get_doubt(did_p) is None  # doubts about them go too
    assert store.get_doubt(did_keep) is not None
    with pytest.raises(KeyError):
        store.forget_person(pid)


# -------------------------------------------------------------------- faces
def test_add_face_clusters_by_similarity(store, rng):
    a, b = unit(rng), unit(rng)
    fa1, ca = store.add_face(a, quality=0.9, ts=1.0, snapshot_path="a1.jpg")
    fa2, ca2 = store.add_face(near(a, rng), quality=0.9, ts=2.0, snapshot_path="a2.jpg")
    _, cb = store.add_face(b, quality=0.9, ts=3.0, snapshot_path=None)
    assert fa1 != fa2 and ca == ca2 and cb != ca
    assert store.cluster_size(ca) == 2 and store.cluster_size(cb) == 1
    with pytest.raises(KeyError):
        store.cluster_size("fc_missing")


def test_add_face_explicit_cluster_and_running_mean_centroid(store, rng):
    a, b = unit(rng), unit(rng)
    _, cid = store.add_face(a, quality=1.0, ts=1.0, snapshot_path=None)
    _, cid2 = store.add_face(b, quality=1.0, ts=2.0, snapshot_path=None, cluster_id=cid)
    assert cid2 == cid and store.cluster_size(cid) == 2
    mean = (a + b) / 2
    top = store.match_face(mean, k=1)[0]
    assert top.cluster_id == cid and top.score == pytest.approx(1.0, abs=1e-4)
    assert store.match_face(a, k=1)[0].score == pytest.approx(float(a @ mean / np.linalg.norm(mean)), abs=1e-4)
    with pytest.raises(KeyError):
        store.add_face(a, quality=1.0, ts=3.0, snapshot_path=None, cluster_id="fc_missing")


def test_match_face_ranks_clusters_and_reports_labels(store, rng):
    vs = [unit(rng) for _ in range(5)]
    cids = [store.add_face(v, quality=0.9, ts=float(i), snapshot_path=None)[1] for i, v in enumerate(vs)]
    assert len(set(cids)) == 5
    pid = store.upsert_person(name="Lab")
    store.label_face_cluster(cids[2], pid)
    q = near(vs[2], rng)
    m = store.match_face(q, k=3)
    assert len(m) == 3
    assert m[0].cluster_id == cids[2] and m[0].person_id == pid and m[0].score > 0.9
    assert m[1].person_id is None and m[1].score < 0.3
    assert [x.score for x in m] == sorted((x.score for x in m), reverse=True)
    assert len(store.match_face(q, k=10)) == 5
    with pytest.raises(KeyError):
        store.label_face_cluster(cids[0], "p_missing")
    with pytest.raises(KeyError):
        store.label_face_cluster("fc_missing", pid)


def test_match_face_empty_store(store, rng):
    assert store.match_face(unit(rng)) == []


# ---------------------------------------------------------------- sightings
def test_sightings_autofill_and_label_backfill(store, rng):
    pid = store.upsert_person(name="Rahul", ts=0.0)
    _, labeled = store.add_face(unit(rng), quality=0.9, ts=1.0, snapshot_path=None)
    store.label_face_cluster(labeled, pid)
    _, unlabeled = store.add_face(unit(rng), quality=0.9, ts=1.0, snapshot_path=None)

    sid = store.add_sighting(Sighting("s_a", 100.0, "door", 7, cluster_id=labeled, zones=("door", "porch"),
                                      activity="walking", snapshot_path="s_a.jpg"))
    assert sid == "s_a"
    s = store.recent_sightings(cluster_id=labeled)[0]
    assert (s.person_id, s.zones, s.activity, s.track_id, s.camera, s.snapshot_path) == \
        (pid, ("door", "porch"), "walking", 7, "door", "s_a.jpg")
    assert store.get_person(pid).last_seen == 100.0

    store.add_sighting(Sighting("s_b", 50.0, "door", 8, cluster_id=unlabeled))
    store.add_sighting(Sighting("s_c", 60.0, "yard", 9, cluster_id=unlabeled))
    assert [s.person_id for s in store.recent_sightings(cluster_id=unlabeled)] == [None, None]
    other = store.upsert_person(name="Other", ts=0.0)
    store.add_sighting(Sighting("s_d", 70.0, "yard", 10, cluster_id=unlabeled, person_id=other))
    store.label_face_cluster(unlabeled, pid)
    got = {s.id: s.person_id for s in store.recent_sightings(cluster_id=unlabeled)}
    assert got == {"s_b": pid, "s_c": pid, "s_d": other}

    assert [s.id for s in store.recent_sightings(person_id=pid)] == ["s_a", "s_c", "s_b"]
    assert [s.id for s in store.recent_sightings(limit=2)] == ["s_a", "s_d"]
    assert [s.id for s in store.recent_sightings(person_id=pid, cluster_id=unlabeled)] == ["s_c", "s_b"]


# ------------------------------------------------------ facts and relations
def test_facts(store):
    pid = store.upsert_person(name="F")
    f2 = store.add_fact(pid, "works_at", "Acme", source="owner", confidence=0.9, ts=20.0)
    f1 = store.add_fact(pid, "relation_to_owner", "friend", source="inferred", confidence=0.6, ts=10.0)
    fs = store.facts(pid)
    assert [f.id for f in fs] == [f1, f2]
    assert (fs[0].key, fs[0].value, fs[0].source, fs[0].confidence, fs[0].ts, fs[0].person_id) == \
        ("relation_to_owner", "friend", "inferred", 0.6, 10.0, pid)
    assert store.add_fact(pid, "x", "y", source="observed", confidence=1).startswith("fact_")
    with pytest.raises(ValueError):
        store.add_fact(pid, "k", "v", source="gossip", confidence=0.5)
    with pytest.raises(KeyError):
        store.add_fact("p_missing", "k", "v", source="owner", confidence=0.5)


def test_relations(store):
    a, b, c = (store.upsert_person(name=n) for n in "ABC")
    store.relate(a, "partner", b, source="owner")
    store.relate(a, "partner", b, source="owner")  # dedup
    store.relate(c, "friend", a, source="inferred")
    assert sorted(store.relations(a)) == [("friend", c, "in"), ("partner", b, "out")]
    assert store.relations(b) == [("partner", a, "in")]
    with pytest.raises(KeyError):
        store.relate(a, "friend", "p_missing", source="owner")


def test_person_context(store, rng):
    pid = store.upsert_person(name="Ctx", ts=1.0)
    other = store.upsert_person(name="O", ts=1.0)
    _, cid = store.add_face(unit(rng), quality=0.9, ts=1.0, snapshot_path=None)
    store.label_face_cluster(cid, pid)
    store.add_fact(pid, "k", "v", source="owner", confidence=1.0, ts=1.0)
    store.relate(pid, "friend", other, source="owner")
    for i in range(12):
        store.add_sighting(Sighting(f"s_ctx{i}", float(i), "door", i, cluster_id=cid))
    ctx = store.person_context(pid)
    assert set(ctx) == {"person", "facts", "relations", "recent_sightings", "clusters"}
    assert ctx["person"].name == "Ctx"
    assert [f.key for f in ctx["facts"]] == ["k"]
    assert ctx["relations"] == [("friend", other, "out")]
    assert [s.id for s in ctx["recent_sightings"]] == [f"s_ctx{i}" for i in range(11, 1, -1)]
    assert ctx["clusters"] == [cid]
    with pytest.raises(KeyError):
        store.person_context("p_missing")


# ------------------------------------------------------------------- doubts
def test_doubt_lifecycle(store):
    pid = store.upsert_person(name="D")
    detail = {"scores": [0.41, 0.38], "nested": {"cluster": "fc_x", "n": 3}, "ok": True}
    did = store.create_doubt(Doubt("", "low_confidence", about_ids=(pid,), snapshot_paths=("c.jpg",),
                                   question="Is this D?", detail=detail))
    d = store.get_doubt(did)
    assert d.id == did and d.status == "open" and d.rounds == 0
    assert d.about_ids == (pid,) and d.snapshot_paths == ("c.jpg",) and d.detail == detail
    assert d.created > 0 and d.updated >= d.created

    d = store.update_doubt(did, status="asked", message_ref="tg:42")
    assert d.status == "asked" and d.message_ref == "tg:42"
    assert store.find_doubt_by_message("tg:42").id == did
    assert store.find_doubt_by_message("tg:43") is None
    d = store.update_doubt(did, status="answered", answer="Yes, that's D")
    d = store.update_doubt(did, status="asked", rounds=d.rounds + 1, last_error=None)
    d = store.update_doubt(did, status="resolved")
    got = store.get_doubt(did)
    assert (got.status, got.answer, got.rounds, got.question, got.detail) == \
        ("resolved", "Yes, that's D", 1, "Is this D?", detail)
    assert got.updated >= got.created

    with pytest.raises(ValueError):
        store.update_doubt(did, status="maybe")
    with pytest.raises(KeyError):
        store.update_doubt("d_missing", status="open")
    with pytest.raises(TypeError):
        store.update_doubt(did, bogus=1)
    with pytest.raises(ValueError):
        store.create_doubt(Doubt("", "vibes"))
    with pytest.raises(ValueError):
        store.create_doubt(Doubt("", "conflict", status="pending"))
    assert store.get_doubt("d_missing") is None


def test_list_doubts(store):
    ids = [store.create_doubt(Doubt("", "unknown_face", created=float(t))) for t in (30, 10, 20)]
    store.update_doubt(ids[0], status="dismissed")
    assert [d.id for d in store.list_doubts()] == [ids[1], ids[2], ids[0]]
    assert [d.id for d in store.list_doubts(status="open")] == [ids[1], ids[2]]
    assert [d.id for d in store.list_doubts(status="dismissed")] == [ids[0]]
    assert len(store.list_doubts(limit=1)) == 1
    explicit = store.create_doubt(Doubt("d_explicit", "metadata_gap", created=5.0))
    assert explicit == "d_explicit" and store.list_doubts()[0].id == "d_explicit"


def test_doubt_detail_nested_json_roundtrip(store):
    detail = {
        "outbox": [{"chat": 1, "text": "Who is this?", "buttons": [["Rahul", "New"], ["Ignore"]],
                    "sent_at": 1795544852.25, "msg": {"id": 77, "ok": True}}],
        "attempts": {"ask": 2, "resolve": 0},
        "history": [{"q": "Who?", "a": "Rahul", "facts": [{"k": "relation", "v": "friend", "c": 0.9}]},
                    {"q": "How do you know him?", "a": None}],
        "unicode": "Prabhat \u00e9\u0939",
        "empty": {"list": [], "dict": {}},
    }
    did = store.create_doubt(Doubt("", "unknown_face", detail=detail))
    assert store.get_doubt(did).detail == detail
    detail2 = {**detail, "attempts": {"ask": 3, "resolve": 1}, "history": detail["history"] + [{"q": "x"}]}
    assert store.update_doubt(did, detail=detail2).detail == detail2
    assert store.get_doubt(did).detail == detail2
    assert store.list_doubts()[0].detail == detail2


def test_list_doubts_large_limit(store):
    ids = [store.create_doubt(Doubt("", "unknown_face", created=float(i + 1))) for i in range(60)]
    assert [d.id for d in store.list_doubts(limit=5000)] == ids
    assert [d.id for d in store.list_doubts(status="open", limit=5000)] == ids


# --------------------------------------------------------------- graph_read
def test_graph_read(store):
    if _is_mem(store):
        with pytest.raises(NotImplementedError):
            store.graph_read("MATCH (p:Person) RETURN p")
        return
    for i in range(5):
        store.upsert_person(name=f"G{i}", ts=float(i))
    rows = store.graph_read("MATCH (p:Person) WHERE p.name STARTS WITH $pre RETURN p.name AS name "
                            "ORDER BY name", {"pre": "G"}, limit=3)
    assert rows == [{"name": "G0"}, {"name": "G1"}, {"name": "G2"}]
    node = store.graph_read("MATCH (p:Person {name: 'G4'}) RETURN p")[0]["p"]
    assert node["name"] == "G4" and node["_labels"] == [TEST_PREFIX + "Person"]
    for bad in ("CREATE (:Person {id: 'x'})", "MATCH (p) DETACH DELETE p",
                "MATCH (p:Person) set p.name = 'x'", "MERGE (p:Person {id: 'x'})",
                "CALL dbms.listConfig()", "MATCH (p) RETURN p; MATCH (q) DELETE q",
                "LOAD CSV FROM 'file:///x' AS r RETURN r"):
        with pytest.raises(ValueError):
            store.graph_read(bad)


def test_graph_read_server_side_read_only(store, monkeypatch):
    """Belt and braces: even if the client check is bypassed, the READ session refuses writes."""
    if _is_mem(store):
        pytest.skip("Neo4j only")
    import worldcam.graph.store as mod
    monkeypatch.setattr(mod, "check_read_only", lambda c: None)
    with pytest.raises(Exception) as ei:
        store.graph_read("CREATE (:Person {id: 'p_sneaky'})")
    assert not isinstance(ei.value, ValueError) or "write" in str(ei.value).lower()
    assert store.get_person("p_sneaky") is None


def test_neo4j_from_env_and_no_arg_constructor(neo4j_driver):
    from worldcam.graph.store import Neo4jGraphStore
    for make in (lambda: Neo4jGraphStore.from_env(init_schema=False),
                 lambda: Neo4jGraphStore(init_schema=False)):
        with make() as s:
            assert s.graph_read("RETURN 1 AS one") == [{"one": 1}]
