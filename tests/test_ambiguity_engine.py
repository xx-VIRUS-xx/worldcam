import time

import pytest

from worldcam.ambiguity.engine import AmbiguityEngine, EngineConfig
from worldcam.ambiguity.tools import WorldcamTools
from worldcam.ambiguity.worker import FakeWorkerRunner
from worldcam.graph.api import Doubt, InMemoryGraphStore
from worldcam.messenger.base import FakeNotifier, OwnerReply


def local(h, day=10):
    return time.mktime((2026, 11, day, h, 0, 0, 0, 0, -1))


NOON = local(12)


@pytest.fixture
def snaps(tmp_path):
    d = tmp_path / "snaps"
    d.mkdir()
    (d / "c.jpg").write_bytes(b"x")
    return d


def mk(store, snaps, n=1, created=NOON):
    ids = []
    for i in range(n):
        ids.append(store.create_doubt(Doubt(id=f"d_{i}", kind="unknown_face", about_ids=(f"fc_{i}",),
                                            snapshot_paths=(str(snaps / "c.jpg"),), created=created)))
        store.clusters[f"fc_{i}"] = {"person_id": None, "centroid": [1.0], "n": 3}
    return ids


def scripted(store, snaps, resolve_name="Rahul", ask_ok=True):
    """A fake worker that behaves like the real one, through the MCP tool layer."""
    def script(mode, d):
        t = WorldcamTools(store, str(snaps), d.id)
        if mode == "ask":
            if not ask_ok:
                return {"action": "asked", "summary": "lied: never called ask_owner"}
            t.ask_owner(d.id, "Who is this?", ["New person", {"label": "Ignore", "value": "ignore"}])
            return {"action": "asked", "summary": "asked"}
        pid = t.upsert_person(name=resolve_name, kind="visitor")["person_id"]
        t.label_face_cluster(d.about_ids[0], pid)
        t.add_fact(pid, "relation_to_owner", "colleague", "owner", 0.9)
        t.resolve_doubt(d.id, "labeled")
        return {"action": "resolved", "summary": "labeled", "person_ids": [pid], "facts_written": 1}
    return script


def engine(store, snaps, script=None, **cfg):
    n = FakeNotifier()
    w = FakeWorkerRunner(script or scripted(store, snaps))
    return AmbiguityEngine(store, n, w, EngineConfig(snapshot_dir=str(snaps), **cfg)), n, w


def test_full_cycle_ask_reply_resolve(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)
    eng, n, w = engine(store, snaps)
    rep = eng.tick(NOON)
    assert rep.asked_runs == [did] and rep.sent == [did]
    ref, q = n.sent[0]
    assert q.text == "Who is this?" and q.image_paths and q.choices[-1].value == "ignore"
    d = store.get_doubt(did)
    assert d.status == "asked" and d.message_ref == ref and "outbox" not in d.detail
    assert eng.handle_reply(OwnerReply(ref, text="That's Rahul, my colleague"), NOON + 60) == did
    assert store.get_doubt(did).status == "answered"
    rep = eng.tick(NOON + 61)
    assert rep.resolve_runs == [did] and w.calls == [("ask", did), ("resolve", did)]
    d = store.get_doubt(did)
    assert d.status == "resolved" and d.last_error is None
    p = store.search_people("Rahul")[0]
    assert p.kind == "visitor" and store.clusters["fc_0"]["person_id"] == p.id
    assert store.facts(p.id)[0].source == "owner"


def test_ignore_choice_dismisses_without_worker(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)
    eng, n, w = engine(store, snaps)
    eng.tick(NOON)
    eng.handle_reply(OwnerReply(n.sent[0][0], choice_value="ignore"), NOON + 5)
    d = store.get_doubt(did)
    assert d.status == "dismissed" and d.detail["ignored_by_owner"]
    eng.tick(NOON + 10)
    assert w.calls == [("ask", did)]


def test_unknown_reply_ref_is_ignored(snaps):
    store = InMemoryGraphStore()
    mk(store, snaps)
    eng, *_ = engine(store, snaps)
    assert eng.handle_reply(OwnerReply("nope", text="hi")) is None


def test_rate_limit_per_hour(snaps):
    store = InMemoryGraphStore()
    mk(store, snaps, n=5)
    eng, n, _ = engine(store, snaps, max_questions_per_hour=2, max_runs_per_tick=10)
    assert len(eng.tick(NOON).sent) == 2
    assert eng.tick(NOON + 600).sent == []
    assert len(store.list_doubts("open")) == 3
    assert len(eng.tick(NOON + 3601).sent) == 2      # window slides


def test_quiet_hours_queue_and_late_outbox(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps, created=local(22))
    eng, n, w = engine(store, snaps)
    for h in (23, 2, 7):
        rep = eng.tick(local(h, day=11 if h < 12 else 10))
        assert rep.quiet and not rep.asked_runs and not n.sent
    assert store.get_doubt(did).status == "open"
    assert eng.tick(local(8, day=11)).sent == [did]
    # a follow-up queued during quiet hours waits in the outbox
    store.update_doubt(did, status="answered", answer="hm")
    WorldcamTools(store, str(snaps), did).ask_owner(did, "Name?")
    assert eng.tick(local(23, day=11)).sent == []
    assert eng.tick(local(9, day=12)).sent == [did]


def test_expiry_after_seven_days(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps, created=NOON)
    eng, *_ = engine(store, snaps, max_questions_per_hour=0)
    assert eng.tick(NOON + 6 * 86400).expired == []
    assert eng.tick(NOON + 7 * 86400 + 1).expired == [did]
    d = store.get_doubt(did)
    assert d.status == "dismissed" and d.detail["dismissed_reason"] == "expired"


def test_max_rounds_enforced(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)

    def always_followup(mode, d):
        t = WorldcamTools(store, str(snaps), d.id)
        t.ask_owner(d.id, f"round {d.rounds + 1}?")
        return {"action": "asked" if mode == "ask" else "followup", "summary": "q"}
    eng, n, w = engine(store, snaps, script=always_followup, retry_backoff_s=0)
    eng.tick(NOON)
    for i in range(2):
        eng.handle_reply(OwnerReply(n.sent[-1][0], text="dunno"), NOON + i)
        eng.tick(NOON + i)
    assert store.get_doubt(did).rounds == 3
    eng.handle_reply(OwnerReply(n.sent[-1][0], text="dunno"), NOON + 10)
    for i in range(3):   # 4th ask refused by the tool -> failed runs -> dismissed
        eng.tick(NOON + 20 + i)
    d = store.get_doubt(did)
    assert d.status == "dismissed" and d.detail["dismissed_reason"] == "worker failed" and len(n.sent) == 3


def test_failed_run_reverts_with_last_error_and_backoff(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)
    eng, n, w = engine(store, snaps, script=scripted(store, snaps, ask_ok=False))
    rep = eng.tick(NOON)
    d = store.get_doubt(did)
    assert rep.failed == [did] and d.status == "open" and "doubt is open" in d.last_error
    assert d.detail["attempts"] == 1
    assert eng.tick(NOON + 10).asked_runs == []          # backoff
    assert eng.tick(NOON + 301).asked_runs == [did]


def test_crashing_runner_reverts(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)

    def boom(mode, d):
        raise RuntimeError("kaput")
    eng, *_ = engine(store, snaps, script=boom)
    eng.tick(NOON)
    d = store.get_doubt(did)
    assert d.status == "open" and "kaput" in d.last_error


def test_engine_is_restartable(snaps):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)
    eng1, n, _ = engine(store, snaps)
    eng1.tick(NOON)
    ref = n.sent[0][0]
    eng2 = AmbiguityEngine(store, n, FakeWorkerRunner(scripted(store, snaps)),
                           EngineConfig(snapshot_dir=str(snaps)))   # "restart"
    eng2.handle_reply(OwnerReply(ref, text="Rahul"), NOON + 1)
    eng2.tick(NOON + 2)
    assert store.get_doubt(did).status == "resolved"
    assert eng2.sent_last_hour(NOON + 2) == 1


def test_engine_drops_images_outside_snapshot_dir(snaps, tmp_path):
    store = InMemoryGraphStore()
    (did,) = mk(store, snaps)
    store.update_doubt(did, status="asked", detail={"outbox": {"text": "q", "choices": [],
                                                               "image_paths": ["/etc/hosts"]}})
    eng, n, _ = engine(store, snaps)
    eng.tick(NOON)
    assert n.sent[0][1].image_paths == ()


def test_poll_once_survives_poller_error(snaps):
    store = InMemoryGraphStore()
    mk(store, snaps)
    eng, n, _ = engine(store, snaps)

    def boom(timeout_s=0):
        raise RuntimeError("409 conflict")
    n.poll_replies = boom
    assert eng.poll_once(0, NOON).sent == ["d_0"]
