"""Scripted day with a fake clock: events -> alerts, curiosity -> the right user -> reply -> settled."""
import random
import time
from types import SimpleNamespace

import pytest

from worldcam.agent.api import MemoryContext, Trigger
from worldcam.agent.budget import InMemoryAttentionBudget, budget_from_users
from worldcam.agent.dispatcher import Dispatcher
from worldcam.agent.loop import AgentConfig, AutonomousAgent
from worldcam.agent.profiles import ProfileSet, load_builtin
from worldcam.agent.uncertainty import UncertaintyManager
from worldcam.ambiguity.engine import AmbiguityEngine, EngineConfig
from worldcam.ambiguity.tools import WorldcamTools
from worldcam.ambiguity.worker import FakeWorkerRunner
from worldcam.brain.api import InMemoryEventStore, Routine, RoutineStatus, new_event
from worldcam.channels.api import FakeChannel
from worldcam.graph.api import InMemoryGraphStore, Sighting
from worldcam.users.api import InMemoryUserDirectory


def local(h, day=10, m=0):
    return time.mktime((2026, 11, day, h, m, 0, 0, 0, -1))


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class World:
    def __init__(self, tmp_path, profiles=("cctv", "companion", "medical"), start=None, **cfg):
        self.clock = Clock(start or local(12))
        self.users = InMemoryUserDirectory()
        self.owner = self.users.add("Olga", "owner", "fake", "1")
        self.house = self.users.add("Hana", "household", "fake", "2", profiles=("companion", "cctv"))
        self.care = self.users.add("Carl", "caregiver", "fake", "3", profiles=("medical",))
        self.guest = self.users.add("Gus", "guest", "fake", "4")
        self.ch = FakeChannel()
        self.budget = budget_from_users(self.users)
        self.disp = Dispatcher(self.users, [self.ch], self.budget, clock=self.clock)
        self.graph = InMemoryGraphStore()
        self.events = InMemoryEventStore()
        self.snaps = tmp_path / "snaps"
        self.snaps.mkdir(parents=True)
        self.ctx = MemoryContext(self.graph, self.events, self.events, self.users, clock=self.clock,
                                 look_now=lambda cam: str(self.snaps / f"{cam}.jpg"))
        self.ps = ProfileSet([load_builtin(n) for n in profiles])
        self.unc = UncertaintyManager(self.graph, self.users, budget=self.budget, profiles=self.ps,
                                      routines=self.events, clock=self.clock)
        self.engine = AmbiguityEngine(self.graph, None, FakeWorkerRunner(self.script), EngineConfig(
            snapshot_dir=str(self.snaps)), dispatcher=self.disp, uncertainty=self.unc)
        self.agent = AutonomousAgent(self.ctx, self.disp, self.ps, uncertainty=self.unc, clock=self.clock,
                                     rng=random.Random(1), engine_tick=lambda: None,
                                     config=AgentConfig(**cfg))

    def script(self, mode, d):
        t = WorldcamTools(self.graph, str(self.snaps), d.id)
        if mode == "ask":
            t.ask_owner(d.id, d.question or "Who is this?", [{"label": "Ignore", "value": "ignore"}])
            return {"action": "asked", "summary": "asked"}
        pid = d.about_ids[0]
        t.add_fact(pid, "relation_to_owner", "friend", "owner", 0.9)
        t.resolve_doubt(d.id, "ok")
        return {"action": "resolved", "summary": "ok", "person_ids": [pid], "facts_written": 1}

    def stranger(self, n=3, t0=None):
        p = SimpleNamespace(id=self.graph.upsert_person(kind="unknown"))
        t0 = t0 or self.clock.t - 3600
        for i in range(n):
            self.graph.add_sighting(Sighting(f"s{p.id}{i}", t0 + i, "cam", i, person_id=p.id))
        return p

    def inbox_for(self, user):
        return [(ref, m) for ref, rid, m in self.ch.sent if rid == user.channel_ids["fake"]]


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


def test_curiosity_asks_the_right_user_and_reply_settles(w):
    p = w.stranger()
    d_id = w.agent.run_task("curiosity")
    assert d_id and w.graph.get_doubt(d_id).detail["origin"] == "curiosity"
    rep = w.engine.tick(w.clock.t)
    assert rep.sent == [d_id]
    # Hana (household, subscribed to companion) is asked, not the owner
    (ref, msg), = w.inbox_for(w.house)
    assert not w.inbox_for(w.owner) and not w.inbox_for(w.care)
    d = w.graph.get_doubt(d_id)
    assert d.status == "asked" and d.detail["asked_to"] == w.house.id and d.detail["asked_role"] == "household"
    assert w.engine.handle_user_reply(w.house, w.ch.push("my friend Rahul", sender_id="2", reply_to_ref=ref),
                                      w.clock.t + 60) == d_id
    d = w.graph.get_doubt(d_id)
    assert d.status == "answered" and d.detail["confirmed_by"]["user_id"] == w.house.id
    assert d.detail["confirmed_by"]["role"] == "household"
    w.engine.tick(w.clock.t + 120)
    assert w.graph.get_doubt(d_id).status == "resolved"
    assert w.graph.facts(p.id)[0].source == "owner"
    # a settled gap is not asked again
    assert w.agent.run_task("curiosity") is None


def test_curiosity_one_question_at_a_time_and_nothing_to_ask(w):
    assert w.agent.run_task("curiosity") is None
    w.stranger(), w.stranger(), w.stranger()
    w.agent.run_task("curiosity")
    w.agent.run_task("curiosity")
    assert w.agent.run_task("curiosity") is None            # max_open_curiosity=2
    assert len(w.graph.list_doubts("open")) == 2


def test_wrong_role_answer_does_not_settle(w):
    w.stranger()
    d_id = w.agent.run_task("curiosity")
    w.engine.tick(w.clock.t)
    ref = w.inbox_for(w.house)[0][0]
    # the guest got hold of the question (forwarded): wrong person and wrong role
    d = w.graph.get_doubt(d_id)
    w.graph.update_doubt(d_id, detail=dict(d.detail, asked_to=w.guest.id))
    w.engine.handle_user_reply(w.guest, w.ch.push("that's the mailman", sender_id="4", reply_to_ref=ref), w.clock.t + 5)
    d = w.graph.get_doubt(d_id)
    assert d.status == "dismissed" and d.answer is None and "confirmed_by" not in d.detail
    review = w.graph.get_doubt(d.detail["review_doubt"])
    assert review.kind == "conflict" and review.status == "open" and "mailman" in review.question
    assert review.detail["asked_exclude"] == [w.guest.id]
    pid = d.about_ids[0]
    fact = [f for f in w.graph.facts(pid) if f.key.startswith("unconfirmed")][0]
    assert fact.source == "inferred" and fact.confidence <= 0.6
    # the review goes to someone who can confirm, never to the guest
    w.engine.tick(w.clock.t + 10)
    assert not w.inbox_for(w.guest)
    assert w.graph.get_doubt(review.id).detail["asked_to"] in (w.house.id, w.owner.id)


def test_reply_from_someone_not_asked_is_ignored(w):
    w.stranger()
    d_id = w.agent.run_task("curiosity")
    w.engine.tick(w.clock.t)
    ref = w.inbox_for(w.house)[0][0]
    assert w.engine.handle_user_reply(w.owner, w.ch.push("x", sender_id="1", reply_to_ref=ref)) is None
    assert w.graph.get_doubt(d_id).status == "asked"


def test_medical_gap_goes_to_caregiver_and_falls_back_to_owner(w):
    p = SimpleNamespace(id=w.graph.upsert_person(name="Mum", kind="household"))
    w.graph.add_fact(p.id, "medication_morning", "metformin 500", source="owner", confidence=0.9, ts=w.clock.t - 400 * 86400)
    d_id = w.agent.run_task("curiosity")
    d = w.graph.get_doubt(d_id)
    assert d.detail["topic"] == "medical" and d.detail["priority"] == 2
    w.engine.tick(w.clock.t)
    assert w.inbox_for(w.care) and not w.inbox_for(w.house)
    # no caregiver available (deactivated): the owner is the fallback
    w.users.deactivate(w.care.id)
    w.graph.update_doubt(d_id, status="open", message_ref=None, detail=dict(
        w.graph.get_doubt(d_id).detail, asked_to=None))
    assert w.unc.ask_target(w.graph.get_doubt(d_id), w.clock.t).id == w.owner.id


def test_own_facts_goes_to_the_person_themselves(w):
    me = w.users.add("Mia", "household", "fake", "9", person_id="p_mia")
    w.graph.upsert_person("p_mia", name="Mia", kind="household")
    r = Routine("r1", "p_mia", "daily_at", "Mia goes for a run around 07:00", support=5, confidence=0.5)
    w.events.upsert_routine(r)
    d_id = w.agent.run_task("curiosity")
    w.engine.tick(w.clock.t)
    assert w.inbox_for(me) and not w.inbox_for(w.owner)


def test_quiet_hours_hold_the_question_until_morning(tmp_path):
    w = World(tmp_path, start=local(22, m=50))
    w.stranger()
    w.agent.run_task("curiosity")
    assert w.engine.tick(local(23, m=30)).sent == []
    assert w.engine.tick(local(2, day=11)).sent == []
    assert w.engine.tick(local(8, day=11, m=5)).sent != []


def test_ignore_tap_backs_off_exponentially(w):
    for _ in range(4):
        w.stranger()
    d1 = w.agent.run_task("curiosity")
    w.engine.tick(w.clock.t)
    ref = w.inbox_for(w.house)[0][0]
    w.engine.handle_user_reply(w.house, w.ch.push(sender_id="2", reply_to_ref=ref, choice_value="ignore"), w.clock.t)
    assert w.graph.get_doubt(d1).status == "dismissed"
    assert w.budget.remaining(w.house.id, w.clock.t)["ignore_streak"] == 1
    w.agent.run_task("curiosity")
    w.engine.tick(w.clock.t + 600)                       # Hana is cooling down: the owner gets it
    assert len(w.inbox_for(w.house)) == 1 and len(w.inbox_for(w.owner)) == 1


def test_unanswered_question_counts_as_ignored(w):
    w.stranger()
    d_id = w.agent.run_task("curiosity")
    w.engine.tick(w.clock.t)
    w.engine.tick(w.clock.t + 13 * 3600)
    assert w.budget.remaining(w.house.id, w.clock.t + 13 * 3600)["ignore_streak"] == 1
    assert w.graph.get_doubt(d_id).status in ("open", "asked")


def test_priority3_alert_bypasses_quiet_hours(tmp_path):
    w = World(tmp_path, start=local(2))
    ev = new_event("seen", ts=local(2), subject_label="unknown person", zone="front_door", confidence=0.8,
                   snapshot_paths=("/x/a.jpg",), profiles=("cctv",))
    w.events.add_event(ev)
    w.agent.submit(Trigger("event", w.clock.t, {"event_id": ev.id}))
    assert w.agent.step() == 1
    msgs = [(rid, m) for _, rid, m in w.ch.sent]
    # household has cctv alert permission, owner too; both get it despite quiet hours
    assert {rid for rid, _ in msgs} == {"1", "2"}
    owner_msg = next(m for rid, m in msgs if rid == "1")
    house_msg = next(m for rid, m in msgs if rid == "2")
    assert "front_door" in owner_msg.text and owner_msg.images == ("/x/a.jpg",)
    assert house_msg.images == ("/x/a.jpg",)               # household may view images
    assert [e for e in w.events.query(kind="alert")]
    # the same event is not alerted twice
    w.agent.submit(Trigger("event", w.clock.t, {"event_id": ev.id}))
    w.agent.step()
    assert len(w.ch.sent) == 2


def test_alert_images_stripped_for_users_without_view_images(tmp_path):
    w = World(tmp_path, start=local(2))
    w.users.revoke(w.house.id, "view_images")
    ev = new_event("seen", ts=local(2), subject_label="unknown", zone="porch", confidence=0.8,
                   snapshot_paths=("/x/a.jpg",))
    w.events.add_event(ev)
    w.agent.step(w.clock.t)
    w.agent.run_task("alert_review")
    house_msg = next(m for _, rid, m in w.ch.sent if rid == "2")
    assert house_msg.images == ()


def test_alert_review_finds_events_and_respects_hard_ceiling(tmp_path):
    w = World(tmp_path, start=local(3))
    for i in range(14):                               # guest-level hard ceiling is irrelevant; use household
        w.budget.record(w.house.id, "alert", w.clock.t - 60 - i)      # household day cap 12, hard 24
    ev = new_event("seen", ts=local(3), subject_label="unknown", zone="front_door", confidence=0.9)
    w.events.add_event(ev)
    fired = w.agent.run_task("alert_review")
    assert fired == [("cctv:unknown_person_door_night", ev.id)]
    assert {rid for _, rid, _ in w.ch.sent} == {"1", "2"}                # priority 3 got through the day cap
    for i in range(12):
        w.budget.record(w.house.id, "alert", w.clock.t - 30 - i)
    assert w.house.id and not w.budget.allow(w.house.id, priority=3, now=w.clock.t)   # 27 >= hard 24


def test_medical_routine_overdue_alert_goes_to_caregivers_only(tmp_path):
    w = World(tmp_path, start=local(11))
    r = Routine("r1", "p1", "daily_at", "takes medication around 09:00", support=9, confidence=0.9)
    w.events.status = lambda now, subject_id=None: [RoutineStatus(r, "overdue", expected_at=local(9))]
    assert w.agent.run_task("alert_review") == [("medical:routine_overdue", "r1")]
    assert {rid for _, rid, _ in w.ch.sent} == {"1", "3"}        # caregiver + owner (never household)
    assert "late" in w.ch.sent[0][2].text
    assert w.agent.run_task("alert_review") == []                      # once a day per routine


def test_no_movement_alert_needs_prior_activity(tmp_path):
    w = World(tmp_path, start=local(15))
    assert w.agent.run_task("alert_review") == []                      # system saw nothing today: stay quiet
    w.events.add_event(new_event("seen", ts=local(9), subject_id="p1", zone="kitchen"))
    assert w.agent.run_task("alert_review") == [("medical:no_movement_daytime", "nomove")]


def test_checkins_per_profile_and_skip_recent_chat(tmp_path):
    w = World(tmp_path, start=local(9, m=30))
    mia = w.users.add("Mia", "caregiver", "fake", "8", profiles=("medical",), person_id="p_mia")
    w.agent.run_task("checkin")
    got = {rid: m for _, rid, m in w.ch.sent}
    assert "8" in got and got["8"].silent                  # medical morning check-in to the person themselves
    # companion midday window is 10-20h: not yet
    assert all("Anything you would like" not in m.text for m in got.values())
    w.ch.sent.clear()
    w.clock.t = local(13)
    w.agent._last_chat[mia.id] = w.clock.t - 60
    w.agent.run_task("checkin")
    texts = {rid: m.text for _, rid, m in w.ch.sent}
    assert "8" not in texts and any("Anything you would like" in t for t in texts.values())


def test_consolidate_proposes_merge_never_merges(w):
    a = SimpleNamespace(id=w.graph.upsert_person(name="Rahul", kind="visitor"))
    b = SimpleNamespace(id=w.graph.upsert_person(name="rahul", kind="visitor"))
    called = []
    w.agent.miner = lambda now: called.append("mine")
    w.agent.summarizer = lambda now: called.append("sum")
    made = w.agent.run_task("consolidate")
    assert called == ["mine", "sum"] and len(made) == 1
    d = w.graph.get_doubt(made[0])
    assert d.kind == "conflict" and set(d.about_ids) == {a.id, b.id} and d.detail["proposal"] == "merge"
    assert w.graph.get_person(a.id) and w.graph.get_person(b.id)       # both still exist
    assert w.agent.run_task("consolidate") == []                       # proposed once


def test_look_around_hands_snapshot_to_callback(w):
    seen = []
    w.agent.describe_scene = lambda cam, path, **kw: seen.append((cam, path, kw["cloud_ok"]))
    w.agent.cfg.cameras = ("door", "hall")
    assert len(w.agent.run_task("look_around")) == 2
    assert seen[0][0] == "door" and seen[0][2] is False
    w.agent.ctx.look_now = None
    assert w.agent.run_task("look_around") == []


def test_queue_overflow_drops_and_counts(tmp_path):
    w = World(tmp_path, queue_size=3)
    ok = [w.agent.submit(Trigger("chat", w.clock.t, {"user_id": "x"})) for _ in range(5)]
    assert ok == [True, True, True, False, False] and w.agent.stats["dropped"] == 2
    assert w.agent.submit(Trigger("bogus", 0)) is False


def test_task_crash_is_isolated(w):
    def boom(*a, **k):
        raise RuntimeError("camera exploded")
    w.agent.describe_scene = boom
    w.agent.submit(Trigger("schedule", w.clock.t, {"task": "look_around"}))
    w.stranger()
    w.agent.submit(Trigger("schedule", w.clock.t, {"task": "curiosity"}))
    assert w.agent.step() == 2
    assert w.agent.stats["errors"] == {"look_around": 1}
    assert w.agent.last_results["curiosity"]                     # the next task still ran
    assert w.agent.errors[0][0] == "look_around"


def test_chat_trigger_prefers_that_user(w):
    w.stranger()
    w.agent.submit(Trigger("chat", w.clock.t, {"user_id": w.owner.id}))
    w.agent.step()
    d = w.graph.list_doubts("open")[0]
    assert d.detail["prefer_user"] == w.owner.id
    w.engine.tick(w.clock.t)
    assert w.inbox_for(w.owner) and not w.inbox_for(w.house)


def test_scheduler_is_jittered_and_deterministic(tmp_path):
    n = [0]
    def schedule(seed):
        n[0] += 1
        w = World(tmp_path / str(n[0]), start=local(12))
        w.agent.rng = random.Random(seed)
        out = []
        for i in range(0, 3 * 3600, 60):
            w.agent._schedule_due(local(12) + i)
            while True:
                try:
                    t = w.agent._q.get_nowait()
                except Exception:
                    break
                out.append((i, t.kind, t.payload.get("task")))
        return out
    a, b = schedule(7), schedule(7)
    assert a == b and a
    assert {x[2] for x in a} >= {"alert_review", "curiosity"}
    assert any(x[1] == "random" for x in a)
    assert schedule(8) != a


def test_thread_start_stop_clean(w):
    w.agent.cfg.poll_s = 0.01
    w.agent.start()
    w.agent.submit(Trigger("schedule", w.clock.t, {"task": "consolidate"}))
    for _ in range(200):
        if w.agent.stats["runs"].get("consolidate"):
            break
        time.sleep(0.01)
    w.agent.stop()
    assert w.agent.stats["runs"]["consolidate"] == 1
    assert w.agent._thread is None
