"""The v2 contracts' fakes behave; every parallel component builds against these."""
import pytest

from worldcam.agent.api import MemoryContext, Profile, Trigger
from worldcam.brain.api import InMemoryEventStore, new_event
from worldcam.channels.api import FakeChannel, OutboundMessage
from worldcam.graph.api import InMemoryGraphStore
from worldcam.models.api import (AgentRunRequest, FakeModelClient, FakeToolRunner, GenRequest,
                                 ToolSet, ToolSpec)
from worldcam.users.api import InMemoryUserDirectory, can, can_confirm


def test_roles_and_confirmation_rules():
    d = InMemoryUserDirectory()
    owner = d.add("Pat", "owner", "telegram", 1)
    mom = d.add("Mom", "household", "telegram", 2)
    nurse = d.add("Nia", "caregiver", "telegram", 3, profiles=["medical"])
    guest = d.add("Gus", "guest", "telegram", 4)
    assert can_confirm(owner, "identity") and can_confirm(mom, "identity")
    assert not can_confirm(guest, "identity") and not can_confirm(nurse, "identity")
    assert can_confirm(nurse, "medical") and not can_confirm(mom, "medical")
    assert can_confirm(guest, "own_facts", about_self=True) and not can_confirm(guest, "own_facts")
    assert d.by_channel("telegram", 9) is None                     # not whitelisted
    assert [u.name for u in d.with_permission("receive_medical_alerts", "medical")] == ["Nia"]
    d.revoke(mom.id, "chat")
    assert not can(d.get(mom.id), "chat")
    d.deactivate(guest.id)
    assert d.by_channel("telegram", 4) is None


def test_event_store_query_and_retention():
    s = InMemoryEventStore()
    s.add_event(new_event("entered", ts=100.0, subject_id="p1", snapshot_paths=("a.jpg",), profiles=("cctv",)))
    s.add_event(new_event("left", ts=200.0, subject_id="p1"))
    s.add_event(new_event("seen", ts=300.0, subject_id="p2"))
    assert [e.kind for e in s.query(subject_id="p1")] == ["left", "entered"]
    assert s.last_event("p1", "entered").ts == 100.0 and s.count(profile="cctv") == 1
    assert s.purge_before(150.0) == ["a.jpg"] and s.count() == 2
    with pytest.raises(ValueError):
        s.add_event(new_event("bogus"))


def test_fake_tool_runner_dispatches_tools():
    seen = []
    ts = ToolSet((ToolSpec("ping", "p", {"type": "object"}, lambda a: seen.append(a) or {"pong": 1}),))
    r = FakeToolRunner(lambda req, call: {"answer": call("ping", {"x": 1})["pong"]})
    res = r.run(AgentRunRequest("chat", "hi", toolset=ts))
    assert res.ok and res.data == {"answer": 1} and res.tool_calls == 1 and seen == [{"x": 1}]
    bad = FakeToolRunner(lambda req, call: call("nope")).run(AgentRunRequest("chat", "hi", toolset=ts))
    assert not bad.ok and "unknown tool" in bad.error


def test_fake_model_client_and_channel_and_context():
    c = FakeModelClient(lambda r: {"activity": "sitting"})
    assert c.generate(GenRequest("vision_scene", "x", json_schema={})).data == {"activity": "sitting"}
    ch = FakeChannel()
    ch.push("hello", sender_id="7")
    assert ch.poll()[0].text == "hello" and ch.poll() == []
    ref = ch.send("7", OutboundMessage("hi"))
    assert ch.sent[0][0] == ref
    ctx = MemoryContext(InMemoryGraphStore(), InMemoryEventStore(), InMemoryEventStore(), InMemoryUserDirectory())
    assert ctx.clock() > 0
    assert Trigger("event", 1.0).priority == 1 and Profile("cctv", "d").enabled
