"""Regressions from the first real Telegram conversation (2026-11-25):
a typed answer fell through to chat; chat tools could not run in the claude sandbox;
the agent never asked the owner's name."""
import time

from worldcam.agent.api import MemoryContext
from worldcam.agent.profiles import ProfileSet, load_profiles
from worldcam.agent.uncertainty import OWNER_INTERVIEW, UncertaintyManager
from worldcam.ambiguity.engine import AmbiguityEngine, EngineConfig
from worldcam.ambiguity.worker import FakeWorkerRunner
from worldcam.brain.api import InMemoryEventStore
from worldcam.channels.api import InboundMessage
from worldcam.chat.tools import build_memory_toolset, build_memory_toolset_from_args
from worldcam.cli.enroll import link_owner
from worldcam.graph.api import Doubt, InMemoryGraphStore
from worldcam.users.api import InMemoryUserDirectory


def _asked_world(store, users, owner, *, minutes_ago=1.0, n=1):
    for i in range(n):
        did = store.create_doubt(Doubt(f"d_{i}", "metadata_gap", about_ids=("p_x",), question="q?"))
        store.update_doubt(did, status="asked", message_ref=f"m{i}", detail={"asked_to": owner.id,
                                                                              "topic": "own_facts"})
        # the in-memory store stamps `updated` with the wall clock; make the age explicit
        store.doubts[did] = store.doubts[did].__class__(**{**store.doubts[did].__dict__,
                                                          "updated": time.time() - 60 * minutes_ago * (n - i)})


def _engine(store, users):
    return AmbiguityEngine(store, None, FakeWorkerRunner(lambda m, d: {"action": "resolved", "summary": "x"}),
                           EngineConfig(), dispatcher=None, uncertainty=None)


def test_typed_answer_goes_to_the_question_the_user_was_last_asked():
    store, users = InMemoryGraphStore(), InMemoryUserDirectory()
    owner = users.add("Pat", "owner", "telegram", 1)
    _asked_world(store, users, owner, n=2)                      # d_0 (older), d_1 (newest)
    eng = _engine(store, users)
    typed = InboundMessage("telegram", "1", "my personal room in my owned flat")        # no reply_to_ref
    assert eng.pending_question_for(owner, typed).id == "d_1"
    assert eng.pending_question_for(owner, InboundMessage("telegram", "1", "/status")) is None
    assert eng.pending_question_for(owner, InboundMessage("telegram", "1", "hi", choice_value="chat:more")) is None
    other = users.add("Gus", "guest", "telegram", 2)
    assert eng.pending_question_for(other, typed) is None                              # only who was asked
    old = _engine(store, users)
    old.cfg.unanswered_after_h = 0.0001
    assert old.pending_question_for(owner, typed) is None                              # stale questions expire


def test_context_factory_roundtrip_rebuilds_the_chat_toolset_for_another_process(tmp_path):
    from worldcam.users.store import open_directory
    users_db = tmp_path / "users.db"
    users = open_directory(users_db, bootstrap=False)
    owner = users.add("Pat", "owner", "telegram", "1")
    ctx = MemoryContext(InMemoryGraphStore(), InMemoryEventStore(), InMemoryEventStore(), users, user=owner,
                        extras={"context_factory": "worldcam.chat.context:build_context",
                                "context_args": {"store": f"file:{tmp_path / 'g.json'}",
                                                 "events_db": str(tmp_path / "e.db"), "users_db": str(users_db),
                                                 "snapshot_dir": str(tmp_path)}})
    ts = build_memory_toolset(ctx, writes=False)
    assert ts.factory and ts.factory_args["context_args"]["events_db"].endswith("e.db")
    rebuilt = build_memory_toolset_from_args(ts.factory_args)          # what the MCP child does
    assert set(rebuilt.names()) >= {"recent_events", "last_seen", "who_was_here"}
    bad = dict(ts.factory_args, context_args={"store": {"evil": 1}})
    try:
        build_memory_toolset_from_args(bad)
        assert False
    except ValueError:
        pass


def test_owner_interview_asks_the_name_first_when_the_account_name_is_a_placeholder():
    store, users = InMemoryGraphStore(), InMemoryUserDirectory()
    owner = users.add("Owner", "owner", "telegram", 1, profiles=("companion",))
    pid = link_owner(store, users, owner)
    assert store.get_person(pid).name is None                    # "Owner" is not copied into the graph
    um = UncertaintyManager(store, users, profiles=ProfileSet(load_profiles(["companion"])))
    g = next(g for g in um.knowledge_gaps(1_800_000_000.0) if g.kind == "self_profile")
    assert g.extra["fact_key"] == "preferred_name" == OWNER_INTERVIEW[0][0]
    store.upsert_person(pid, name="Prabhat")                     # once known, never asked again
    keys = [g.extra["fact_key"] for g in um.knowledge_gaps(1_800_000_000.0) if g.kind == "self_profile"]
    assert "preferred_name" not in keys and keys


def test_users_can_be_renamed():
    users = InMemoryUserDirectory()
    u = users.add("Owner", "owner", "telegram", 1)
    assert users.rename(u.id, "  Prabhat ").name == "Prabhat"


def test_chat_factory_works_when_called_the_way_the_mcp_server_calls_it(tmp_path):
    """Regression: mcp_server does factory(**args); the chat factory only accepted one dict."""
    from worldcam.users.store import open_directory
    users = open_directory(tmp_path / "u.db", bootstrap=False)
    owner = users.add("Pat", "owner", "telegram", "1")
    ctx = MemoryContext(InMemoryGraphStore(), InMemoryEventStore(), InMemoryEventStore(), users, user=owner,
                        extras={"context_factory": "worldcam.chat.context:build_context",
                                "context_args": {"store": f"file:{tmp_path / 'g.json'}",
                                                 "events_db": str(tmp_path / "e.db"),
                                                 "users_db": str(tmp_path / "u.db"), "snapshot_dir": str(tmp_path)}})
    ts = build_memory_toolset(ctx, writes=False)
    from worldcam.ambiguity.mcp_server import load_factory
    rebuilt = load_factory(ts.factory)(**ts.factory_args)
    assert "recent_events" in rebuilt.names()
