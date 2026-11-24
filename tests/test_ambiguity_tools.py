import asyncio
import os

import pytest

from worldcam.ambiguity.file_store import FileGraphStore, open_store
from worldcam.ambiguity.tools import TOOL_NAMES, ToolInputError, WorldcamTools, confine_image
from worldcam.graph.api import Doubt, InMemoryGraphStore


@pytest.fixture
def env(tmp_path):
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    crop = snaps / "crop.jpg"
    crop.write_bytes(b"\xff\xd8fake")
    store = InMemoryGraphStore()
    did = store.create_doubt(Doubt(id="d_1", kind="unknown_face", about_ids=("fc_1",),
                                   snapshot_paths=(str(crop),)))
    store.clusters["fc_1"] = {"person_id": None, "centroid": [1.0], "n": 3}
    tools = WorldcamTools(store, str(snaps), did, clock=lambda: 100.0)
    return store, tools, snaps, crop, tmp_path


def test_mcp_server_exposes_exactly_the_design_tools(env):
    from worldcam.ambiguity.mcp_server import build_server
    store, tools, *_ = env
    srv = build_server(tools)
    names = sorted(t.name for t in asyncio.run(srv.list_tools()))
    assert names == sorted(TOOL_NAMES)
    assert len(TOOL_NAMES) == 13 and "forget_person" not in names


def test_mcp_server_maps_validation_errors(env):
    from mcp.server.mcpserver.exceptions import ToolError
    from worldcam.ambiguity.mcp_server import build_server
    store, tools, *_ = env
    srv = build_server(tools)
    with pytest.raises(ToolError, match="only act on doubt"):
        asyncio.run(srv.call_tool("get_doubt", {"doubt_id": "d_other"}))


def test_doubt_scope_and_id_validation(env):
    store, tools, *_ = env
    assert tools.get_doubt("d_1")["kind"] == "unknown_face"
    for bad in ("d_other", "../x", "", 5, "a" * 65):
        with pytest.raises(ToolInputError):
            tools.get_doubt(bad)


def test_upsert_label_fact_flow(env):
    store, tools, *_ = env
    with pytest.raises(ToolInputError):
        tools.upsert_person()                       # new person needs a name
    with pytest.raises(ToolInputError):
        tools.upsert_person(name="X", kind="owner")  # worker can't create owners
    with pytest.raises(ToolInputError):
        tools.upsert_person(name="bad\x00name")
    pid = tools.upsert_person(name="Rahul", kind="visitor")["person_id"]
    tools.label_face_cluster("fc_1", pid)
    assert store.clusters["fc_1"]["person_id"] == pid
    with pytest.raises(ToolInputError):
        tools.label_face_cluster("fc_other", pid)  # cluster not in this doubt
    with pytest.raises(ToolInputError, match="owner answer"):
        tools.add_fact(pid, "relation_to_owner", "colleague", "owner", 0.9)
    store.update_doubt("d_1", answer="colleague")
    tools.add_fact(pid, "relation_to_owner", "colleague", "owner", 0.9)
    for args in [(pid, "Bad Key", "v", "owner", 0.9), (pid, "k", "v", "guess", 0.9),
                 (pid, "k", "v", "owner", 1.5), (pid, "k", "v", "owner", True), (pid, "k", "", "owner", 0.5)]:
        with pytest.raises(ToolInputError):
            tools.add_fact(*args)
    assert [f.source for f in store.facts(pid)] == ["owner"]


def test_out_of_scope_person_is_protected(env):
    store, tools, *_ = env
    other = store.upsert_person(name="Bystander", kind="household")
    with pytest.raises(ToolInputError, match="not part of doubt"):
        tools.upsert_person(person_id=other, name="Hacked")
    with pytest.raises(ToolInputError):
        tools.add_fact(other, "note", "x", "inferred", 0.1)
    with pytest.raises(ToolInputError):
        tools.set_persona_summary(other, "x")
    pid = tools.upsert_person(name="Rahul")["person_id"]
    with pytest.raises(ToolInputError, match="conflict"):
        tools.merge_people(pid, other)
    tools.relate(pid, "colleague", other, "owner")  # one endpoint in scope is enough
    assert store.get_person(other).name == "Bystander"


def test_merge_only_for_conflict(tmp_path):
    store = InMemoryGraphStore()
    a, b = store.upsert_person(name="A"), store.upsert_person(name="B")
    store.create_doubt(Doubt(id="d_c", kind="conflict", about_ids=(a, b)))
    t = WorldcamTools(store, str(tmp_path), "d_c")
    t.merge_people(a, b)
    assert store.get_person(b) is None


def test_graph_read_validation(env):
    store, tools, *_ = env
    with pytest.raises(ToolInputError, match="read-only"):
        tools.graph_read("MATCH (p) DETACH DELETE p")
    with pytest.raises(ToolInputError, match="unavailable"):
        tools.graph_read("MATCH (p:Person) RETURN p.name")


def test_ask_owner_outbox_and_rounds(env):
    store, tools, snaps, crop, _ = env
    r = tools.ask_owner("d_1", "Who is this?", ["New person", {"label": "Ignore", "value": "ignore"}])
    d = store.get_doubt("d_1")
    assert r["status"] == "asked" and d.status == "asked" and d.rounds == 1 and d.message_ref is None
    assert d.detail["outbox"]["image_paths"] == [os.path.realpath(crop)]   # default: doubt's crop
    assert d.detail["outbox"]["choices"][0] == {"label": "New person", "value": "New person"}
    with pytest.raises(ToolInputError, match="status asked"):
        tools.ask_owner("d_1", "again?")
    store.update_doubt("d_1", status="answered", answer="Rahul", rounds=3)
    with pytest.raises(ToolInputError, match="max 3 rounds"):
        tools.ask_owner("d_1", "more?")


def test_ask_owner_choice_validation(env):
    store, tools, *_ = env
    with pytest.raises(ToolInputError):
        tools.ask_owner("d_1", "q", [{"label": "x", "value": "é" * 40}])  # > 64 bytes
    with pytest.raises(ToolInputError):
        tools.ask_owner("d_1", "q", [str(i) for i in range(9)])
    with pytest.raises(ToolInputError):
        tools.ask_owner("d_1", "")


def test_image_path_confinement(env, tmp_path):
    store, tools, snaps, crop, root = env
    outside = root / "secret.jpg"
    outside.write_bytes(b"x")
    (snaps / "link.jpg").symlink_to(outside)
    (snaps / "notes.txt").write_text("x")
    for bad in [str(outside), str(snaps / ".." / "secret.jpg"), str(snaps / "link.jpg"),
                str(snaps / "notes.txt"), str(snaps / "missing.jpg"), "/etc/hosts", "", "a\x00b.jpg"]:
        with pytest.raises(ToolInputError):
            confine_image(bad, str(snaps))
        with pytest.raises(ToolInputError):
            tools.ask_owner("d_1", "q", image_paths=[bad])
    assert confine_image("crop.jpg", str(snaps)) == os.path.realpath(crop)
    with pytest.raises(ToolInputError):
        tools.ask_owner("d_1", "q", image_paths=[str(crop)] * 3)
    assert store.get_doubt("d_1").status == "open"


def test_resolve_and_dismiss_rules(env):
    store, tools, *_ = env
    with pytest.raises(ToolInputError):
        tools.resolve_doubt("d_1", "done")          # not answered yet
    store.update_doubt("d_1", status="answered", answer="x")
    tools.resolve_doubt("d_1", "done")
    assert store.get_doubt("d_1").status == "resolved"
    with pytest.raises(ToolInputError):
        tools.dismiss_doubt("d_1", "meh")


def test_file_store_roundtrip(tmp_path):
    spec = f"file:{tmp_path}/g.pkl"
    a = open_store(spec)
    pid = a.upsert_person(name="Rahul", kind="visitor")
    b = open_store(spec)                      # second "process"
    assert b.get_person(pid).name == "Rahul"
    assert isinstance(b, FileGraphStore)
    with pytest.raises(ValueError):
        open_store("mysql://x")
