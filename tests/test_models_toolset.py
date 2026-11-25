import asyncio
import json

import pytest

from worldcam.ambiguity.file_store import FileGraphStore
from worldcam.ambiguity.mcp_server import (build_server_from_toolset, build_toolset_from_ref, load_factory)
from worldcam.ambiguity.tools import TOOL_NAMES, WorldcamTools, build_toolset
from worldcam.ambiguity.worker import FakeWorkerRunner, make_worker_runner, load_schema
from worldcam.graph.api import Doubt
from worldcam.models.api import FakeToolRunner, NotConfigured


@pytest.fixture
def env(tmp_path):
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    path = f"file:{tmp_path}/g.pkl"
    store = FileGraphStore(str(tmp_path / "g.pkl"))
    did = store.create_doubt(Doubt(id="d_1", kind="unknown_face", about_ids=("fc_1",)))
    return store, path, str(snaps), did


def test_toolset_specs_match_tools_and_flags(env):
    store, path, snaps, did = env
    ts = WorldcamTools(store, snaps, did).toolset()
    assert ts.names() == TOOL_NAMES and ts.factory is None
    by = {s.name: s for s in ts.specs}
    assert not by["graph_read"].writes and not by["get_doubt"].writes
    assert by["add_fact"].writes and by["ask_owner"].writes
    assert "delete_person" not in by and "forget_person" not in by
    assert by["add_fact"].input_schema["required"] == ["person_id", "key", "value", "source", "confidence"]
    assert by["get_doubt"].fn({"doubt_id": did})["kind"] == "unknown_face"
    with pytest.raises(ValueError, match="only act on doubt"):
        by["get_doubt"].fn({"doubt_id": "d_other"})
    with pytest.raises(ValueError, match="bad arguments"):
        by["get_doubt"].fn({"nope": 1})


def test_factory_round_trip_keeps_scope_and_security(env):
    store, path, snaps, did = env
    ts = build_toolset(path, snaps, did)
    assert ts.factory == "worldcam.ambiguity.tools:build_toolset"
    json.dumps(ts.factory_args)                           # JSON-able
    rebuilt = build_toolset_from_ref(ts.factory, ts.factory_args)   # what the child process does
    by = {s.name: s for s in rebuilt.specs}
    assert by["get_doubt"].fn({"doubt_id": did})["id"] == did
    with pytest.raises(ValueError):
        by["upsert_person"].fn({"name": "X", "kind": "owner"})       # worker can't create an owner
    with pytest.raises(ValueError, match="read-only"):
        by["graph_read"].fn({"cypher": "MATCH (n) DETACH DELETE n"})
    with pytest.raises(ValueError, match="outside snapshot"):
        by["ask_owner"].fn({"doubt_id": did, "text": "q?", "image_paths": ["/etc/passwd"]})


def test_factory_ref_is_restricted():
    for bad in ("os:system", "subprocess:run", "worldcam", "worldcamx.y:z"):
        with pytest.raises(ValueError):
            load_factory(bad)


def test_generic_mcp_server_serves_exact_schemas_and_calls(env):
    store, path, snaps, did = env
    ts = build_toolset(path, snaps, did)
    srv = build_server_from_toolset(ts)
    tools = {t.name: t for t in asyncio.run(srv.list_tools())}
    assert tuple(tools) == TOOL_NAMES
    ann = tools["add_fact"].annotations
    assert ann.read_only_hint is False and tools["get_doubt"].annotations.read_only_hint is True
    assert tools["add_fact"].input_schema["required"] == ["person_id", "key", "value", "source", "confidence"]
    assert "confidence" in tools["add_fact"].input_schema["properties"]
    res = asyncio.run(srv.call_tool("get_doubt", {"doubt_id": did}))
    assert did in json.dumps(res.content[0].text if hasattr(res, "content") else res, default=str)
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError, match="only act on doubt"):
        asyncio.run(srv.call_tool("get_doubt", {"doubt_id": "d_other"}))


def test_custom_toolset_through_generic_server():
    from worldcam.models.api import ToolSet, ToolSpec
    ts = ToolSet((ToolSpec("echo", "echo", {"type": "object", "properties": {"x": {"type": "string"},
                                                                              "n": {"type": "integer"}},
                                            "required": ["x"]}, lambda a: {"got": a}),))
    srv = build_server_from_toolset(ts)
    res = asyncio.run(srv.call_tool("echo", {"x": "hi"}))
    assert "'got'" in str(res) or '"got"' in str(res)
    assert "'n'" not in str(res)                           # omitted optional arg is not passed


def test_make_worker_runner_is_explicit_config(env, monkeypatch):
    store, path, snaps, did = env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from worldcam.models.router import ConfiguredModelRouter
    # gemini for resolve but no key -> clear error at startup, no silent switch to claude
    r = ConfiguredModelRouter(config={"resolve": "gemini:gemini-2.5-flash"}, env={},
                              backend_kwargs={"claude": {"spawn": lambda *a: (0, "", "")}})
    with pytest.raises(NotConfigured, match="GEMINI_API_KEY"):
        make_worker_runner(path, snaps, router=r)
    with pytest.raises(NotConfigured, match="GEMINI_API_KEY"):
        make_worker_runner(path, snaps, worker="gemini:gemini-2.5-flash")
    ok = ConfiguredModelRouter(config={}, env={}, backend_kwargs={"claude": {"spawn": lambda *a: (0, "", "")}})
    w = make_worker_runner(path, snaps, router=ok, worker_model="haiku")
    assert w._claude["ask"].model == "haiku" and w._claude["resolve"].model == "haiku"


def test_routed_runner_generic_path_with_fake_tool_runner(env):
    store, path, snaps, did = env
    from worldcam.ambiguity.worker import RoutedWorkerRunner
    from worldcam.models.api import FakeModelRouter

    def script(req, call_tool):
        assert req.output_schema == load_schema() and req.task == "resolve"
        assert call_tool("get_doubt", {"doubt_id": did})["id"] == did
        return {"action": "dismissed", "summary": "nothing to do"}

    class R(FakeModelRouter):
        def describe(self):
            return {"question": "fake:x", "resolve": "fake:x"}
    w = RoutedWorkerRunner(path, snaps, R(runner=FakeToolRunner(script)))
    res = w.run("resolve", store.get_doubt(did))
    assert res.ok and res.action == "dismissed", res.error
