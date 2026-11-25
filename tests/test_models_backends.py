import json

import pytest

from worldcam.models.api import AgentRunRequest, GenRequest, NotConfigured, ToolSet, ToolSpec
from worldcam.models.claude_cli import ClaudeCliClient, ClaudeCliRunner, parse_result
from worldcam.models.claude_api import ClaudeApiClient
from worldcam.models.gemini import FINAL_TOOL, GeminiClient, GeminiToolRunner, require_key
from worldcam.models.jev import JevClient
from worldcam.models.util import ModelError, check_schema, get_secret

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20
SCHEMA = {"type": "object", "properties": {"colour": {"type": "string"}}, "required": ["colour"],
          "additionalProperties": False}


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "crop.png"
    p.write_bytes(PNG)
    return str(p)


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


# ---- claude CLI ---------------------------------------------------------------
def result_json(**kw):
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, **kw})


def test_claude_oneshot_command_has_no_tools_and_prompt_not_in_argv():
    c = ClaudeCliClient("haiku", claude_bin="/opt/claude")
    cmd, stdin_text = c.build_command(GenRequest("decide", "PRIVATE", system="be brief", json_schema=SCHEMA))
    assert cmd[:2] == ["/opt/claude", "-p"] and flag(cmd, "--tools") == ""
    assert flag(cmd, "--output-format") == "json" and "--input-format" not in cmd
    assert json.loads(flag(cmd, "--json-schema")) == SCHEMA
    assert flag(cmd, "--model") == "haiku" and flag(cmd, "--append-system-prompt") == "be brief"
    assert flag(cmd, "--setting-sources") == "" and "--no-session-persistence" in cmd
    assert "PRIVATE" not in " ".join(cmd) and stdin_text == "PRIVATE"
    assert "--mcp-config" not in cmd and "--dangerously-skip-permissions" not in cmd


def test_claude_images_use_stream_json_blocks(img):
    cmd, stdin_text = ClaudeCliClient().build_command(GenRequest("vision_crop", "what colour?", images=(img,)))
    assert flag(cmd, "--input-format") == "stream-json" and flag(cmd, "--output-format") == "stream-json"
    msg = json.loads(stdin_text)["message"]["content"]
    assert msg[0]["type"] == "image" and msg[0]["source"]["media_type"] == "image/png"
    assert msg[1] == {"type": "text", "text": "what colour?"}
    with pytest.raises(ModelError):
        ClaudeCliClient().build_command(GenRequest("x", "y", images=("/nope.png",)))


def test_claude_generate_structured_and_stream_parse():
    out = result_json(result="", structured_output={"colour": "red"}, usage={"input_tokens": 9, "output_tokens": 3},
                      total_cost_usd=0.002)
    spawn = lambda *a: (0, "\n".join([json.dumps({"type": "system"}), out]), "")   # noqa: E731 (stream-json)
    r = ClaudeCliClient("haiku", spawn=spawn).generate(GenRequest("decide", "x", json_schema=SCHEMA))
    assert r.data == {"colour": "red"} and r.usage["cost_usd"] == 0.002 and r.model == "claude:haiku"
    bad = lambda *a: (0, result_json(result="", structured_output={"colour": 5}), "")   # noqa: E731
    with pytest.raises(ModelError, match="violates"):
        ClaudeCliClient(spawn=bad).generate(GenRequest("decide", "x", json_schema=SCHEMA))
    with pytest.raises(ModelError, match="exit 2"):
        ClaudeCliClient(spawn=lambda *a: (2, "", "boom")).generate(GenRequest("x", "y"))
    with pytest.raises(ModelError):
        parse_result(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True}))


def test_claude_runner_command_mcp_config_and_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    seen = {}

    def spawn(cmd, stdin_text, env, cwd, timeout):
        seen.update(cmd=cmd, stdin=stdin_text, env=env)
        cfg = json.load(open(flag(cmd, "--mcp-config")))
        seen["cfg"] = cfg
        return 0, result_json(result="", structured_output={"colour": "red"}, num_turns=3), ""

    ts = ToolSet((ToolSpec("t", "d", {"type": "object"}, lambda a: 1),),
                 factory="worldcam.ambiguity.tools:build_toolset", factory_args={"store": "file:/x", "doubt_id": "d_1"})
    r = ClaudeCliRunner("sonnet", claude_bin="/opt/claude", python_bin="/py", src_dir="/src", spawn=spawn).run(
        AgentRunRequest("resolve", "PRIVATE", toolset=ts, output_schema=SCHEMA, max_turns=7))
    assert r.ok and r.data == {"colour": "red"} and r.tool_calls == 2
    cmd = seen["cmd"]
    assert flag(cmd, "--tools") == "" and "--strict-mcp-config" in cmd
    assert flag(cmd, "--allowedTools") == "mcp__worldcam__*" and flag(cmd, "--permission-mode") == "dontAsk"
    assert flag(cmd, "--max-turns") == "7"
    srv = seen["cfg"]["mcpServers"]["worldcam"]
    assert srv["args"] == ["-m", "worldcam.ambiguity.mcp_server"]
    assert srv["env"]["WORLDCAM_TOOLSET_FACTORY"] == "worldcam.ambiguity.tools:build_toolset"
    assert json.loads(srv["env"]["WORLDCAM_TOOLSET_ARGS"])["doubt_id"] == "d_1"
    assert "TELEGRAM_BOT_TOKEN" not in seen["env"] and "GEMINI_API_KEY" not in seen["env"]
    assert "PRIVATE" not in " ".join(cmd) and "tok" not in " ".join(cmd) + json.dumps(seen["cfg"])


def test_claude_runner_needs_factory_for_tools():
    ts = ToolSet((ToolSpec("t", "d", {"type": "object"}, lambda a: 1),))
    r = ClaudeCliRunner(spawn=lambda *a: (0, "", "")).run(AgentRunRequest("agent", "x", toolset=ts))
    assert not r.ok and "factory" in r.error


# ---- gemini -------------------------------------------------------------------
def test_gemini_requires_key_and_never_puts_it_in_url(monkeypatch):
    with pytest.raises(NotConfigured, match="GEMINI_API_KEY"):
        require_key({})
    assert require_key({"GEMINI_API_KEY": "k"}) == "k"
    seen = {}

    def tr(url, headers, body, timeout):
        seen.update(url=url, headers=headers)
        raise ModelError("HTTP 400 echo SECRETKEY")

    with pytest.raises(ModelError) as e:
        GeminiClient("gemini-2.5-flash", "SECRETKEY", transport=tr).generate(GenRequest("x", "y"))
    assert "SECRETKEY" not in str(e.value) and "SECRETKEY" not in seen["url"]
    assert seen["headers"] == {"x-goog-api-key": "SECRETKEY"}


def test_get_secret_reads_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\nGEMINI_API_KEY='abc'   # trailing\nX=1\n")
    assert get_secret("GEMINI_API_KEY", {}, [f]) == "abc"
    assert get_secret("GEMINI_API_KEY", {"GEMINI_API_KEY": "env"}, [f]) == "env"
    assert get_secret("GEMINI_API_KEY", {}) is None      # explicit env never reads real files


def test_gemini_payload_images_system_schema(img):
    c = GeminiClient("gemini-2.5-flash", "k")
    body = c.build_payload(GenRequest("vision_crop", "describe", system="sys", images=(img,),
                                      json_schema=SCHEMA, max_tokens=300))
    parts = body["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "image/png" and parts[1] == {"text": "describe"}
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    g = body["generationConfig"]
    assert g["responseMimeType"] == "application/json" and g["responseJsonSchema"] == SCHEMA and g["maxOutputTokens"] == 300
    assert "tools" not in body


def test_gemini_generate_parses_json_and_validates():
    ok = {"candidates": [{"content": {"parts": [{"text": "```json\n{\"colour\": \"red\"}\n```"}]}}],
          "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}}
    r = GeminiClient("m", "k", transport=lambda *a: ok).generate(GenRequest("x", "y", json_schema=SCHEMA))
    assert r.data == {"colour": "red"} and r.usage == {"input_tokens": 7, "output_tokens": 2}
    bad = {"candidates": [{"content": {"parts": [{"text": "{\"colour\": 3}"}]}}]}
    with pytest.raises(ModelError, match="violates"):
        GeminiClient("m", "k", transport=lambda *a: bad).generate(GenRequest("x", "y", json_schema=SCHEMA))
    with pytest.raises(ModelError, match="no candidates"):
        GeminiClient("m", "k", transport=lambda *a: {"promptFeedback": {"blockReason": "SAFETY"}}).generate(
            GenRequest("x", "y"))


def test_gemini_tool_loop_declarations_dispatch_and_final_answer():
    got = []
    spec = ToolSpec("add", "adds", {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                    lambda a: got.append(a) or {"sum": a["a"] + 1}, writes=True)
    boom = ToolSpec("boom", "fails", {"type": "object"}, lambda a: (_ for _ in ()).throw(ValueError("nope")))
    replies = iter([
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "add", "args": {"a": 2}}, "thoughtSignature": "sig"},
                                               {"functionCall": {"name": "boom", "args": {}}}]}}],
         "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1}},
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": FINAL_TOOL, "args": {"colour": "red"}}}]}}],
         "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 2}},
    ])
    bodies = []

    def tr(url, headers, body, timeout):
        bodies.append(json.loads(json.dumps(body)))
        return next(replies)

    r = GeminiToolRunner("gemini-2.5-flash", "k", transport=tr).run(
        AgentRunRequest("agent", "go", toolset=ToolSet((spec, boom)), output_schema=SCHEMA))
    assert r.ok and r.data == {"colour": "red"} and r.tool_calls == 2 and r.usage["input_tokens"] == 30
    decl = bodies[0]["tools"][0]["functionDeclarations"]
    assert [d["name"] for d in decl] == ["add", "boom", FINAL_TOOL]
    assert decl[0]["parametersJsonSchema"]["required"] == ["a"] and decl[2]["parametersJsonSchema"] == SCHEMA
    assert got == [{"a": 2}]
    last = bodies[1]["contents"]
    assert last[1]["parts"][0]["thoughtSignature"] == "sig"                       # model turn echoed
    fr = [p["functionResponse"] for p in last[2]["parts"]]
    assert fr[0]["response"] == {"sum": 3} and fr[1]["response"] == {"error": "nope"}


def test_gemini_runner_rejects_schema_violation_and_unknown_tool_and_turn_limit():
    def call(name, args):
        return {"candidates": [{"content": {"parts": [{"functionCall": {"name": name, "args": args}}]}}]}
    replies = iter([call("ghost", {}), call(FINAL_TOOL, {"colour": 1}), call(FINAL_TOOL, {"colour": "ok"})])
    r = GeminiToolRunner("m", "k", transport=lambda *a: next(replies)).run(
        AgentRunRequest("agent", "go", output_schema=SCHEMA))
    assert r.ok and r.data == {"colour": "ok"} and r.tool_calls == 1
    loop = GeminiToolRunner("m", "k", transport=lambda *a: call("ghost", {})).run(
        AgentRunRequest("agent", "go", max_turns=2))
    assert not r.error and not loop.ok and "turns" in loop.error


# ---- jev / claude-api ------------------------------------------------------------
def test_jev_payload_and_answers():
    schema = {"type": "object", "properties": {
        "urgent": {"x-jev": {"type": "noul", "instructions": "Urgent?"}},
        "kind": {"x-jev": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}}}}
    seen = {}

    def tr(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=body)
        return {"model": "typesafe/jev-1.14", "usage": {"input_tokens": 4, "output_tokens": 1},
                "answers": {"urgent": {"type": "noul", "noul": 0.9}, "kind": {"type": "choice", "choice": "a"}}}

    r = JevClient("default", "KEY", transport=tr).generate(GenRequest("decide", "state text", json_schema=schema))
    assert r.data == {"urgent": 0.9, "kind": "a"}
    assert seen["body"]["state"] == "state text" and seen["body"]["questions"]["urgent"]["type"] == "noul"
    assert seen["headers"] == {"Authorization": "Bearer KEY"} and seen["url"].endswith("/v1/decisions")
    with pytest.raises(ModelError, match="x-jev"):
        JevClient("default", "k", transport=tr).generate(
            GenRequest("decide", "s", json_schema={"type": "object", "properties": {"q": {"type": "string"}}}))


def test_claude_api_payload(img):
    seen = {}

    def tr(url, headers, body, timeout):
        seen.update(headers=headers, body=body)
        return {"content": [{"type": "tool_use", "name": "respond", "input": {"colour": "red"}}],
                "usage": {"input_tokens": 3, "output_tokens": 1}}

    r = ClaudeApiClient("claude-haiku-5-5", "K", transport=tr).generate(
        GenRequest("vision_crop", "what", images=(img,), json_schema=SCHEMA, system="s"))
    assert r.data == {"colour": "red"} and seen["headers"]["x-api-key"] == "K"
    b = seen["body"]
    assert b["messages"][0]["content"][0]["source"]["type"] == "base64" and b["tool_choice"]["name"] == "respond"
    with pytest.raises(NotConfigured):
        ClaudeApiClient("default", "K")


def test_check_schema_subset():
    check_schema({"colour": "x"}, SCHEMA)
    for bad in ({}, {"colour": "x", "z": 1}, {"colour": 1}, []):
        with pytest.raises(ValueError):
            check_schema(bad, SCHEMA)
