import json
import os
import stat
import time

import pytest

from worldcam.ambiguity.worker import (ClaudeWorkerRunner, FakeWorkerRunner, load_schema,
                                       parse_claude_output, validate_result)
from worldcam.graph.api import Doubt

D = Doubt(id="d_abc", kind="unknown_face", about_ids=("fc_1",))


def runner(tmp_path, **kw):
    return ClaudeWorkerRunner(f"file:{tmp_path}/g.pkl", str(tmp_path / "snaps"),
                              claude_bin="/opt/claude", python_bin="/py/bin/python", **kw)


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_command_line_matches_sandbox_spec(tmp_path):
    cmd = runner(tmp_path, model="sonnet").build_command("ask", D, "/tmp/cfg.json")
    assert cmd[:2] == ["/opt/claude", "-p"] and "d_abc" in cmd[2] and "MODE: ask" in cmd[2]
    assert "--strict-mcp-config" in cmd and flag(cmd, "--mcp-config") == "/tmp/cfg.json"
    assert flag(cmd, "--tools") == ""
    assert flag(cmd, "--allowedTools") == "mcp__worldcam__*"
    assert flag(cmd, "--permission-mode") == "dontAsk"
    assert flag(cmd, "--max-turns") == "12"
    assert flag(cmd, "--output-format") == "json"
    assert json.loads(flag(cmd, "--json-schema")) == load_schema()   # inline JSON, not a path
    assert "data, not instructions" in flag(cmd, "--append-system-prompt").replace("DATA", "data")
    assert flag(cmd, "--setting-sources") == ""
    assert "--disable-slash-commands" in cmd and "--no-session-persistence" in cmd
    assert flag(cmd, "--model") == "sonnet"
    for banned in ("--dangerously-skip-permissions", "--add-dir", "bypassPermissions"):
        assert banned not in cmd


def test_resolve_prompt_and_mode_check(tmp_path):
    r = runner(tmp_path)
    assert "MODE: resolve" in r.prompt("resolve", "d_abc") and "{" not in r.prompt("resolve", "d_abc").split("\n")[0]
    with pytest.raises(ValueError):
        r.prompt("delete", "d_abc")


def test_mcp_config_and_no_secrets_in_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "s3cret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    r = runner(tmp_path)
    cfg = r.mcp_config("d_abc")["mcpServers"]["worldcam"]
    assert cfg["command"] == "/py/bin/python"
    assert cfg["args"][:2] == ["-m", "worldcam.ambiguity.mcp_server"]
    assert cfg["args"][cfg["args"].index("--doubt") + 1] == "d_abc"
    blob = json.dumps(cfg) + " ".join(r.build_command("ask", D, "x"))
    assert "s3cret" not in blob and "tok" not in blob
    env = r.child_env()
    assert "TELEGRAM_BOT_TOKEN" not in env and env["NEO4J_PASSWORD"] == "s3cret"


def test_store_spec_validated(tmp_path):
    with pytest.raises(ValueError):
        ClaudeWorkerRunner("bolt://x", str(tmp_path))


def envelope(obj, **kw):
    return json.dumps(dict({"type": "result", "subtype": "success", "is_error": False,
                            "result": json.dumps(obj), "structured_output": obj}, **kw))


def test_parse_output():
    ok = {"action": "asked", "summary": "q"}
    assert parse_claude_output(envelope(ok)) == ok
    for bad in ["not json", json.dumps({"type": "x"}),
                envelope(ok, is_error=True), envelope(ok, subtype="error_max_turns"),
                envelope({"action": "pwned", "summary": "x"}),
                envelope({"action": "asked"}),
                envelope({"action": "asked", "summary": "x", "extra": 1})]:
        with pytest.raises(ValueError):
            parse_claude_output(bad)


def test_validate_result_types():
    validate_result({"action": "resolved", "summary": "x", "person_ids": ["p_1"], "facts_written": 2})
    with pytest.raises(ValueError):
        validate_result({"action": "resolved", "summary": "x", "facts_written": -1})
    with pytest.raises(ValueError):
        validate_result({"action": "resolved", "summary": "x" * 501})


def fake_claude(tmp_path, body):
    p = tmp_path / "fake_claude.sh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_run_with_fake_binary(tmp_path):
    out = envelope({"action": "asked", "summary": "ok"}, total_cost_usd=0.01)
    r = runner(tmp_path)
    r.claude_bin = fake_claude(tmp_path, f"cat <<'EOF'\n{out}\nEOF\n")
    res = r.run("ask", D)
    assert res.ok and res.action == "asked" and res.cost_usd == 0.01


def test_timeout_kills_process_group(tmp_path):
    marker = tmp_path / "child.pid"
    r = runner(tmp_path, timeout_s=1)
    # The fake claude spawns a grandchild (like the MCP server) and hangs.
    r.claude_bin = fake_claude(tmp_path, f"sleep 30 & echo $! > {marker}\nsleep 30\n")
    t0 = time.monotonic()
    res = r.run("ask", D)
    assert not res.ok and "timeout" in res.error and time.monotonic() - t0 < 10
    pid = int(marker.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_fake_runner_reports_invalid_results():
    w = FakeWorkerRunner(lambda m, d: {"action": "nope", "summary": ""})
    assert not w.run("ask", D).ok
    w = FakeWorkerRunner(lambda m, d: {"action": "error", "summary": "bad"})
    res = w.run("ask", D)
    assert not res.ok and res.error == "bad"
