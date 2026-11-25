"""`claude` backend: one-shot and tool-loop runs through the Claude Code CLI (`claude -p`).

Flags verified against claude 2.1.307 (see ambiguity/worker.py and docs/models.md):
  --tools ""                  no built-in tools (no Bash/Read/Write/WebFetch)
  --json-schema <json text>   structured output, returned as `structured_output`
  --output-format json        one result object (text input)
  --input-format stream-json  accepts image content blocks (base64) on stdin; verified
                              empirically (a red rectangle is described as "Red"), so
                              claude CAN do vision. Output then uses stream-json too.
  --strict-mcp-config + --mcp-config <file>, --allowedTools mcp__worldcam__*,
  --permission-mode dontAsk   only our generated MCP server, nothing prompts.
The prompt goes through stdin, never argv. The child gets a scrubbed environment
(no Telegram/Gemini/Jev secrets) and runs in an empty temp dir.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from .api import AgentRunRequest, AgentRunResult, GenRequest, GenResponse
from .util import ModelError, check_schema, load_images, parse_json_text

SCRUB_PREFIXES = ("TELEGRAM_", "GEMINI_", "GOOGLE_API", "JEV_", "AIMLAPI_")
MCP_NAME = "worldcam"

# runner(cmd, stdin_text, env, cwd, timeout) -> (returncode, stdout, stderr)   (fakeable)
Spawn = Callable[[list, str, dict, str, float], tuple[int, str, str]]


def default_spawn(cmd: list, stdin_text: str, env: dict, cwd: str, timeout: float) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.communicate()
        raise ModelError(f"claude timed out after {timeout:.0f}s") from None
    finally:
        if proc.poll() is None:
            _kill_group(proc)
    return proc.returncode, out, err


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def child_env(extra: Optional[dict] = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(SCRUB_PREFIXES)}
    env.update(extra or {})
    return env


def stdin_payload(prompt: str, images: tuple[str, ...]) -> tuple[str, bool]:
    """-> (stdin text, uses_stream_json). Images require stream-json input."""
    if not images:
        return prompt, False
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
              for mime, b64 in load_images(images)]
    blocks.append({"type": "text", "text": prompt})
    return json.dumps({"type": "user", "message": {"role": "user", "content": blocks}}) + "\n", True


def parse_result(stdout: str) -> dict:
    """The final `result` event from `--output-format json` or stream-json output."""
    stdout = stdout.strip()
    cands = []
    try:
        v = json.loads(stdout)
        cands = v if isinstance(v, list) else [v]
    except json.JSONDecodeError:
        for line in stdout.splitlines():
            try:
                cands.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    res = [c for c in cands if isinstance(c, dict) and c.get("type") == "result"]
    if not res:
        raise ModelError("claude output has no result event")
    env = res[-1]
    if env.get("is_error") or env.get("subtype") != "success":
        raise ModelError(f"claude run failed: {env.get('subtype')} {str(env.get('result'))[:200]}")
    return env


def _usage(env: dict) -> dict:
    u = env.get("usage") or {}
    inp = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
        + (u.get("cache_creation_input_tokens") or 0)
    return {"input_tokens": inp, "output_tokens": u.get("output_tokens") or 0,
            "cost_usd": env.get("total_cost_usd")}


def _structured(env: dict, schema: Optional[dict]) -> Optional[dict]:
    if schema is None:
        return None
    obj = env.get("structured_output")
    if obj is None:
        obj = parse_json_text(env.get("result") or "")
    if obj is None:
        raise ModelError("no structured_output in claude result")
    try:
        check_schema(obj, schema)
    except ValueError as e:
        raise ModelError(f"claude result violates schema: {e}") from None
    return obj


class _ClaudeBase:
    def __init__(self, model: Optional[str] = None, *, claude_bin: str = "claude",
                 spawn: Spawn = default_spawn):
        self.model = model or None
        self.claude_bin = claude_bin
        self.spawn = spawn
        self.name = f"claude:{self.model or 'default'}"

    def _base_cmd(self, *, stream: bool, schema: Optional[dict], system: Optional[str],
                  max_turns: int) -> list:
        cmd = [self.claude_bin, "-p", "--tools", "",
               "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
               "--max-turns", str(max_turns)]
        if stream:
            cmd += ["--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
        else:
            cmd += ["--output-format", "json"]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        if system:
            cmd += ["--append-system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def _exec(self, cmd: list, stdin_text: str, env: dict, cwd: str, timeout: float) -> dict:
        rc, out, err = self.spawn(cmd, stdin_text, env, cwd, timeout)
        if rc != 0 and not out.strip():
            raise ModelError(f"claude exit {rc}: {err.strip()[:300]}")
        return parse_result(out)


class ClaudeCliClient(_ClaudeBase):
    """One-shot: `claude -p` with no tools. Text and image prompts, optional JSON schema."""

    def __init__(self, model: Optional[str] = None, *, timeout_s: float = 90.0, **kw):
        super().__init__(model, **kw)
        self.timeout_s = timeout_s

    def build_command(self, req: GenRequest) -> tuple[list, str]:
        stdin_text, stream = stdin_payload(req.prompt, req.images)
        return self._base_cmd(stream=stream, schema=req.json_schema, system=req.system,
                              max_turns=2 if req.json_schema else 1), stdin_text

    def generate(self, req: GenRequest) -> GenResponse:
        t0 = time.perf_counter()
        cmd, stdin_text = self.build_command(req)
        with tempfile.TemporaryDirectory(prefix="worldcam-claude-") as cwd:
            env = self._exec(cmd, stdin_text, child_env(), cwd, self.timeout_s)
        data = _structured(env, req.json_schema)
        text = env.get("result") or ""
        if data is not None and not text:
            text = json.dumps(data)
        return GenResponse(text, data=data, model=self.name, usage=_usage(env),
                           latency_ms=(time.perf_counter() - t0) * 1000)


class ClaudeCliRunner(_ClaudeBase):
    """Tool loop: `claude -p` + a generated stdio MCP server rebuilt from the ToolSet's factory ref."""

    def __init__(self, model: Optional[str] = None, *, python_bin: str = sys.executable,
                 src_dir: Optional[str] = None, **kw):
        super().__init__(model, **kw)
        self.python_bin = python_bin
        self.src_dir = src_dir or str(Path(__file__).resolve().parents[2])

    def mcp_config(self, factory: str, factory_args: dict) -> dict:
        # factory + args travel in the server's env (config file is 0600 in a temp dir), not argv
        return {"mcpServers": {MCP_NAME: {
            "type": "stdio", "command": self.python_bin, "args": ["-m", "worldcam.ambiguity.mcp_server"],
            "env": {"PYTHONPATH": self.src_dir, "WORLDCAM_TOOLSET_FACTORY": factory,
                    "WORLDCAM_TOOLSET_ARGS": json.dumps(factory_args)}}}}

    def build_command(self, req: AgentRunRequest, mcp_config_path: Optional[str]) -> tuple[list, str]:
        stdin_text, stream = stdin_payload(req.prompt, req.images)
        cmd = self._base_cmd(stream=stream, schema=req.output_schema, system=req.system,
                             max_turns=req.max_turns)
        if mcp_config_path:
            cmd += ["--strict-mcp-config", "--mcp-config", mcp_config_path,
                    "--allowedTools", f"mcp__{MCP_NAME}__*", "--permission-mode", "dontAsk"]
        return cmd, stdin_text

    def run(self, req: AgentRunRequest) -> AgentRunResult:
        t0 = time.perf_counter()
        try:
            if req.toolset.specs and not req.toolset.factory:
                raise ModelError("claude backend needs a ToolSet with a factory ref "
                                 "(it runs the tools in a child MCP server)")
            with tempfile.TemporaryDirectory(prefix="worldcam-claude-") as tmp:
                cfg = None
                if req.toolset.specs:
                    cfg = os.path.join(tmp, "mcp.json")
                    fd = os.open(cfg, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "w") as f:
                        json.dump(self.mcp_config(req.toolset.factory, req.toolset.factory_args), f)
                cwd = os.path.join(tmp, "cwd")
                os.mkdir(cwd)
                cmd, stdin_text = self.build_command(req, cfg)
                env = self._exec(cmd, stdin_text, child_env(), cwd, req.timeout_s)
            data = _structured(env, req.output_schema)
            return AgentRunResult(True, data=data, text=env.get("result") or "", model=self.name,
                                  tool_calls=max(int(env.get("num_turns") or 1) - 1, 0),
                                  usage=_usage(env), latency_ms=(time.perf_counter() - t0) * 1000)
        except ModelError as e:
            return AgentRunResult(False, error=str(e), model=self.name,
                                  latency_ms=(time.perf_counter() - t0) * 1000)
