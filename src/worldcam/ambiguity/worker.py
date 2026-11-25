"""Worker runners: one stateless `claude -p` run per doubt step.

Verified against claude 2.1.307 (see docs/ambiguity.md, "Worker sandbox"):
  --tools ""                    -> the init event reports tools: [] (no Bash/Read/Write/WebFetch...)
  --strict-mcp-config + --mcp-config <file>  -> only our worldcam server
  --allowedTools mcp__worldcam__* + --permission-mode dontAsk -> MCP tools run, nothing prompts
  --max-turns works although `--help` does not list it
  --json-schema takes the schema JSON *text* (a file path is rejected), the
    validated object comes back in `structured_output`
Extra hardening: --setting-sources "" (no user/project settings or hooks),
--disable-slash-commands (no skills), --no-session-persistence, and the run's
cwd is an empty temp dir (no project CLAUDE.md).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from worldcam.graph.api import Doubt

PROMPTS = Path(__file__).with_name("prompts")
MODES = ("ask", "resolve")
ACTIONS = ("asked", "resolved", "followup", "dismissed", "error")
SECRET_ENV_PREFIXES = ("TELEGRAM_",)          # the worker never needs these


@dataclass
class WorkerResult:
    ok: bool
    action: str = "error"
    summary: str = ""
    error: Optional[str] = None
    wall_s: float = 0.0
    cost_usd: Optional[float] = None
    data: dict = field(default_factory=dict)


class WorkerRunner(Protocol):
    def run(self, mode: str, doubt: Doubt) -> WorkerResult: ...


def load_schema() -> dict:
    return json.loads((PROMPTS / "worker_result.schema.json").read_text())


def validate_result(obj: Any, schema: Optional[dict] = None) -> dict:
    """Small validator for the worker result schema (no jsonschema dependency)."""
    schema = schema or load_schema()
    if not isinstance(obj, dict):
        raise ValueError("result is not an object")
    props = schema["properties"]
    for k in schema["required"]:
        if k not in obj:
            raise ValueError(f"missing {k}")
    for k, v in obj.items():
        if k not in props:
            raise ValueError(f"unexpected key {k}")
        p = props[k]
        t = p["type"]
        if t == "string" and (not isinstance(v, str) or len(v) > p.get("maxLength", 10**6)):
            raise ValueError(f"{k} must be a string")
        if t == "integer" and (isinstance(v, bool) or not isinstance(v, int) or v < p.get("minimum", -10**9)):
            raise ValueError(f"{k} must be an integer")
        if t == "array" and (not isinstance(v, list) or len(v) > p.get("maxItems", 10**6)
                             or not all(isinstance(x, str) for x in v)):
            raise ValueError(f"{k} must be a list of strings")
        if "enum" in p and v not in p["enum"]:
            raise ValueError(f"{k} must be one of {p['enum']}")
    return obj


def parse_claude_output(stdout: str) -> dict:
    """Parse `--output-format json` output and return the validated structured result."""
    try:
        env = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"claude output is not JSON: {e}") from None
    if not isinstance(env, dict) or env.get("type") != "result":
        raise ValueError("claude output is not a result object")
    if env.get("is_error") or env.get("subtype") != "success":
        raise ValueError(f"claude run failed: {env.get('subtype')} {str(env.get('result'))[:200]}")
    obj = env.get("structured_output")
    if obj is None:  # fall back to the text result
        try:
            obj = json.loads(env.get("result") or "")
        except json.JSONDecodeError:
            raise ValueError("no structured_output in claude result") from None
    return validate_result(obj)


class ClaudeWorkerRunner:
    def __init__(self, store_spec: str, snapshot_dir: str, *, claude_bin: str = "claude",
                 python_bin: str = sys.executable, model: Optional[str] = None,
                 timeout_s: float = 120.0, max_turns: int = 12, max_rounds: int = 3,
                 src_dir: Optional[str] = None):
        if store_spec != "neo4j" and not store_spec.startswith("file:"):
            raise ValueError("store_spec must be 'neo4j' or 'file:<path>'")
        self.store_spec = store_spec
        self.snapshot_dir = os.path.abspath(snapshot_dir)
        self.claude_bin = claude_bin
        self.python_bin = python_bin
        self.model = model
        self.timeout_s = timeout_s
        self.max_turns = max_turns
        self.max_rounds = max_rounds
        self.src_dir = src_dir or str(Path(__file__).resolve().parents[2])
        self.last_stdout = ""

    # -- building blocks (unit-tested) ----------------------------------------
    def mcp_config(self, doubt_id: str) -> dict:
        return {"mcpServers": {"worldcam": {
            "type": "stdio",
            "command": self.python_bin,
            "args": ["-m", "worldcam.ambiguity.mcp_server", "--store", self.store_spec,
                     "--snapshot-dir", self.snapshot_dir, "--doubt", doubt_id,
                     "--max-rounds", str(self.max_rounds)],
            "env": {"PYTHONPATH": self.src_dir},
        }}}

    def prompt(self, mode: str, doubt_id: str) -> str:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return (PROMPTS / f"{mode}.md").read_text().format(doubt_id=doubt_id)

    def build_command(self, mode: str, doubt: Doubt, mcp_config_path: str) -> list[str]:
        cmd = [self.claude_bin, "-p", self.prompt(mode, doubt.id),
               "--strict-mcp-config", "--mcp-config", mcp_config_path,
               "--tools", "",
               "--allowedTools", "mcp__worldcam__*",
               "--permission-mode", "dontAsk",
               "--max-turns", str(self.max_turns),
               "--output-format", "json",
               "--json-schema", json.dumps(load_schema(), separators=(",", ":")),
               "--append-system-prompt", (PROMPTS / "system.md").read_text(),
               "--setting-sources", "",
               "--disable-slash-commands",
               "--no-session-persistence"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def child_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(SECRET_ENV_PREFIXES)}
        env["WORLDCAM_SNAPSHOT_DIR"] = self.snapshot_dir
        return env

    # -- run ------------------------------------------------------------------
    def run(self, mode: str, doubt: Doubt) -> WorkerResult:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="worldcam-worker-") as tmp:
            cfg = os.path.join(tmp, "worldcam-mcp.json")
            with open(cfg, "w") as f:
                json.dump(self.mcp_config(doubt.id), f)
            cwd = os.path.join(tmp, "cwd")
            os.mkdir(cwd)
            cmd = self.build_command(mode, doubt, cfg)
            proc = subprocess.Popen(cmd, cwd=cwd, env=self.child_env(), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    start_new_session=True)
            try:
                out, err = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                return WorkerResult(False, error=f"timeout after {self.timeout_s:.0f}s",
                                    wall_s=time.monotonic() - t0)
            finally:
                if proc.poll() is None:
                    _kill_group(proc)
        wall = time.monotonic() - t0
        self.last_stdout = out
        if proc.returncode != 0 and not out.strip():
            return WorkerResult(False, error=f"claude exit {proc.returncode}: {err.strip()[:300]}", wall_s=wall)
        try:
            obj = parse_claude_output(out)
        except ValueError as e:
            return WorkerResult(False, error=str(e), wall_s=wall)
        cost = None
        try:
            cost = json.loads(out).get("total_cost_usd")
        except Exception:
            pass
        return WorkerResult(obj["action"] != "error", obj["action"], obj["summary"],
                            None if obj["action"] != "error" else obj["summary"],
                            wall_s=wall, cost_usd=cost, data=obj)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class FakeWorkerRunner:
    """Test runner: `script(mode, doubt) -> dict` mutates the store like a worker
    would (usually through WorldcamTools) and returns the result object."""

    def __init__(self, script: Callable[[str, Doubt], dict]):
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def run(self, mode: str, doubt: Doubt) -> WorkerResult:
        self.calls.append((mode, doubt.id))
        try:
            obj = validate_result(self.script(mode, doubt))
        except Exception as e:  # mirrors a crashed / invalid run
            return WorkerResult(False, error=f"{type(e).__name__}: {e}")
        ok = obj["action"] != "error"
        return WorkerResult(ok, obj["action"], obj["summary"], None if ok else obj["summary"], data=obj)


WORKER_BACKENDS = ("auto", "gemini", "claude")


def __getattr__(name: str):
    if name in ("GeminiWorkerRunner", "get_gemini_setting"):
        from worldcam.ambiguity import gemini
        return getattr(gemini, name)
    raise AttributeError(name)


def make_worker_runner(
    backend: str = "auto",
    store_spec: str = "neo4j",
    snapshot_dir: str = "captures/snapshots",
    *,
    model: Optional[str] = None,
    claude_bin: str = "claude",
    timeout_s: float = 120.0,
    max_turns: int = 12,
    max_rounds: int = 3,
) -> WorkerRunner:
    """Construct the ambiguity worker runner (`gemini`, `claude`, or `auto`).

    `auto` uses Gemini when GEMINI_API_KEY is set (env or .env), otherwise `claude -p`.
    """
    from worldcam.ambiguity.gemini import GeminiWorkerRunner, get_gemini_setting

    if backend not in WORKER_BACKENDS:
        raise ValueError(f"worker backend must be one of {WORKER_BACKENDS}, got {backend!r}")
    use_gemini = backend == "gemini" or (backend == "auto" and bool(get_gemini_setting("GEMINI_API_KEY")))
    if use_gemini:
        return GeminiWorkerRunner(
            store_spec, snapshot_dir, model=model, timeout_s=timeout_s,
            max_turns=max_turns, max_rounds=max_rounds,
        )
    return ClaudeWorkerRunner(
        store_spec, snapshot_dir, claude_bin=claude_bin, model=model,
        timeout_s=timeout_s, max_turns=max_turns, max_rounds=max_rounds,
    )
