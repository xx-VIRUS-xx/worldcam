"""ConfiguredModelRouter: task -> "backend:model" from models.json + env, no silent fallback.

Config file (JSON, repo root `models.json`, see `models.example.json`):
    {"agent": "claude:sonnet", "vision_crop": "gemini:gemini-2.5-flash", ...,
     "options": {"claude_bin": "claude", "usage_log": "captures/usage.jsonl"}}
Per-task env override: WORLDCAM_MODEL_<TASK>=backend:model (e.g. WORLDCAM_MODEL_VISION_CROP=claude:haiku).
Config path: constructor arg > $WORLDCAM_MODELS_CONFIG > ./models.json > <repo>/models.json > built-in defaults.
A backend without credentials raises NotConfigured when a task asks for it, naming the variable.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .api import (TASKS, AgentRunRequest, AgentRunResult, GenRequest, GenResponse, ModelClient,
                  NotConfigured, ToolRunner)
from .usage import UsageLog
from .util import REPO_ROOT, get_secret

DEFAULTS = {
    "agent": "claude:sonnet",
    "chat": "claude:sonnet",
    "question": "claude:haiku",
    "resolve": "claude:sonnet",
    "vision_crop": "gemini:gemini-2.5-flash",
    "vision_scene": "gemini:gemini-2.5-flash",
    "decide": "claude:haiku",       # built-in default switches to jev:default when a Jev key is present
    "summarize": "claude:sonnet",
}
TASK_ENV_PREFIX = "WORLDCAM_MODEL_"

# backend name -> factory(model, options, env, kwargs) -> object with generate() and/or run()
BACKENDS: dict[str, Callable[..., Any]] = {}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    BACKENDS[name] = factory


def _claude(model, options, env, kw):
    from .claude_cli import ClaudeCliClient, ClaudeCliRunner

    claude_bin = options.get("claude_bin", "claude")
    if "spawn" not in kw and shutil.which(claude_bin) is None:
        raise NotConfigured(f"claude backend needs the `claude` CLI on PATH (or options.claude_bin); "
                            f"not found: {claude_bin!r}. Install Claude Code and log in with `claude`")
    kw = {"claude_bin": claude_bin, **kw}
    client, runner = ClaudeCliClient(model, **kw), ClaudeCliRunner(model, **kw)
    client.run = runner.run          # one object serves both shapes
    return client


def _gemini(model, options, env, kw):
    from .gemini import GeminiClient, GeminiToolRunner, require_key
    key = require_key(env)
    if not model or model == "default":
        raise NotConfigured("gemini needs an explicit model, e.g. gemini:gemini-2.5-flash")
    client, runner = GeminiClient(model, key, **kw), GeminiToolRunner(model, key, **kw)
    client.run = runner.run
    return client


def _jev(model, options, env, kw):
    from .jev import JevClient, require_key
    url = get_secret("JEV_BASE_URL", env) or options.get("jev_url")
    return JevClient(model or "default", require_key(env), **({"url": url} if url else {}), **kw)


def _claude_api(model, options, env, kw):
    from .claude_api import ClaudeApiClient, require_key
    return ClaudeApiClient(model or "default", require_key(env), **kw)


register_backend("claude", _claude)
register_backend("gemini", _gemini)
register_backend("jev", _jev)
register_backend("claude-api", _claude_api)


def parse_spec(spec: str, what: str = "model spec") -> tuple[str, str]:
    backend, sep, model = str(spec).partition(":")
    if not sep or not backend.strip():
        raise ValueError(f"{what} {spec!r} must look like 'backend:model' (e.g. claude:sonnet)")
    return backend.strip(), model.strip()


class _Metered:
    """Wraps a backend object: records usage per task, never prompts or outputs."""

    def __init__(self, inner: Any, task: str, log: UsageLog):
        self._inner, self._task, self._log = inner, task, log
        self.name = inner.name

    def generate(self, req: GenRequest) -> GenResponse:
        try:
            r = self._inner.generate(req)
        except Exception:
            self._log.record(self._task, self.name, ok=False)
            raise
        self._log.record(self._task, r.model or self.name, r.usage, r.latency_ms)
        return r

    def run(self, req: AgentRunRequest) -> AgentRunResult:
        r = self._inner.run(req)
        self._log.record(self._task, r.model or self.name, r.usage, r.latency_ms, ok=r.ok)
        return r


class ConfiguredModelRouter:
    def __init__(self, config: Optional[dict] = None, *, config_path: Optional[str] = None,
                 env: Optional[dict] = None, usage: Optional[UsageLog] = None,
                 backend_kwargs: Optional[dict[str, dict]] = None):
        """`config` (task->spec dict) beats files; `env` defaults to os.environ (tests pass a dict and
        then no .env file is read); `backend_kwargs[backend]` injects transport/spawn for tests."""
        self.env = os.environ if env is None else env
        raw = config if config is not None else self._load(config_path)
        self.options: dict = dict(raw.get("options", {}))
        specs = {t: v for t, v in raw.items() if t != "options"}
        bad = sorted(set(specs) - set(TASKS))
        if bad:
            raise ValueError(f"unknown task(s) in model config: {bad}; valid: {list(TASKS)}")
        self.specs: dict[str, str] = dict(self._defaults())
        self.specs.update(specs)
        for t in TASKS:                                   # env override wins
            v = self.env.get(TASK_ENV_PREFIX + t.upper())
            if v:
                self.specs[t] = v
        for t, s in self.specs.items():
            parse_spec(s, f"task {t!r}")                 # fail early on malformed specs
        log_path = self.options.get("usage_log") or self.env.get("WORLDCAM_USAGE_LOG")
        self.usage = usage or UsageLog(path=log_path)
        self.backend_kwargs = backend_kwargs or {}
        self._cache: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()

    def _defaults(self) -> dict:
        d = dict(DEFAULTS)
        from .jev import find_key
        if find_key(self.env):
            d["decide"] = "jev:default"
        return d

    def _load(self, path: Optional[str]) -> dict:
        cands = [path] if path else [self.env.get("WORLDCAM_MODELS_CONFIG"), Path.cwd() / "models.json",
                                     REPO_ROOT / "models.json"]
        for p in cands:
            if p and Path(p).is_file():
                try:
                    return json.loads(Path(p).read_text())
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}: invalid JSON: {e}") from None
            if p and path:
                raise FileNotFoundError(f"model config not found: {p}")
        return {}

    # ---- ModelRouter ---------------------------------------------------------
    def _backend(self, task: str) -> Any:
        if task not in TASKS:
            raise KeyError(f"unknown task {task!r}; valid: {list(TASKS)}")
        spec = self.specs[task]
        backend, model = parse_spec(spec)
        factory = BACKENDS.get(backend)
        if factory is None:
            raise NotConfigured(f"task {task!r}: unknown backend {backend!r} in {spec!r}; "
                                f"known: {sorted(BACKENDS)}")
        key = (backend, model, task)
        with self._lock:
            if key not in self._cache:
                inner = factory(model, self.options, self.env, dict(self.backend_kwargs.get(backend, {})))
                self._cache[key] = (inner, _Metered(inner, task, self.usage))
            return self._cache[key]

    def client(self, task: str) -> ModelClient:
        inner, metered = self._backend(task)
        if not hasattr(inner, "generate"):
            raise NotConfigured(f"task {task!r}: backend {self.specs[task]!r} cannot do one-shot generation")
        return metered

    def runner(self, task: str) -> ToolRunner:
        inner, metered = self._backend(task)
        if not hasattr(inner, "run"):
            raise NotConfigured(f"task {task!r}: backend {self.specs[task]!r} cannot run tool loops "
                                f"(use claude: or gemini:)")
        return metered

    def describe(self) -> dict[str, str]:
        return {t: self.specs[t] for t in TASKS}
