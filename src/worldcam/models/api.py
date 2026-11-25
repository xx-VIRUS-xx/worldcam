"""Model layer contract. Backends (claude, gemini, jev) implement ModelClient
and/or ToolRunner; callers only see this module plus ModelRouter.

Two shapes of work:
  ModelClient.generate  one shot: text or JSON in, text or JSON out, optional images.
                        Cheap/frequent jobs: describe a crop, classify a reply, write a summary.
  ToolRunner.run        a tool loop: the model calls ToolSpecs until it returns the final JSON.
                        Agent work: resolve a doubt, answer a chat question, a curiosity loop.

Tasks pick a backend by explicit config; there is no silent fallback to another backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

TASKS = (
    "agent",         # autonomous loop work (tools)
    "chat",          # answering a user (tools)
    "question",      # wording a question to the owner
    "resolve",       # turning an answer into graph writes (tools)
    "vision_crop",   # describe a crop: breed, colour, clothing, object type
    "vision_scene",  # describe a scene / activity
    "decide",        # small typed decisions (classify a reply, rate a doubt); Jev's job
    "summarize",     # day summaries, persona summaries
)


class NotConfigured(RuntimeError):
    """The configured backend for a task has no credentials / is unavailable."""


@dataclass(frozen=True)
class GenRequest:
    task: str
    prompt: str
    system: Optional[str] = None
    images: tuple[str, ...] = ()          # local image paths (confined by the caller)
    json_schema: Optional[dict] = None    # if set, response.data must validate against it
    max_tokens: int = 1024


@dataclass(frozen=True)
class GenResponse:
    text: str
    data: Optional[dict] = None           # parsed JSON when json_schema was given
    model: str = ""
    usage: dict = field(default_factory=dict)   # input_tokens, output_tokens, cost_usd if known
    latency_ms: float = 0.0


class ModelClient(Protocol):
    name: str                             # "gemini:gemini-2.5-flash"
    def generate(self, req: GenRequest) -> GenResponse: ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict                    # JSON Schema (object)
    fn: Callable[[dict], Any]             # in-process implementation; returns JSON-able data
    writes: bool = False                  # True for tools that change state


@dataclass(frozen=True)
class ToolSet:
    """Tools a run may call. `factory` ("pkg.module:build") + `factory_args`
    (JSON-able) let an out-of-process runner (claude -p + MCP server) rebuild the
    same tools in its child process; in-process runners use `specs` directly."""
    specs: tuple[ToolSpec, ...]
    factory: Optional[str] = None
    factory_args: dict = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)


@dataclass(frozen=True)
class AgentRunRequest:
    task: str
    prompt: str
    system: Optional[str] = None
    toolset: ToolSet = field(default_factory=lambda: ToolSet(()))
    images: tuple[str, ...] = ()
    output_schema: Optional[dict] = None
    max_turns: int = 12
    timeout_s: float = 120.0


@dataclass(frozen=True)
class AgentRunResult:
    ok: bool
    data: Optional[dict] = None           # validated against output_schema when given
    text: str = ""
    error: Optional[str] = None
    model: str = ""
    tool_calls: int = 0
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class ToolRunner(Protocol):
    name: str
    def run(self, req: AgentRunRequest) -> AgentRunResult: ...


class ModelRouter(Protocol):
    def client(self, task: str) -> ModelClient:
        """Raises NotConfigured if the task's backend can't be used."""
    def runner(self, task: str) -> ToolRunner: ...
    def describe(self) -> dict[str, str]:
        """task -> "backend:model" actually configured (for logs / --help)."""


# ---- fakes for tests --------------------------------------------------------

class FakeModelClient:
    """Returns scripted responses: `script(req) -> str | dict | GenResponse`."""

    def __init__(self, script: Callable[[GenRequest], Any] = lambda r: "ok", name: str = "fake:model"):
        self.script, self.name = script, name
        self.calls: list[GenRequest] = []

    def generate(self, req: GenRequest) -> GenResponse:
        self.calls.append(req)
        t0 = time.perf_counter()
        out = self.script(req)
        if isinstance(out, GenResponse):
            return out
        if isinstance(out, dict):
            import json
            return GenResponse(json.dumps(out), data=out, model=self.name,
                               latency_ms=(time.perf_counter() - t0) * 1000)
        return GenResponse(str(out), model=self.name, latency_ms=(time.perf_counter() - t0) * 1000)


class FakeToolRunner:
    """`script(req, call_tool)` plays the model: it may call `call_tool(name, args)`
    (which dispatches to the request's ToolSpecs) and returns the final dict."""

    def __init__(self, script: Callable[[AgentRunRequest, Callable[[str, dict], Any]], dict],
                 name: str = "fake:runner"):
        self.script, self.name = script, name
        self.calls: list[AgentRunRequest] = []

    def run(self, req: AgentRunRequest) -> AgentRunResult:
        self.calls.append(req)
        specs = {s.name: s for s in req.toolset.specs}
        n = 0

        def call_tool(name: str, args: Optional[dict] = None) -> Any:
            nonlocal n
            n += 1
            if name not in specs:
                raise KeyError(f"unknown tool {name!r}")
            return specs[name].fn(args or {})

        try:
            data = self.script(req, call_tool)
        except Exception as e:  # a failing script = a failed run, like a real backend
            return AgentRunResult(False, error=f"{type(e).__name__}: {e}", model=self.name, tool_calls=n)
        return AgentRunResult(True, data=data, model=self.name, tool_calls=n)


class FakeModelRouter:
    def __init__(self, client: Optional[ModelClient] = None, runner: Optional[ToolRunner] = None):
        self._client, self._runner = client or FakeModelClient(), runner

    def client(self, task: str) -> ModelClient:
        return self._client

    def runner(self, task: str) -> ToolRunner:
        if self._runner is None:
            raise NotConfigured(f"no runner for task {task!r}")
        return self._runner

    def describe(self) -> dict[str, str]:
        return {t: "fake" for t in TASKS}
