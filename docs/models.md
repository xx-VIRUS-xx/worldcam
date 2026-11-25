# Models layer

`ModelRouter` (contract: `src/worldcam/models/api.py`) maps each **task** to a `backend:model`. Callers
ask `router.client(task)` for one-shot work (`generate`) or `router.runner(task)` for tool loops (`run`).
There is **no silent fallback**: a backend without credentials raises `NotConfigured` naming the missing
variable, when the task first asks for it.

## Config
`models.json` at the repo root (copy `models.example.json`). Lookup order: constructor arg,
`$WORLDCAM_MODELS_CONFIG`, `./models.json`, `<repo>/models.json`, built-in defaults. Keys are tasks;
`options` holds `claude_bin`, `usage_log` (JSONL path), `jev_url`. Per-task env override:
`WORLDCAM_MODEL_<TASK>=backend:model` (e.g. `WORLDCAM_MODEL_VISION_CROP=claude:haiku`). Unknown tasks and
malformed specs fail at startup. `router.describe()` returns the effective task -> spec map for logs.

| Task | Default | Notes |
|---|---|---|
| agent, chat, resolve, summarize | `claude:sonnet` | tool loops (agent/chat/resolve) |
| question | `claude:haiku` | ask-mode of the ambiguity worker |
| vision_crop, vision_scene | `gemini:gemini-2.5-flash` | `claude:haiku` also works (see below) |
| decide | `claude:haiku` | the built-in default becomes `jev:default` when `JEV_API_KEY`/`AIMLAPI_KEY` is set; an explicit config always wins |

## Backends
| Backend | Credential | One-shot | Tool loop | Images | Structured output |
|---|---|---|---|---|---|
| `claude:<alias>` | the `claude` CLI login (on PATH or `options.claude_bin`) | yes | yes (generated MCP server) | **yes** | `--json-schema` |
| `gemini:<model>` | `GEMINI_API_KEY` (env or `.env`) | yes | yes (function calling) | yes | `responseJsonSchema` |
| `jev:default` | `JEV_API_KEY` or `AIMLAPI_KEY` | yes (`decide` only) | no | no | typed questions |
| `claude-api:<full model id>` | `ANTHROPIC_API_KEY` | yes | no | yes | forced tool |

**Claude and images.** `claude -p --input-format stream-json` accepts base64 image blocks (verified:
a red rectangle came back as `"red"` with `--json-schema`), so Claude can do vision tasks; it costs
more tokens per call than Gemini Flash. Prompts go over stdin, never argv.

**Claude tool loops.** `ToolSet.factory` (`"worldcam.ambiguity.tools:build_toolset"`) plus JSON `factory_args`
are written to a 0600 MCP config in a temp dir; the child `python -m worldcam.ambiguity.mcp_server`
rebuilds the tools (factory refs must live under `worldcam.`). Same sandbox as the original worker:
`--tools ""`, `--strict-mcp-config`, `--allowedTools mcp__worldcam__*`, `--permission-mode dontAsk`,
no settings/slash commands/session persistence, empty cwd, scrubbed env (no `TELEGRAM_*`, `GEMINI_*`,
`JEV_*`, `AIMLAPI_*`). Factory args are not secrets (secrets such as `NEO4J_PASSWORD` stay in the env).

**Gemini tool loops.** `functionDeclarations` come from the ToolSpecs; a required output schema becomes an
extra `final_answer` function (Gemini cannot mix function calling and JSON mode). Tool errors go back to
the model as `{"error": ...}`. The API key is sent as the `x-goog-api-key` header and redacted in errors.
Unverified live (no key was available).

**Jev.** TypeSafe's own REST docs were not reachable; the only documented API found is the AI/ML API
aggregator (`POST https://api.aimlapi.com/v1/decisions`, bearer key, `state` + typed `questions`).
The adapter maps a `GenRequest` to it: `prompt` = state, `json_schema.properties.<q>["x-jev"]` =
`{type: noul|choice|score, instructions, criteria}`; `data[q]` is P(yes), the chosen key, or the score.
Set `JEV_BASE_URL` to point elsewhere. Unverified live (early access, no key).

## What leaves the machine
- `claude`: the prompt, system text, images and tool results go to Anthropic through the CLI; the graph
  itself stays local (the MCP server runs locally, only results the model asks for are sent).
- `claude-api`: same, direct to `api.anthropic.com`.
- `gemini`: prompt, images and tool results go to Google (`generativelanguage.googleapis.com`).
- `jev`: the state text (no images) goes to the configured endpoint (aimlapi.com by default).
Choose per task: keep face crops on a backend you accept, or on `claude` only.

## Ambiguity worker
`make_worker_runner(store_spec, snapshot_dir, router=None, worker=None, worker_model=None)` builds the
engine's `WorkerRunner`. Default: per mode from the router (`ask` -> task `question`, `resolve` -> task
`resolve`). `claude` uses the verified `ClaudeWorkerRunner`; other backends use the generic ToolRunner path
over `build_toolset` (same per-doubt scope, image confinement, no delete tool, read-only `graph_read`,
no owner creation). `worker="claude:sonnet"` / `"gemini:<model>"` forces a backend for both modes (the old
`--worker`); `worker_model` is the old `--worker-model`. `cli/run.py` should call this instead of
`ClaudeWorkerRunner(...)` and catch `NotConfigured` at startup.

## Usage log
`router.usage` (`UsageLog`) keeps the last 500 calls in memory and appends JSONL to `options.usage_log`
(or `$WORLDCAM_USAGE_LOG`): ts, task, model, ok, input/output tokens, latency, cost (CLI-reported, or
estimated for known Gemini models). No prompts, images or outputs. `router.usage.summary()` aggregates per task.

## Adding a backend
Write a class with `name` and `generate(GenRequest)` and/or `run(AgentRunRequest)`, raise `NotConfigured`
(naming the env var) from its factory when credentials are missing, then
`register_backend("name", lambda model, options, env, kwargs: Backend(...))` in `models/router.py`. Take a
transport/spawn callable so tests run with no network (see `tests/test_models_backends.py`).
