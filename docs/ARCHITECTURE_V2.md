# Architecture v2: the agent layer

Builds on `AMBIGUITY_ENGINE.md` (perception -> identity -> graph -> doubts -> worker -> messenger),
which keeps working. Product intent: `VISION.md`.

```
 EYES                                BRAIN                         AGENT                       CHAT / CHANNELS
 camera -> motion -> YOLO -> tracker  graph (Neo4j): people,        AgentLoop (thread)          Channel (Telegram, later voice)
  -> faces (ArcFace)                   animals, objects, faces,      - event triggers             -> ChatHandler (permissions, intent,
  -> entity re-id (embeddings)         sightings, facts, doubts      - schedule + random loops       memory tools, citations)
  -> zones, scene/activity (VLM)      events + routines + summaries - attention budget per user  UserDirectory (roles)
  -> EventRecorder -> BrainEvent      (brain/)                      - profiles: cctv|companion|medical
                                                                    - UncertaintyManager -> AmbiguityEngine
 MODELS: ModelRouter picks a backend per task (claude | gemini | jev); ToolRunner runs tool loops.
```

## Interfaces (the contract; all have in-memory fakes for tests)
| File | Provides |
|---|---|
| `src/worldcam/models/api.py` | `GenRequest/GenResponse`, `ModelClient` (one-shot), `ToolSpec/ToolSet`, `AgentRunRequest/Result`, `ToolRunner` (tool loop), `ModelRouter`, `FakeModelClient`, `FakeToolRunner` |
| `src/worldcam/users/api.py` | roles, permissions, `User`, `UserDirectory`, `InMemoryUserDirectory`, `can_confirm(user, topic)` |
| `src/worldcam/channels/api.py` | `InboundMessage`, `OutboundMessage`, `Channel`, `FakeChannel` |
| `src/worldcam/brain/api.py` | `BrainEvent`, `EventStore`, `Routine`, `RoutineStore`, `InMemoryEventStore` |
| `src/worldcam/agent/api.py` | `Profile`, `Trigger`, `AttentionBudget`, `MemoryContext`, `AgentLoop`, `ChatHandler` |
| `src/worldcam/graph/api.py`, `messenger/base.py`, `pipeline.py` | existing contracts (graph store, notifier, track/identity types) |

## Rules every component follows
- **Answer decides.** No component writes an identity/relation/medical fact as `source="owner"` unless a user with `can_confirm(user, topic)` said it. Agent guesses are `inferred` with confidence <= 0.6.
- **Provenance.** Facts and events record `source` and who confirmed (`confirmed_by` in detail).
- **Models by explicit config.** `ModelRouter` maps task -> `backend:model`; no silent fallback to another backend. Missing credentials = a clear error.
- **Tool sandbox.** Workers get only `ToolSet` tools (graph/event read, scoped writes, ask/notify). No shell, files or web.
- **Bounded everything.** No unbounded queues, histories or caches.
- **Secrets** never logged, never in argv.
- **Tests run with no camera, no network, no Neo4j** (Neo4j contract tests skip cleanly if it is down).

## Component ownership (parallel build, no shared files)
| Agent | Owns (create/edit only these) |
|---|---|
| models | `models/*`, `ambiguity/{worker,gemini,tools,mcp_server}.py`, `docs/models.md`, `tests/test_models_*` |
| users+channels | `users/*`, `channels/*`, `messenger/telegram.py`, `cli/users_admin.py`, `docs/users.md`, `tests/test_users_*`, `tests/test_channels_*`, `tests/test_telegram_*` |
| brain | `brain/*` (except `brain/api.py` additions needing agreement), `cli/brain_admin.py`, `docs/brain.md`, `tests/test_brain_*` |
| entity re-id | `perception/reid.py`, `ambiguity/{entities,triggers}.py`, `graph/*`, `docs/reid.md`, `tests/contract_graph.py`, `tests/test_reid_*`, `tests/test_ambiguity_{triggers,entities}.py` |
| scene + zones | `perception/{scene,zones,eventgate,fuser}.py`, `cli/scene_replay.py`, `docs/scene.md`, `tests/test_scene_*`, `tests/test_zones_*` |
| agent loop | `agent/*` (except `agent/api.py` additions), `ambiguity/engine.py`, `docs/agent.md`, `tests/test_agent_*`, `tests/test_ambiguity_engine.py` |
| chat | `chat/*`, `docs/chat.md`, `tests/test_chat_*` |
| coordinator (me) | `cli/run.py`, `pyproject.toml`, all `api.py` contract files, cross-cutting docs |

Need a contract change? Don't edit `api.py`; describe it in your report.
