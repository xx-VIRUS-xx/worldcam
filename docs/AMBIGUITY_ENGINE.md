# Perception + ambiguity engine (weeks 2–4 + 6, combined)

Status: design approved on 2026-11-24. This doc is the contract every component builds against.

## Owner decisions (2026-11-24)
| Topic | Decision |
|---|---|
| Messenger | Telegram bot, long polling from the Mac (no inbound port). Behind a `Notifier` interface so a Cloudflare messenger can be added later. |
| Worker | `claude -p` (headless Claude Code), one short run per step, locked down (see "Worker sandbox"). |
| Graph DB | Neo4j 5 Community in Docker (colima), with a vector index for face embeddings. |
| Personas | Built for **everyone** seen, not only household. |

### What this changes in the week-1 rules
- **Local-only is relaxed** for exactly two outbound paths: (1) Claude (via `claude -p`) receives question context and the crop being asked about; (2) Telegram receives the question and that crop, sent to the owner's chat only. Nothing else leaves the Mac. No full frames, no video, no continuous upload.
- **New dependencies** are allowed as optional extras in `pyproject.toml` (`[perception]`, `[identity]`, `[graph]`, `[engine]`). Core capture still needs only opencv + numpy.
- Guardrails kept, because personas cover visitors: owner-only bot (chat-id allowlist), snapshot retention window, `worldcam-forget <person>` hard delete, every stored fact carries its source and confidence, and secrets live only in `.env` (git-ignored, never logged). The owner is responsible for local privacy law on profiling visitors; this is noted, not enforced.

## Data flow
```
DualGrabber(sub, main)
  -> MotionGate (frame diff on sub)          perception/motion.py
  -> Detector (YOLO on sub, motion regions)  perception/detector.py
  -> Tracker (ids, lifecycle, BestSnapshot)  perception/tracker.py
  -> FaceIdentifier (on BestSnapshot crop)   identity/faces.py     -> FaceObservation
  -> GraphStore.match_face / add_sighting    graph/store.py        (Neo4j)
  -> DoubtDetector (rules, rate limits)      ambiguity/triggers.py -> Doubt
  -> AmbiguityEngine                         ambiguity/engine.py
        spawns `claude -p` worker            ambiguity/worker.py
        worker tools = worldcam MCP server   ambiguity/mcp_server.py
  -> Notifier (Telegram)                     messenger/telegram.py
        owner reply -> engine -> new worker run -> graph updated
```

## Doubts: when the system asks
| Kind | Trigger | Example question |
|---|---|---|
| `unknown_face` | Face embedding with no match above `MATCH_T` (0.45 cosine) after N=3 good-quality sightings of the same cluster (so passers-by don't each cause a question). | "I've seen this person 3 times today at the door. Who is it?" [crop] [buttons: known names…, New person, Ignore] |
| `low_confidence` | Best match between `ASK_T` (0.35) and `MATCH_T`. | "Is this Prabhat?" [Yes] [No, someone else] |
| `conflict` | One track got two different identities, or a face cluster matches two persons. | "Are these the same person?" [2 crops] |
| `metadata_gap` | A known person with missing core attributes (relation to owner, name) after K sightings. | "How do you know Rahul?" |

Rate limits (config): at most 6 questions per hour, quiet hours 23:00–08:00 (questions queue as `open`), dedup by face cluster, and at most 3 follow-up rounds per doubt.

## Doubt lifecycle (all state lives in Neo4j; workers are stateless)
```
open --worker run "ask"--> asked --owner replies--> answered --worker run "resolve"--> resolved
  \                                                     |  (worker may ask a follow-up -> asked, rounds+1)
   `--> dismissed (owner taps Ignore, or doubt expires after 7 days)
```
1. The engine picks an `open` doubt (under the rate limits) and runs the worker in `ask` mode. The worker reads context through MCP tools and calls `ask_owner(...)`, which sends the Telegram message and sets the doubt to `asked`. The worker exits; it never waits for a reply.
2. The Telegram poller receives the owner's reply (a text message or a button tap), stores it on the doubt (`answered`) and wakes the engine.
3. The engine runs the worker in `resolve` mode with the doubt and the reply. The worker extracts the name, relation and facts, writes them through MCP tools (upsert person, label the face cluster, add facts, add relations, refresh the persona summary), then either resolves the doubt or asks one follow-up.

## Worker sandbox (`claude -p`)
Owner replies are untrusted text, and so is anything the camera sees (for example, a note held up to the camera). So the worker gets no shell, no files and no web:
```
claude -p "<task prompt>" \
  --strict-mcp-config --mcp-config <generated worldcam-mcp.json> \
  --tools "" --allowedTools "mcp__worldcam__*" \
  --permission-mode dontAsk --max-turns 12 \
  --output-format json --json-schema <worker_result.schema.json> \
  --append-system-prompt "<role + rules>"
```
(Verify each flag against `claude --help` on the installed version. If `--tools ""` doesn't disable built-ins, use `--disallowedTools` with the full built-in list.)
- The MCP server exposes only: `get_doubt`, `person_context`, `search_people`, `graph_read` (parameterised read-only Cypher in a READ transaction), `upsert_person`, `label_face_cluster`, `add_fact`, `relate`, `set_persona_summary`, `merge_people`, `ask_owner`, `resolve_doubt`, `dismiss_doubt`. No delete tool: `forget` is owner-only, through the CLI.
- Each run has a hard timeout of 120 s and is killed if it runs over. The result JSON is validated, and a failed run puts the doubt back to its previous state with `last_error`.
- The prompt marks the owner's reply and any OCR'd text as data, not instructions.

## Graph schema (Neo4j)
```
(:Person {id, name, aliases[], kind: 'owner'|'household'|'visitor'|'unknown',
          first_seen, last_seen, persona_summary, persona_updated})
(:FaceCluster {id, person_id?, n_faces, centroid: vector(512), created})
(:Face {id, embedding: vector(512), quality, ts, snapshot_path})-[:IN]->(:FaceCluster)
(:FaceCluster)-[:BELONGS_TO]->(:Person)
(:Sighting {id, ts, camera, zones[], track_id, activity?, snapshot_path})-[:OF]->(:Person|:FaceCluster)
(:Fact {id, key, value, source: 'owner'|'observed'|'inferred', confidence, ts})<-[:HAS_FACT]-(:Person)
(:Person)-[:RELATED {relation: 'partner'|'friend'|'colleague'|..., source, ts}]->(:Person)
(:Doubt {id, kind, status, rounds, question, answer, created, updated, last_error,
         telegram_message_id})-[:ABOUT]->(:Person|:FaceCluster|:Sighting)
```
Constraints: unique `id` on every label. Vector index `face_embedding` on `Face.embedding` (cosine, 512 dims).

## Python contracts
The shared types and protocols are in `src/worldcam/graph/api.py` and `src/worldcam/messenger/base.py`. Every component codes against those, and tests use the in-memory fakes `InMemoryGraphStore` and `FakeNotifier` from the same modules. The Neo4j implementation must pass the same contract tests (`tests/contract_graph.py`).

## Model and library choices
- Detector: Ultralytics YOLO (`yolo11n.pt` to start, MPS on Apple Silicon). Licence: AGPL-3.0, fine for personal use; revisit before any commercial use.
- Faces: InsightFace `buffalo_l` (SCRFD detection + ArcFace 512-d embedding) through onnxruntime (CPU or CoreML provider). Model weights are for non-commercial research use only; personal use only.
- Graph driver: `neo4j` (official).
- MCP server: `mcp` (official Python SDK), stdio transport.
- Telegram: plain HTTPS to the Bot API with stdlib `urllib`, long polling (`getUpdates`). No bot framework.

## Config (`.env`, git-ignored)
```
TELEGRAM_BOT_TOKEN=...          # from @BotFather
TELEGRAM_OWNER_CHAT_ID=...      # only this chat is served
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
WORLDCAM_SNAPSHOT_DIR=captures/snapshots
WORLDCAM_SNAPSHOT_RETENTION_DAYS=30
```
