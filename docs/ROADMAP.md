# Roadmap and design decisions

Goal: own the full pipeline (camera -> world state -> memory), borrowing proven
NVR design patterns rather than reinventing them. Reference projects to read
(MIT/Apache, read the source, don't copy blindly): Frigate, Viseron, LLM Vision.
Framework behaviour described below is from general knowledge of those
projects; verify against their current docs before depending on it.

## Patterns adopted
| # | Pattern | Where it lives |
|---|---------|----------------|
| 1 | Substream for detection, mainstream for crops | `sources/dual.py` (`DualGrabber`) |
| 2 | Motion gate before the detector | `MotionGate` in `pipeline.py` (week 2) |
| 3 | One process per stage, shared-memory frames | week 2+, only if profiling shows the GIL hurts |
| 4 | Track lifecycle + best snapshot | `Track`, `TrackState`, `BestSnapshot` (week 3) |
| 5 | Zones/masks, evaluated at the foot point | `geometry.py`, `Zone`, `ZoneResolver` (week 3) |
| 6 | Stationary-object handling | `TrackState.STATIONARY` (week 3) |
| 7 | VLM sees best-snapshot crops, constrained JSON, hourly budget | `VisionLLM`, `EventGate` (week 5) |
| 8 | Structured events as the integration surface | `Event` (week 6) |

## The product (not borrowed)
- **WorldState**: entity-centric live model (who, where, doing what, since when, confidence).
- **Memory**: append-only structured events + derived summaries, agent-queryable, entity continuity across time/cameras.
- **ActivityFuser**: activity is a hypothesis fused from VLM + zone + objects + dwell, never one caption.

## Conventions
- Boxes and polygons are normalized 0..1 everywhere. Never pass pixel coords between stages.
- No unbounded queues. Newest frame wins.
- Identity runs once per track on its best snapshot. "unknown" is a valid answer.
- The VLM never writes free text into memory; it fills a typed hypothesis.
- Local-only. Credentials are never logged. Retention window is configured, not implicit.

## Weeks (no code beyond week 1 before sign-off on the previous week's numbers)
1. **Capture on real hardware.** Done when `AGENT_PROMPT.md` acceptance criteria hold.
2. **Motion gate + detector** on the substream; thread-based stages with clean interfaces. Move to processes + shared memory only if profiling justifies it.
3. **Tracker + lifecycle + zones.** Done when track ids are stable through brief occlusion on recorded clips, stationary objects stop consuming detector time, and zone dwell times are correct on a labeled clip.
4. **Identity.** Face embeddings per track on best snapshot; explicit unknown class; measured on labeled clips (report FAR/FRR at the chosen threshold).
5. **Event gate + VLM.** Local VLM on best-snapshot crops; typed JSON; evaluated on ~100 hand-labeled events; VLM calls per hour reported.
6. **World state + memory.** Redis (TTL) for live entities; Postgres or Elasticsearch event log; daily summarizer; query layer ("when did I last leave the desk?").
7. **CCTV swap + hardening.** RTSP substream/mainstream split, soak test, night/IR behaviour, retention and deletion.

## Risks
- Cheap CCTV substreams may be too low-res for faces; crops must come from the mainstream, so the mainstream must stay connected.
- Sub and main are separate connections, so a crop can be a few tens of ms off the detection frame. Fine for slow subjects only.
- Night/IR will degrade identity and VLM quality; evaluate separately.
- Biometric data (face embeddings, footage of other people): local-only, encrypted at rest, retention-limited, and a stated policy for anyone else who lives in or visits the flat.
