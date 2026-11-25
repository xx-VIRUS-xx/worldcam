# Brain: the time dimension (events, routines, summaries)

The graph says WHO/WHAT exists; the brain says WHAT HAPPENED and WHEN. Contract: `src/worldcam/brain/api.py`.

```
TrackUpdate / ResolveResult / entity sightings
   -> EventRecorder (debounce, subject resolution, per-subject cap)
   -> EventStore (Neo4j or SQLite)  -> mine_routines -> Routine (inferred)
                                    -> routine_status (on_time/due/overdue)
                                    -> build_day_summary (+ optional polish_summary)
```

## Modules
| File | What |
|---|---|
| `brain/store.py` | `Neo4jEventStore`, `SqliteEventStore` (both EventStore + RoutineStore) |
| `brain/recorder.py` | `EventRecorder`, `RecorderConfig`, `DEFAULT_PROFILE_MAP` |
| `brain/routines.py` | `mine_routines`, `apply_mining`, `routine_status` |
| `brain/summary.py` | `build_day_summary`, `polish_summary` |
| `brain/retention.py` | `RetentionPolicy.apply`, `forget_subject`, `delete_files` |
| `cli/brain_admin.py` | `python -m worldcam.cli.brain_admin [--sqlite PATH] stats|timeline|routines|mine|summary|purge|forget` |

## Schema (Neo4j)
- `(:Event {id, ts, kind, subject_id, subject_label, zone, camera, activity, confidence, source, snapshot_paths[], profiles[], detail(JSON string)})`
- `(:Routine {id, subject_id, kind, description, params(JSON string), support, confidence, first_seen, last_seen, source, profiles[]})`
- `(:Event)-[:ABOUT]->(:Person)` and `(:Routine)-[:ABOUT]->(:Person)` when a Person with `subject_id` exists in the graph (browsable; the property is the source of truth, so events survive without a Person node, e.g. unidentified visitors).
- Constraints: unique `Event.id`, `Routine.id`. Range indexes: `Event.ts`, `Event.subject_id`, `Event.kind`, `Routine.subject_id`.
- `label_prefix` (e.g. `WcTest`) namespaces labels and index names; tests use it, real data is never touched. `from_env()` uses the same `.env` as the graph store.
- `add_event` is an upsert by id. `forget_subject` also deletes that subject's routines. `delete_event(id)` is an extra (not in the protocol) used by retention.

SQLite mirrors this in two tables with indexes `(ts)`, `(subject_id, ts)`, `(kind, ts)`.

## Which store? Measured latency (MacBook Air M2, local, 10k events, 20 subjects)
| op (median / p95 ms) | Neo4j (bolt, Docker) | SQLite (in-memory page cache) |
|---|---|---|
| query last 50 | 28 / 87 | 0.4 / 0.8 |
| query subject, limit 50 | 7.5 / 22 | 0.6 / 1.0 |
| query 1 h window | 17 / 26 | 0.9 / 2.4 |
| kind+subject+1 day window | 23 / 122 | 0.3 / 0.4 |
| last_event(subject, kind) | 11 / 23 | 0.03 |
| count(subject) | 3.3 / 8 | 0.01 |
| add_event | 5.8 / 38 | 0.02 |

SQLite at 100k events: unchanged (0.4-0.9 ms medians; bulk load 1.4 s). Neo4j at 100k was not measured (shared 8 GB machine).
Recommendation: Neo4j latency is fine for the chat/agent path (tens of ms) but the recorder writes per event and the miner
scans months of history; for a high-volume timeline use `SqliteEventStore` (file on disk, one process) and keep the graph
in Neo4j. The `:ABOUT` edges are the only thing Neo4j gives for events; the agent can still join via `subject_id`.
Both pass the same contract suite, so switching is a one-line change.

## Recorder rules (see recorder.py docstring)
Per camera/track: `entered` after the track is confirmed (not NEW, >= `confirm_s`) and identified or `identity_wait_s` passed;
`left` after `leave_grace_s` of no live track (event ts = last seen; a return within the grace cancels it, no new `entered`);
`zone_changed` after the zone persists `zone_min_s`; `stationary` once per episode; `seen` every `seen_every_s`;
late identification re-keys the presence and emits one `seen` with `detail.identified_late`.
Subject: `person_id` when matched; otherwise `subject_id=None`, label "unknown person" (or the class label) and
`detail.cluster_id`/`detail.unidentified`. Hard cap `max_events_per_min` per subject (`stats()["dropped"]` counts drops;
this can drop a `left`, by design). Bounded: `max_tracks`, `max_subjects`, per-subject deque. Profiles come from
`RecorderConfig.profile_map` (label -> profiles; labels not in the map are ignored unless `"*"` is present).
Snapshot paths: from `ResolveResult.crop_path` (the recorder never writes files).

## Routines
Mined from `entered` and `activity_changed` events (first per local day per (subject, kind, activity)), clustered by
minute-of-day within +-`tolerance_min` (default 45); needs >= 3 distinct days and >= 60 % coverage of the observed days.
`weekday_at` when the habit holds on a fixed weekday subset (>= 2 hits per weekday). `presence_window` from first arrival
and last departure. Confidence = coverage x regularity x support. Not wrapped across midnight. Drifting or irregular
schedules are rejected. `source="user"` routines are never edited/deleted and suppress duplicate inferred ones.
`routine_status`: `on_time` (done today, or nothing owed yet), `due` (inside the window, not done), `overdue`
(window passed, not done), `not_applicable` (other weekday / outside presence window). Local timezone unless `tz` is given.
Run `apply_mining(store, store)` periodically (e.g. nightly); it is idempotent (stable ids) and drops stale inferred routines.

## Summaries
`build_day_summary(events, routines, day, profile)` is deterministic: first/last seen, visitors, unidentified visits, dwell
per zone (time between a presence event and the next zone change/left), activities, alerts, missed routines (window closed
with no matching event). `polish_summary(summary, ModelClient)` (task `summarize`) rewrites the prose, and is discarded in
favour of the deterministic text on error, empty output, or any number/time not in the facts.

## Privacy and retention
- Events hold identities and snapshot paths. Stores never touch files; `purge_before` / `forget_subject` / `RetentionPolicy.apply`
  RETURN paths and the caller deletes them (`delete_files(paths, root=...)` refuses paths outside `root`).
- `RetentionPolicy(days_by_profile, default_days)`: an event lives as long as the longest horizon among its profiles
  (medical can be kept longer than cctv); untagged events use `default_days`. A store without `delete_event` only
  gets the leading fully-expired run purged.
- `forget_subject` is the hard delete (events + routines). It does not touch the graph; use `graph_admin forget` for that.
- Snapshot files shared by several events are returned once; make sure no surviving event needs a file before deleting
  (the recorder gives each event the track's single crop, so a surviving event of the same track can reference it).
- CLI: `purge --days N [--profile-days medical=90 cctv=14] [--delete-files]`, `forget SUBJECT --yes [--delete-files]`.
