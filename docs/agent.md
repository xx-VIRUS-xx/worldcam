# Agent loop (implementation notes)

Contract: `agent/api.py`. Product intent: `VISION.md`. Everything proactive spends one **attention budget per user**.

| File | Role |
|---|---|
| `agent/profiles.py` + `agent/profiles/*.json` | load/validate/merge profiles (data, not code), alert-rule matching |
| `agent/budget.py` | `InMemoryAttentionBudget`, `SqliteAttentionBudget` (caps, quiet hours, ignore back-off) |
| `agent/dispatcher.py` | the only way the agent reaches users (budget, routing, `view_images`) |
| `agent/uncertainty.py` | `UncertaintyManager`: rank doubts and gaps, pick WHO to ask, decide if an answer settles |
| `agent/loop.py` | `AutonomousAgent`: bounded trigger queue, scheduler, tasks |
| `ambiguity/engine.py` | the doubt engine, now with a multi-user mode behind the same surface |

## Loop

```
 event (BrainEvent) ─┐                                   ┌─ curiosity ─► create ONE doubt (metadata_gap)
 doubt  ────────────┤                                   ├─ look_around ► look_now(cam) -> describe_scene(cam, path)
 chat (user_id) ────┼─► bounded Queue ─► AutonomousAgent ├─ consolidate ► miner/summarizer hooks; merge PROPOSALS (conflict doubts)
 scheduler ─────────┤   (full = drop)      (one thread,  ├─ checkin ────► Dispatcher.send(prompt) per profile
  fixed + jittered  ┘                       crash-isolated) └─ alert_review ► rules vs events -> Dispatcher.alert
                                                                                  │
 AmbiguityEngine.tick ◄── doubts ────────────────────────────────────────────────┘ (curiosity/consolidate)
   ask worker -> outbox -> UncertaintyManager.ask_target -> Dispatcher.send -> user
 user reply -> engine.handle_user_reply -> UncertaintyManager.record_answer -> answered (+confirmed_by) | review doubt
```

The loop never writes to a user's channel itself. It creates doubts (which the engine asks) or calls the Dispatcher.

## Triggers
| Trigger kind | Source | What happens |
|---|---|---|
| `event` | new `BrainEvent` (`payload.event` or `event_id`) | profile alert rules evaluated for that event (dedupe per rule+event, per-rule cooldown) |
| `doubt` | a doubt changed | `engine_tick()` callback |
| `chat` | a user spoke (`payload.user_id`) | marks them available (check-ins skip them for `chat_quiet_s`); runs curiosity preferring that user |
| `schedule` | scheduler, fixed interval | `payload.task` in `curiosity look_around consolidate checkin alert_review` |
| `random` | scheduler, interval with jitter | same, at `every * (1 +- jitter)` |

Defaults (`AgentConfig.schedule`): curiosity 1 h +-30 %, look_around 30 min +-30 %, consolidate 6 h +-20 %, alert_review 5 min +-10 %, plus each profile check-in. Tests inject `clock` and `rng` and call `step(now)` instead of `start()`.

Failure behaviour: every task runs under `_guard` (logged, counted in `stats["errors"]`, last 50 in `errors`); a full queue returns `False` from `submit` and counts `stats["dropped"]`; `stop()` joins the thread.

## Attention budget
Per user, by role (hour/day): owner 6/30, household 3/12, caregiver 4/16, guest 1/3; `profile_limits` can tighten per profile. Quiet hours default 23:00-08:00 and can be set per user (`set_quiet_hours`). **Priority 3** bypasses quiet hours, the normal caps and ignore cooldowns, but never the hard ceiling (2x the daily cap). Ignore back-off: every consecutive `record(user, "ignored")` (Ignore tap, or no answer after `unanswered_after_h`=12 h) doubles the cooldown from 2 h up to 48 h; `record(user, "answered")` resets it. `remaining(user)` reports hour/day/hard_day/quiet/cooldown. The engine's old fixed 6/hour is used only in legacy mode.

## Who is asked
`UncertaintyManager.candidates(doubt)`: users who hold the topic's ask permission (identity/relations: `receive_companion`; medical: `receive_medical_alerts`), satisfy `can_confirm(user, topic)` and pass `budget.allow`. Order: the person the question is about (own facts) -> `prefer_user` (the user who just chatted) -> profile subscribers -> role (household/caregiver before owner) -> most remaining daily budget. If nobody qualifies the doubt waits. The owner is a candidate for everything, so with no household users it is the fallback.
Ranking is value / cost: doubt kind, how many sightings an answer would explain, profile goals, staleness, +3 for medical, over `1 + 0.5 * rounds`.

## Answers
`engine.handle_user_reply(user, InboundMessage)` matches on `reply_to_ref` and `detail.asked_to`.
- Permitted user: doubt becomes `answered`, `detail.confirmed_by = {user_id, role, ts}`.
- Not permitted (e.g. a guest on an identity question): **not settled**. The statement is kept in `detail.unconfirmed_statements` and as an `inferred` fact (confidence 0.3, key `unconfirmed_<topic>`); the doubt is dismissed (`unconfirmed answer`) and a `conflict` review doubt is created for someone who can confirm (the speaker is excluded).
- Ignore tap: doubt dismissed, budget back-off.
Worker prompts should read `detail.confirmed_by` when writing facts (the owner-source rule is enforced by who may settle, not by the tool).

## Knowledge gaps (curiosity)
Unlabeled people seen >= 3 times; named people seen >= 5 times with no relation; owner-sourced facts older than 180 days; inferred routines with confidence 0.35-0.7. A gap already asked about (any open doubt, or any within 14 days) is skipped; at most 2 curiosity doubts are open at once.

## Profiles
JSON with the `Profile` fields. Alert rule: `{"id", "when", "to_permission", "priority" 0-3, "cooldown_min", "text"}`; `when` keys: `event_kind subject zone activity label min_confidence night hours` (event rules) and `routine_state no_movement_minutes` (state rules). Text placeholders: `{zone} {time} {subject} {activity} {routine} {minutes}`. Check-in: `{"id", "every_hours", "jitter", "hours": [start,end), "audience": "permission"|"self", "to_permission", "quiet", "prompt"}` (`quiet` = silent message; `self` = users linked to a person and subscribed to the profile).

Combining: `merge_profiles` unions goals, vocabularies, rules and check-ins; the strictest privacy wins (`send_images_to_cloud` only if every profile allows it; shortest retention). Alert recipients = holders of `to_permission` subscribed to the rule's profile plus owners (everyone holding it if nobody subscribed). Events tagged with `profiles` only trigger rules of those profiles.

Add a profile: write `agent/profiles/<name>.json` (or any path ending in `.json`), then `load_profiles(["cctv", "my.json"])`; `validate_profile_dict` lists problems. Medical profiles are wellness aids: wording says "may", never diagnoses.
