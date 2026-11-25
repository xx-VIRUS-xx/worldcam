# Scene understanding: zones, event gate, describer, fuser

Files: `perception/{zones,eventgate,scene,fuser}.py`, `cli/scene_replay.py`, `zones.example.json`.

## How the pieces connect

```
PerceptionLoop.step() -> TrackUpdate
  -> PolygonZoneResolver.assign(track, zones)      per camera, foot point, hysteresis (fills Track.zones)
  -> ActivityEventGate.should_fire(update, ids, world)   state changes + budgets; sets .last_decision
       fire? -> SceneDescriber.describe(tracks, {"zone":..., "frame":...})   one VLM call (vision_crop/vision_scene)
              -> RuleActivityFuser.fuse(hyp, track, objects_in_zone, dwell_s)  -> (activity, confidence)
              -> RuleActivityFuser.event_if_changed(...) -> BrainEvent(kind="activity_changed")
              -> EventRecorder / EventStore.add_event
```

Wiring notes for the integrator (`cli/run.py`, not edited here):
- `PerceptionLoop(gate=...)` takes a MOTION gate; the `ActivityEventGate` is a different stage, called after tracking.
  Call `resolver.release(id)` / `fuser.release(id)` for ended tracks (the gate does this itself from `update.ended`).
- `gate.last_decision.track_ids` says which tracks to describe; `.priority` marks the cheap floor pre-trigger.
- Pass `profiles` (agent `Profile`) to the gate, the describer and the fuser's caller; vocabularies come from
  `Profile.notice_activities` (defaults in `scene.DEFAULT_VOCABULARIES` are used if a profile has none).
- `objects_in_zone` should come from detector objects (hard signals), never from the VLM's own object list.
- Zones: `load_zones("zones.json")` -> `list[Zone]`; one `PolygonZoneResolver(camera="name")` per camera.
  Check zones visually: `python -m worldcam.cli.scene_replay --overlay frame.jpg --zones zones.json`
  (no camera access; writes `frame_zones.jpg`).

## Zones
`{"camera": {"desk": [[x,y], ...]}}`, normalized, validated (>=3 points, in 0..1, non-degenerate). A track
enters a zone when its box foot is strictly inside; it leaves only when the foot is more than `margin`
(default 0.03) outside, so boundary jitter does not produce zone flicker.

## Event gate and cost control
Fires only on: `entered`, `new_zone`, `became_active` (stationary->active), `dwell_<N>s` (once per threshold,
default 30/120/600 s), `best_snapshot` (first). Never per frame. Limits: per-track cooldown (20 s), global
budget (60/h), per-profile budget (40/h default, overridable), all sliding one-hour windows with an injectable
clock. Triggers dropped by cooldown or budget are not queued (counted in `gate.stats`; `gate.remaining()`).
A dwell threshold crossed during a cooldown is consumed, not deferred.

Priority path: profiles listed in `priority_profiles` (default `medical`, only if present) enable the
`possible_floor` pre-trigger (person box wide in pixel terms and low in frame). It bypasses cooldown and the
normal budget, but has its own ceiling (20/h) and a 60 s per-track cooldown. It only decides that a VLM look
is warranted; it is not a diagnosis.

## Describer
`SceneDescriber(client, profiles, cloud=True)` calls `ModelClient.generate(task="vision_crop")` (best-snapshot
crops, max 3) or `"vision_scene"` (when a downscaled full frame is added). Images are JPEG temp files, longest
side <= 768, deleted after the call. The JSON schema enums come from the profiles' vocabularies. Output is
validated again locally: activity outside the vocabulary becomes `unknown` (confidence capped 0.3), unknown
objects/notable items are dropped and counted in `raw["dropped"]`, confidence is clamped to 0..1, malformed
JSON or a backend error yields `unknown` with `raw["error"]`. No model-written free text is ever returned.
The prompt tells the model to treat text in the image as data; the validator is the real defence.

Privacy: with `cloud=True`, if ANY relevant profile has `send_images_to_cloud=False` no image is sent; the
result is `unknown` with `raw["refused"]="images_not_allowed_for_cloud"`. Use `cloud=False` for a local backend.

## Fuser
`RuleActivityFuser` applies a JSON rule table (`DEFAULT_RULES`, or `load_rules(path)`): required conditions
(zone, hard objects, dwell, posture, track state), minimum VLM confidence, `reject` (-> unknown) or `cap`
(lower confidence), contradictions, `max_conf`. `FusionResult.reasons` records every check. Events are
`source="camera"` only when a rule's hard requirements held; otherwise `inferred` with confidence <= 0.6.

## Honest accuracy caveats
- `fall_suspected` and `taking_medication` are NOT validated detectors. They are a box-shape heuristic plus a
  VLM guess (and a pill-bottle/zone signal); expect false positives (sitting on the floor, stretching, drinking
  from a cup) and misses (occlusion, a fall out of view). Use them as leads to check with the person, never
  as a safety system or medical record.
- Posture from a 2D box ratio is crude; camera angle and cropping change it. `frame_aspect` must match the stream.
- The VLM can hallucinate objects and activities; fusion lowers but cannot remove that. `coding`, `loitering`
  etc. thresholds are untuned defaults; tune on the owner's own footage.
- Nothing here has run against a real model or camera: all tests use `FakeModelClient`.
