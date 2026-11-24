# Identity: faces + resolver

Contract: `docs/AMBIGUITY_ENGINE.md`. Code: `src/worldcam/identity/`.

```
track.best.crop --FaceAnalyzer.analyze--> [FaceObservation] --IdentityResolver.resolve--> ResolveResult
                                                                 |  store.match_face (before adding)
                                                                 |  store.add_face, store.add_sighting
                                                                 v
                                            Identity + Decision (+ cluster/candidate ids) -> ambiguity engine
```

## FaceAnalyzer (`identity/faces.py`)
- InsightFace `buffalo_l`, only `detection` (SCRFD-10G, 5 landmarks) and `recognition` (ArcFace w600k_r50, 512-d). Loaded lazily on the first call; `warmup()` loads it eagerly. Weights live in `~/.insightface` (not in git; non-commercial licence).
- `analyze(crop_bgr, track, ts=None)`: `crop_bgr=None` uses `track.best.crop`. Returns faces sorted by quality, with:
  - `embedding`: 512 floats, L2-normalized (re-normalized in float64).
  - `box`: normalized full-frame box. The crop is assumed to cover `track.best.box` padded by `crop_pad` (default 0.1) and clamped, exactly what `geometry.crop_norm` / `DualGrabber.crop` produce.
  - `quality` = det × size × sharpness × frontal, each 0..1:
    - det: `(det_score - 0.5) / 0.3`, clipped (0.8+ is full).
    - size: `min_side_px / 112` (ArcFace input size), clipped.
    - sharpness: Laplacian std / intensity std on the face resized to 112 px, mapped `(r - 0.05) / 0.2`. It doesn't depend on face size or exposure: a dark but sharp face scores like a bright one.
    - frontal: from the 5 landmarks after levelling the eyes (roll is free, since the aligner undoes it). It penalises yaw (nose off the midline, over 0.15 eye distances) and pitch.
  - Faces with a min side under 32 px or quality under 0.1 are dropped.
  - `crop_path`: face plus 30% context, saved as JPEG to `$WORLDCAM_SNAPSHOT_DIR` (default `captures/snapshots`), named `face_t<track>_<ms>_<i>.jpg`. Turn it off with `save_crops=False`. A failed save logs and leaves `crop_path=None`.
- Provider: **CPU by default** (see the measurements). Override with `providers="coreml"` or `WORLDCAM_FACE_PROVIDER=coreml`. If CoreML fails to load, the analyzer falls back to CPU.

## IdentityResolver (`identity/resolver.py`)
`IdentityResolver(store, match_t=0.45, ask_t=0.35, conflict_margin=0.05, reverify_s=600, upgrade_margin=0.2, camera="default")`.

`resolve(track, faces) -> ResolveResult` uses the best-quality face. Scores are cosine similarities to cluster centroids, computed **before** the face is added, so a face can't vote for itself.

| Decision | Rule | Stored as |
|---|---|---|
| `CONFLICT` | Two different labeled persons are both ≥ ask_t and within `conflict_margin`, **or** a re-verification of a track that was MATCHED to A now matches B (`detail.reason` is `two_persons_close` or `track_identity_changed`, with `previous_person_id` / `previous_crop_path`) | store chooses the cluster |
| `MATCHED` | Best labeled cluster ≥ match_t | forced into that person's cluster; the sighting has `person_id` |
| `UNKNOWN_CLUSTER` | Best cluster ≥ ask_t is unlabeled | store chooses the cluster. `cluster_id`/`cluster_size` are where the face landed, which is what `unknown_face` (N=3) counts |
| `LOW_CONFIDENCE` | Best labeled cluster is in [ask_t, match_t) | store chooses; with cluster_threshold=0.45 this is a new unlabeled cluster, so unsure faces never pollute a labeled cluster. When the owner says "yes", label that cluster |
| `NEW_FACE` | Nothing ≥ ask_t | a new cluster (size 1) |
| `NO_FACE` | No usable face | nothing stored; not cached for good |

`Identity.name` is the person's name (or person id if unnamed) only for MATCHED, and `"unknown"` otherwise. `Identity.conf` is the cosine of the top candidate. `candidates` holds the best cluster per labeled person plus the unlabeled clusters, sorted by score, each with `person_id`, `cluster_id`, `score` and `name`. Also returned: `face_id`, `sighting_id`, `crop_path`, `quality`, `ts` and `cached`.

**Per-track cache.** Each track is resolved once. Later calls return the cached result (`cached=True`) and store nothing more, until one of these happens:
- `reverify_s` (600 s) of track time has passed.
- The cached decision was NO_FACE and a face appears.
- The cached decision was not MATCHED and a face with quality at least `+upgrade_margin` higher arrives.

The cache is LRU-bounded (2048). The pipeline should call `forget(track_id)` when a track ends. The engine should call `invalidate(cluster_id=…)` or `invalidate(person_id=…)` after labeling or merging, so tracks that are still live pick up the new name.

## Measurements (2026-11-24, MacBook Air M2 8 GB)
**The machine was heavily loaded while measuring** (load average 22-27: four other agents, a VM and headless browsers). Absolute latencies are inflated and noisy. Re-run on an idle machine: `PYTHONPATH=src python -m worldcam.identity.bench`.

Latency, from 4 paired runs (`[provider]`, medians in ms over 30 calls; a person crop is about 170×330 px with one ~100 px face):

| run | provider | load | first call | person crop detect+embed (p90) | embed only, 112 px (p90) |
|---|---|---|---|---|---|
| 1 | cpu | 2.7 s | 5.3 s | 896 (1296) | 429 (521) |
| 2 | coreml | 22.9 s | 1.1 s | 344 (395) | 113 (128) |
| 3 | cpu | 1.1 s | 1.3 s | 487 (647) | 123 (165) |
| 4 | coreml | 24.8 s | 2.5 s | 1167 (1443) | 400 (491) |
| bench | cpu / coreml | 2.2 / 35.0 s | 7.3 / 38.6 s (first CoreML compile) | 1013 / 465 | 354 / 140 |

- CPU vs CoreML embeddings: cosine **1.00000** (minimum over the 6 faces). CoreML is numerically correct.
- Speed: CoreML won 2 of 3 comparisons and lost 1. That is not reliably faster, and it loads 10-20× slower. **Decision: CPU default.** Identity runs once per track, so 0.5-1 s per crop is acceptable. Revisit on an idle machine: if CoreML is consistently faster there, switch with the env var and call `warmup()` at startup.

Similarity, with the CPU provider. Six identities come from sample `t1`; a person-like crop is made around each face, then 13 synthetic transforms are applied. The aligned `Tom_Hanks` sample adds a 7th identity with 3 variants.
```
same-person: n= 549 min 0.648 p5 0.762 median 0.930 p95 0.984 max 0.998
diff-person: n=3192 min -0.097 p5 -0.054 median 0.039 p95 0.165 max 0.267
  t=0.35 and t=0.45: same >= t 100.0%   diff >= t 0.0%

per transform: cosine to own 'orig' (min / median), quality (median)
  orig            ref            q 0.51     flip           0.926/0.948 q 0.38
  tight_crop      0.957/0.975 q 0.42        scale_2x       0.982/0.993 q 0.57
  scale_0.6       0.981/0.988 q 0.31        scale_0.4      0.929/0.959 q 0.23  (~42 px face)
  blur_k5         0.975/0.983 q 0.31        blur_k9        0.881/0.935 q 0.18
  dark_0.4        0.959/0.968 q 0.54        bright_1.6     0.711/0.823 q 0.52
  jpeg_q15        0.793/0.848 q 0.59        rot_15         0.944/0.972 q 0.38
  noise_12        0.896/0.934 q 0.57        lowres_0.3_up  0.835/0.908 q 0.21
```

### What this does and doesn't show
This is a **sanity check, not a FAR/FRR evaluation**. There are 7 identities and every "same-person" pair is the *same photo* under a transform, with the same pose, expression, lighting, day and camera. The pairs therefore show:
- The pipeline is wired correctly: no detector misses, unit embeddings, boxes mapped correctly, and CoreML equals CPU.
- Different people sit far below both thresholds here (max 0.27, so ask_t=0.35 has a margin of 0.08).
- Overexposure (`bright_1.6`, down to 0.71) and heavy JPEG (q15, 0.79) hurt the embedding most, **but quality doesn't catch them**: those faces score q≈0.5-0.6. Blur and low resolution lower both quality and similarity, as intended.

They say **nothing about real false rejects**. Real same-person pairs across days, angles, glasses, beards and lighting score much lower than same-photo pairs. This is the case that matters, and it can't be measured from these samples.

### Threshold recommendation
- **Keep match_t=0.45 and ask_t=0.35 for now.** Nothing here argues for changing them: there were no impostor scores above 0.27, and every transformed same-person score was ≥ 0.65. Moving them on 7 identities would be overfitting.
- **Keep the store's `cluster_threshold` equal to match_t (0.45).** Then LOW_CONFIDENCE faces start their own cluster, and the owner's "yes" labels it without contaminating a labeled cluster.
- For `unknown_face`, count only sightings with `quality ≥ 0.3` toward N=3. On these samples, q<0.3 was blur k9 (0.18), low-res upscale (0.21) and 42 px faces (0.23).
- Expect the real risk to be **false rejects**, which show up as cluster fragmentation: one visitor split into several unknown clusters that each stay below N=3. If labeled data shows genuine pairs often scoring 0.35-0.45, lower cluster_threshold and match_t together (for example to 0.40) rather than lowering ask_t.

### Data the owner needs to collect for a real FAR/FRR
Recorded from **the actual camera(s)**, at their mounting height, resolution and stream (mainstream crops):
1. **Genuine pairs.** At least 5-10 people (household plus willing visitors), each seen on **at least 5 different days**, with at least 20 tracks per person. Cover morning, evening, artificial light and night/IR, and include glasses on and off, hats, masks, and walking toward and across the camera. Label each track with a person id; a CSV of `clip, track_id, person_id` is enough.
2. **Impostors.** As many *other* distinct faces as possible: visitors, couriers, and TV/phone screens if they are in view. Impostor pairs grow quadratically, but people who look alike (family members) matter most, so include them.
3. **Protocol.** Enroll each person from the first 1-2 days, then test on later days only. Report FRR at match_t, and FAR (rate of impostors ≥ match_t, and ≥ ask_t) with confidence intervals. With 1000 impostor pairs, a FAR of 0 still only bounds the true FAR at about 0.3% (95%). Report night/IR separately (ROADMAP risk). Also report cluster purity and fragmentation from running `add_face` over the stream, because that is what drives `unknown_face`.
4. **Consent.** Everyone recorded must know about it and agree; delete the set after the evaluation (biometric data, see ROADMAP).
