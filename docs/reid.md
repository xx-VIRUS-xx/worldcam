# Entity re-identification (animals and objects)

Problem: people are recognised again by their face embedding; animals and objects were not.
After the owner answered "that's my mug / that's Bruno", the next sighting asked again
(reproduced: question 1 answered, a new question opened 400 s later). Now an animal or
object is embedded, matched against known entity clusters of the SAME YOLO label, and only
asked about when it is genuinely unknown or uncertain.

## Pieces
| File | Role |
|---|---|
| `perception/reid.py` | `EntityEmbedder("auto"\|"cnn"\|"classical")`, `THRESHOLDS`, `--fetch`, `--bench` |
| `graph/api.py`, `store.py`, `schema.py` | `add_entity_view`, `match_entity`, `label_entity_cluster`, `entity_cluster_size`; Neo4j `EntityCluster` / `EntityView` nodes + `entity_centroid` vector index (dim 1280) |
| `ambiguity/entities.py` | `EntityObserver`: embed best crop -> match -> sighting / doubt signal |
| `ambiguity/triggers.py` | per-cluster dedup, `entity_min_sightings=2`, ignore cooldown by cluster, daily cap |

Setup (weights are never committed; `models/` is gitignored):
`python -m worldcam.perception.reid --fetch` downloads MobileNetV2 (14 MB, ONNX model zoo) and
exposes its pooled 1280-d features (needs the `onnx` package once, at fetch time only).
Without the file, `auto` falls back to the classical features.
Measure on your own crops: `python -m worldcam.perception.reid --bench <dir of <label>__<id>.jpg>`.

## Decision rules (mirror of faces, scoped by label)
Scores are plain cosine between the new embedding and cluster centroids, taken BEFORE the view is stored.

| Outcome | Condition | Effect |
|---|---|---|
| MATCHED | best labeled cluster >= `match_t` | Sighting of the KNOWN entity (person_id filled via the cluster). No doubt, no signal. View added until the cluster holds 30. |
| CONFLICT | two different labeled entities both >= `ask_t` and within 0.03 of each other | view starts its own cluster; `conflict` doubt (e.g. two similar white mugs) |
| LOW_CONFIDENCE | best labeled in [`ask_t`, `match_t`) | view joins/starts an UNLABELED cluster (never grows the labeled one silently); `low_confidence` doubt names the candidate |
| UNKNOWN | best unlabeled cluster >= `cluster_t` (or nothing) | joins it (or starts one); ONE `unknown_entity` doubt per cluster |

Doubt creation (`DoubtDetector`): needs `entity_min_sightings = 2` sightings OF THAT CLUSTER (a single
detection is noise; sightings are rate-limited, so the second is >= `cooldown_s` = 300 s later);
deduped on the cluster id while a doubt is open/asked/answered; "Ignore" blocks that cluster for 30 days;
at most `entity_max_new_doubts_per_day` = 5 entity doubts per rolling day (all kinds, hard cap).
Answering labels the cluster (`label_face_cluster` accepts entity cluster ids, so the worker's
existing tool works) and back-fills the cluster's sightings with the person.

Cooldown (`cooldown_s`, default 300): per track (a stationary track is not even re-embedded) and per
cluster (a re-created track of the same stationary object does not write another sighting).

Default labels: `cat, dog, backpack, handbag, suitcase, laptop`. `cup, bottle, book, cell phone` are
opt-in (`--entity-labels`): many look alike and the agent would ask about each.

## Measurements (MacBook Air M2, CPU)
Data: 51 crops (>= 70 px) detected by YOLO11n in COCO images (coco128 subset; cup 13, cat 9, dog 6, bird 8,
bottle 3, laptop 4, horse 2, ...). Each crop is one "instance". Same-instance pair = two synthetic views of it;
different-instance pairs = crops from different images. Not committed.
"Mild" = 0-3 % crop jitter + brightness 0.85-1.15 (a stationary object seen again by the same camera).
"Harsh" = 0-8 % jitter, scale 0.5-1.2, brightness 0.7-1.3 (+-20), 50 % Gaussian blur.
Rates below are over 4 seeds; "different" = different instance of the SAME label (the hard negative; 1344 pairs).

| Embedder | dim | ms/crop | mild: same >= match_t | mild: different >= ask_t / max | harsh: same >= match_t / in-between / < ask_t | harsh: different >= ask_t / max |
|---|---|---|---|---|---|---|
| CNN MobileNetV2 | 1280 | ~10 | 100 % | 2.5 % / 0.867 | 36 % / 36 % / 28 % | 0.8 % / 0.831 |
| classical (HSV + HOG + LBP) | 511 | ~0.8 | 72 % | 1.3 % / 0.907 | 26 % / 35 % / 39 % | 0.1 % / 0.904 |

AUC (same-instance score > different-instance same-label score), first run (harsh): classical 0.98, CNN 0.95;
second seed: classical 0.95, CNN 0.96; mild: classical 0.996, CNN 1.000. Concatenating both gave the best harsh AUC (0.977)
but costs both. Decision: **CNN is the default** (best margin on the realistic mild case: same p5 0.94 vs different max 0.86;
classical overlaps there: same p5 0.86 vs different max 0.90), classical is the zero-download fallback.
Different-label pairs are well below same-label (CNN max 0.67-0.68), so label scoping removes the easy errors for free.

Thresholds (CNN): `match_t = 0.90`, `ask_t = 0.80`, `cluster_t = 0.88`, conflict margin 0.03.
No different-instance pair reached 0.88 in any run (CNN), so false merges did not occur in this data; the cost is
that a heavily rescaled or blurred re-sighting often falls into "ask" (harsh: 64 % do not match outright).
That is the intended bias: ask rather than guess. Classical: 0.95 / 0.88 / 0.93 (poorly separated; use only as fallback).

## Honest limits
- Two products that look alike (two white mugs, two black laptops) are close in any appearance embedding. They land in
  [`ask_t`, `match_t`) or produce a CONFLICT, so the agent asks "is this A or B?" instead of guessing. That is a feature.
- Re-sighting from a very different pose (a dog from the back vs the front, a bag open vs closed) can drop below
  `ask_t`; it then becomes a new cluster and one more question. The owner can answer "same as X", which labels the new
  cluster to the existing entity (the entity then has several clusters, matched per person).
- The data is 51 crops with synthetic transforms, not video of the same object. Thresholds are a starting point;
  re-run `--bench` on real captures and adjust `THRESHOLDS`. Real dog/cat identity needs more than this (fur pattern helps, indoor lighting hurts).
- YOLO may flip `handbag`/`backpack` for the same bag: the label scope then splits it into two clusters (one extra question).
- `match_entity` over-fetches 200 nearest from the global vector index and filters by label; with very many clusters
  of other labels this could hide a candidate (not an issue below a few thousand clusters).
- Embedding dimension is fixed per database (index `entity_centroid`, default 1280). Switching to the classical embedder
  (511) needs `Neo4jGraphStore(entity_dim=511)` and a fresh index; mixing methods in one database is not supported.

## Contract additions (graph/api.py, additive)
`add_entity_view(label, embedding, *, quality, ts, snapshot_path, cluster_id=None, new_cluster=False) -> (view_id, cluster_id)`,
`match_entity(label, embedding, k=3) -> [FaceMatch]` (plain cosine, same label only),
`label_entity_cluster(cluster_id, person_id)`, `entity_cluster_size(cluster_id)`.
`label_face_cluster` / `cluster_size` also accept entity cluster ids. `forget_person` removes the person's entity views and
clusters and returns their snapshot paths; `merge_people` moves entity clusters; `person_context()["clusters"]` lists them.
`InMemoryGraphStore(entity_cluster_threshold=0.88)`, `Neo4jGraphStore(entity_dim=1280, entity_cluster_threshold=0.88)`.
