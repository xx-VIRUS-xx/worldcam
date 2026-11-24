# Graph store (Neo4j)

The production `GraphStore` is `worldcam.graph.store.Neo4jGraphStore`, running against
Neo4j **5.26.33 Community** (LTS) in Docker through colima. The schema follows
`docs/AMBIGUITY_ENGINE.md` ("Graph schema"). Everything outside `graph/` codes against
`graph/api.py`.

## Running it
```bash
cp .env.example .env            # then set NEO4J_PASSWORD (random; never commit .env)
docker compose up -d            # container: worldcam-neo4j
python -m worldcam.cli.graph_admin init     # constraints + vector indexes (idempotent)
python -m worldcam.cli.graph_admin status
```
- Ports are bound to **127.0.0.1 only**: bolt `7687`, http `7474`.
- Memory: heap 512m (initial = max), pagecache 256m, total tx memory 256m.
- Telemetry is off: `NEO4J_dbms_usage__report_enabled=false`. Verified with
  `SHOW SETTINGS`, which reports `dbms.usage_report.enabled = false`.
- Data lives in the named volume `worldcam_worldcam-neo4j-data`. The compose project is
  pinned to `name: worldcam`, so every checkout and worktree shares one container and
  one volume.
- `.env` is found by walking up from the current directory, so a git worktree under
  `.claude/worktrees/` uses the repo root's `.env`. For compose, run it from the repo
  root or pass `--env-file`. Real environment variables override `.env`.

### Browsing the graph
Open http://127.0.0.1:7474. Connect URL `bolt://127.0.0.1:7687`, user `neo4j`, and the
password from `.env`. Some useful queries:
```cypher
MATCH (p:Person) RETURN p ORDER BY p.last_seen DESC LIMIT 25
MATCH (p:Person {name: 'Rahul'})<-[:BELONGS_TO]-(c:FaceCluster)<-[:IN]-(f:Face) RETURN p, c, f LIMIT 50
MATCH (d:Doubt {status: 'open'})-[:ABOUT]->(x) RETURN d, x
MATCH (a:Person)-[r:RELATED]->(b:Person) RETURN a, r, b
```

## Python
```python
from worldcam.graph.store import Neo4jGraphStore
store = Neo4jGraphStore.from_env()   # same as Neo4jGraphStore(): NEO4J_* from env / .env
```
Constructor options: `cluster_threshold=0.45`, `label_prefix=""`, `dim=512`,
`init_schema=True`, `read_timeout_s=10`, `driver=` (reuse a driver).
There are also superset helpers that are not in the protocol: `list_people`,
`list_clusters`, `stats`, `export`, `wipe`, `close` and context-manager use.

## Schema
| Label | Unique | Other indexes |
|---|---|---|
| Person | id | name |
| FaceCluster | id | person_id; **vector `cluster_centroid`** (centroid, cosine, 512) |
| Face | id | cluster_id; **vector `face_embedding`** (embedding, cosine, 512) |
| Sighting | id | person_id, cluster_id, ts |
| Fact | id | person_id |
| Doubt | id | status, message_ref |

Edges: `(Face)-[:IN]->(FaceCluster)-[:BELONGS_TO]->(Person)`,
`(Sighting)-[:OF]->(Person|FaceCluster)`, `(Person)-[:HAS_FACT]->(Fact)`,
`(Person)-[:RELATED {relation, source, ts}]->(Person)`, `(Doubt)-[:ABOUT]->(Person|FaceCluster|Sighting)`.
Foreign ids (`person_id`, `cluster_id`, `about_ids`) are also stored as properties. The
properties give an exact round trip with `InMemoryGraphStore`; the edges make the graph
browsable. `Doubt.detail` is stored as a JSON string, and nested dicts and lists round-trip.
`Doubt.message_ref` is stored as given, so pass a `str` consistently.

Implementation notes:
- `match_face` takes candidates from the `cluster_centroid` HNSW index (k' = max(k, 10)).
  It then re-scores them exactly with `vector.similarity.cosine` and returns plain cosine
  (Neo4j reports (1+cos)/2). This matters because the raw index score was off by up to
  about 1e-3.
- `add_face` does the match, the optional cluster creation, the running-mean centroid
  update and the Face insert in one write transaction.
- `graph_read` makes a client-side check first. It rejects write and admin keywords
  anywhere in the text, `;`, and any `CALL` except a small allowlist of read-only `db.*`
  procedures. It then runs in a READ-access session with `execute_read`, a 10 s timeout
  and at most `limit` rows (capped at 1000). Embedding and centroid vectors are elided
  from its output. A test bypasses the client check to confirm that the server still
  refuses the write.
- `forget_person` hard-deletes the person, their facts, relations, the person's clusters
  and those clusters' faces, and sightings of the person or those clusters. It returns the
  sorted, de-duplicated snapshot paths. `graph_admin forget` also deletes those files.
  Doubts are not deleted (see "Protocol gaps").

## Tests
`tests/contract_graph.py` is one suite, parametrised over `[memory, neo4j]` and collected
by `tests/test_graph_contract.py`. The Neo4j case is skipped if bolt is unreachable or
`NEO4J_PASSWORD` is unset. It never touches real data. Community has a single user
database, so the suite uses the label prefix `WcTest` (`:WcTestPerson`, ... with their
own `wctest_*` indexes), wipes only those labels between tests and drops its indexes
at the end. `tests/test_graph_config.py` covers `.env` parsing, label prefixing and
the `graph_read` guard without needing Neo4j.

```
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
128 passed, 1 skipped, 3 xfailed      (Neo4j up; graph files alone: 63 passed, 1 skipped, 3 xfailed)
NEO4J_URI=bolt://127.0.0.1:1 ... tests/test_graph_contract.py -> 19 passed, 25 skipped, 3 xfailed
```
The 3 xfails are strict xfails on `InMemoryGraphStore` only (see below).

## Measurements (M2 MacBook Air, 8 GB, colima VM 8 GB / 4 CPU, Neo4j 5.26.33)
The run used random unit 512-d vectors in an isolated label prefix. Each random face forms
its own cluster, so N faces means N clusters. `match_face(k=3)` was timed over 300 queries
after 20 warm-up queries, end to end from Python (driver session, vector query, exact
re-score).

| Clusters | match_face p50 | p95 | max |
|---|---|---|---|
| 1,000 | 10.8 ms | 25.9 ms | 83 ms |
| 10,000 | 18.1 ms | 44.4 ms | 249 ms |

- `add_face` (match, create cluster and face) for the first 1,000 faces: 56 ms per face on
  average. That is fine for faces arriving at a few per second; bulk imports should use
  UNWIND.
- HNSW recall at 10k: top-1 matched an exact numpy scan for 99 of 100 noisy queries.
- `docker stats --no-stream` for worldcam-neo4j: **1.00 GiB idle** (empty graph) →
  1.05 GiB at 1k → **1.20 GiB at 10k** faces and clusters → 1.24 GiB after cleanup
  (the JVM does not hand memory back). CPU is below 1% when idle.
- The benchmark data and its indexes were deleted afterwards.

## Admin CLI
```
python -m worldcam.cli.graph_admin init
python -m worldcam.cli.graph_admin status
python -m worldcam.cli.graph_admin people [--limit N]
python -m worldcam.cli.graph_admin show <person id or name>
python -m worldcam.cli.graph_admin doubts [--status open|asked|answered|resolved|dismissed]
python -m worldcam.cli.graph_admin forget <person_id> [--yes]   # also deletes snapshot files
python -m worldcam.cli.graph_admin export graph.json
python -m worldcam.cli.graph_admin wipe --yes-really             # all worldcam nodes; schema kept
```

## InMemoryGraphStore bugs (reported, not fixed here: api.py is owned elsewhere)
1. `merge_people(a, a)` deletes `a`. Neo4j raises `ValueError`.
2. `merge_people` rewrites relations without de-duplicating them. That leaves duplicate
   `(keep, r, x)` tuples and self-loops: a `keep–drop` relation becomes `keep→keep`.
   Neo4j de-duplicates them and drops the self-loops.
3. `merge_people` leaves `Doubt.about_ids` pointing at the deleted `drop_id`. Neo4j
   rewrites the ids to `keep_id`.
4. Minor: `update_doubt(id, updated=...)` raises a confusing duplicate-kwarg `TypeError`,
   and `update_doubt(id, id=...)` silently corrupts the dict key. Neo4j rejects both with
   `TypeError`.

## Protocol gaps (suggestions, not implemented in api.py)
- `forget_person` does not cover doubts. A Doubt about a forgotten person keeps its
  question text, its `snapshot_paths` (crops of that person) and its ids. A privacy
  reading of "hard delete" suggests also deleting (or scrubbing) doubts whose `about_ids`
  touch the person, their clusters or their sightings, and returning their `snapshot_paths`.
- There is no `list_people()` in the protocol: `search_people("")` skips unnamed people.
  The store has `list_people` as a superset method.
- `message_ref` is typed `str`, but Telegram ids are ints. Callers should always `str()` them.
