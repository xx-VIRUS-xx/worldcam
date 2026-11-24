"""Neo4jGraphStore: the production GraphStore (see graph/api.py for the contract).

Semantics follow InMemoryGraphStore; tests/contract_graph.py runs the same
suite against both. Notes:
- Parameterised Cypher only. Labels are the one thing formatted into query
  text, and only from the validated `label_prefix` (tests use it to isolate
  their data, since Neo4j Community has a single user database).
- Writes go through `session.execute_write` (one transaction per call).
- Relationship-ish ids (`person_id`, `cluster_id`) are kept as properties AND
  as edges (BELONGS_TO, OF, HAS_FACT, ABOUT, IN): properties give the exact
  InMemory round-trip, edges make the graph browsable.
- `match_face` takes candidates from the `cluster_centroid` vector index (HNSW,
  approximate; its score is off by up to ~1e-3), re-scores them exactly with
  vector.similarity.cosine and converts Neo4j's (1+cos)/2 back to plain cosine.
- `graph_read` runs in a READ-access session/transaction with a timeout and a
  row cap, after a client-side check that rejects write/admin clauses.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import fields as dc_fields, replace
from itertools import islice
from typing import Any, Optional, Sequence

from .api import (DOUBT_KINDS, DOUBT_STATUSES, FACT_SOURCES, PERSON_KINDS, Doubt, Fact,
                  FaceMatch, Person, Sighting, new_id)
from .schema import EMBED_DIM, LABELS, apply_prefix, check_prefix, ensure_schema, index_name

_DOUBT_FIELDS = {f.name for f in dc_fields(Doubt)}
_VECTOR_KEYS = ("embedding", "centroid")
GRAPH_READ_MAX_ROWS = 1000

# Client-side guard for graph_read. Deliberately strict: matches anywhere in
# the text (strings and comments included). The server-side READ access mode
# is the real enforcement; this just refuses obvious writes before sending.
_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD|ALTER|GRANT|DENY|REVOKE|"
    r"RENAME|START|STOP|TERMINATE|ENABLE|SHOW|USE|TRANSACTIONS|DEALLOCATE|REALLOCATE)\b",
    re.IGNORECASE)
_CALL_RE = re.compile(r"\bCALL\s*([A-Za-z_][\w.]*)?", re.IGNORECASE)
_READ_PROCS = {"db.index.vector.querynodes", "db.labels", "db.relationshiptypes",
               "db.propertykeys", "db.schema.visualization", "db.schema.nodetypeproperties",
               "db.schema.reltypeproperties"}


def check_read_only(cypher: str) -> None:
    """Raise ValueError if `cypher` contains anything but read clauses."""
    if ";" in cypher:
        raise ValueError("graph_read: one statement only")
    m = _WRITE_RE.search(cypher)
    if m:
        raise ValueError(f"graph_read: write/admin clause {m.group(1).upper()!r} not allowed")
    for m in _CALL_RE.finditer(cypher):
        proc = m.group(1)
        if proc and proc.lower() not in _READ_PROCS:
            raise ValueError(f"graph_read: procedure {proc!r} not allowed")


def _dedup(items) -> list:
    return list(dict.fromkeys(items))


class Neo4jGraphStore:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, *, database: Optional[str] = None,
                 cluster_threshold: float = 0.45, label_prefix: str = "", dim: int = EMBED_DIM,
                 init_schema: bool = True, read_timeout_s: float = 10.0, driver=None,
                 connection_timeout_s: float = 5.0):
        from neo4j import GraphDatabase

        from .config import Neo4jConfig
        cfg = Neo4jConfig.from_env() if driver is None else None
        self.uri = uri or (cfg.uri if cfg else None)
        self.database = database or (cfg.database if cfg else "neo4j")
        self.cluster_threshold = cluster_threshold
        self.prefix = check_prefix(label_prefix)
        self.dim = dim
        self.read_timeout_s = read_timeout_s
        self._owns_driver = driver is None
        if driver is None:
            driver = GraphDatabase.driver(
                self.uri, auth=(user or cfg.user, password if password is not None else cfg.password),
                connection_timeout=connection_timeout_s)
        self._driver = driver
        self._centroid_index = index_name("cluster_centroid", self.prefix)
        self._qcache: dict[str, str] = {}
        if init_schema:
            ensure_schema(self._driver, self.database, self.prefix, dim)

    @classmethod
    def from_env(cls, env_file=None, **kw) -> "Neo4jGraphStore":
        """Connect using NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
        from the environment or .env (environment wins). `Neo4jGraphStore()`
        with no arguments does the same."""
        from .config import Neo4jConfig
        cfg = Neo4jConfig.from_env(env_file)
        return cls(cfg.uri, cfg.user, cfg.password, database=cfg.database, **kw)

    # ---------------------------------------------------------------- plumbing
    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _q(self, cypher: str) -> str:
        q = self._qcache.get(cypher)
        if q is None:
            q = self._qcache[cypher] = apply_prefix(cypher, self.prefix)
        return q

    def _write(self, fn):
        with self._driver.session(database=self.database) as s:
            return s.execute_write(fn)

    def _read(self, fn):
        with self._driver.session(database=self.database) as s:
            return s.execute_read(fn)

    def _rows(self, tx, cypher: str, **params) -> list[dict]:
        return [r.data() for r in tx.run(self._q(cypher), params)]

    def _one(self, tx, cypher: str, **params) -> Optional[dict]:
        rows = self._rows(tx, cypher, **params)
        return rows[0] if rows else None

    def _read_rows(self, cypher: str, **params) -> list[dict]:
        return self._read(lambda tx: self._rows(tx, cypher, **params))

    # converters
    @staticmethod
    def _person(p: dict) -> Person:
        return Person(id=p["id"], name=p.get("name"), kind=p.get("kind") or "unknown",
                      aliases=tuple(p.get("aliases") or ()), first_seen=p.get("first_seen", 0.0),
                      last_seen=p.get("last_seen", 0.0), persona_summary=p.get("persona_summary"),
                      persona_updated=p.get("persona_updated"))

    @staticmethod
    def _sighting(s: dict) -> Sighting:
        return Sighting(id=s["id"], ts=s["ts"], camera=s["camera"], track_id=s["track_id"],
                        person_id=s.get("person_id"), cluster_id=s.get("cluster_id"),
                        zones=tuple(s.get("zones") or ()), activity=s.get("activity"),
                        snapshot_path=s.get("snapshot_path"))

    @staticmethod
    def _fact(f: dict) -> Fact:
        return Fact(f["id"], f["person_id"], f["key"], f["value"], f["source"],
                    float(f["confidence"]), f["ts"])

    @staticmethod
    def _doubt(d: dict) -> Doubt:
        return Doubt(id=d["id"], kind=d["kind"], status=d["status"],
                     about_ids=tuple(d.get("about_ids") or ()),
                     snapshot_paths=tuple(d.get("snapshot_paths") or ()),
                     question=d.get("question"), answer=d.get("answer"), rounds=d.get("rounds", 0),
                     created=d.get("created", 0.0), updated=d.get("updated", 0.0),
                     last_error=d.get("last_error"), message_ref=d.get("message_ref"),
                     detail=json.loads(d["detail"]) if d.get("detail") else {})

    def _emb(self, embedding: Sequence[float]) -> list[float]:
        emb = [float(x) for x in embedding]
        if len(emb) != self.dim:
            raise ValueError(f"embedding has {len(emb)} dims, expected {self.dim}")
        return emb

    def _need_person_tx(self, tx, person_id: str) -> dict:
        row = self._one(tx, "MATCH (p:Person {id: $id}) RETURN p{.*} AS p", id=person_id)
        if row is None:
            raise KeyError(f"unknown person {person_id}")
        return row["p"]

    # ------------------------------------------------------------------ people
    def upsert_person(self, person_id=None, *, name=None, kind=None, aliases=(), ts=None):
        ts = time.time() if ts is None else float(ts)
        if kind is not None and kind not in PERSON_KINDS:
            raise ValueError(f"bad kind {kind!r}")
        pid = person_id or new_id("p")

        def tx_fn(tx):
            tx.run(self._q("""
                MERGE (p:Person {id: $id})
                ON CREATE SET p.name = $name, p.kind = coalesce($kind, 'unknown'),
                              p.aliases = $aliases, p.first_seen = $ts, p.last_seen = $ts
                ON MATCH SET p.name = coalesce($name_upd, p.name),
                             p.kind = coalesce($kind, p.kind),
                             p.aliases = reduce(acc = coalesce(p.aliases, []), a IN $aliases |
                                                CASE WHEN a IN acc THEN acc ELSE acc + a END),
                             p.last_seen = CASE WHEN p.last_seen > $ts THEN p.last_seen ELSE $ts END
                """), id=pid, name=name, name_upd=name or None, kind=kind or None,
                aliases=_dedup(str(a) for a in aliases), ts=ts).consume()
        self._write(tx_fn)
        return pid

    def get_person(self, person_id):
        rows = self._read_rows("MATCH (p:Person {id: $id}) RETURN p{.*} AS p", id=person_id)
        return self._person(rows[0]["p"]) if rows else None

    def search_people(self, text, limit=10):
        rows = self._read_rows("""
            MATCH (p:Person)
            WHERE (p.name IS NOT NULL AND p.name <> '' AND toLower(p.name) CONTAINS $t)
               OR any(a IN coalesce(p.aliases, []) WHERE toLower(a) CONTAINS $t)
            RETURN p{.*} AS p ORDER BY p.first_seen, p.id LIMIT $limit
            """, t=text.lower().strip(), limit=int(limit))
        return [self._person(r["p"]) for r in rows]

    def list_people(self, limit: int = 1000) -> list[Person]:
        """Superset helper (CLI): every person, most recently seen first."""
        rows = self._read_rows("MATCH (p:Person) RETURN p{.*} AS p ORDER BY p.last_seen DESC LIMIT $limit",
                               limit=int(limit))
        return [self._person(r["p"]) for r in rows]

    def set_persona_summary(self, person_id, summary):
        def tx_fn(tx):
            self._need_person_tx(tx, person_id)
            tx.run(self._q("MATCH (p:Person {id: $id}) SET p.persona_summary = $s, p.persona_updated = $now"),
                   id=person_id, s=summary, now=time.time()).consume()
        self._write(tx_fn)

    def merge_people(self, keep_id, drop_id):
        if keep_id == drop_id:
            raise ValueError("merge_people: keep_id == drop_id")

        def tx_fn(tx):
            keep = self._need_person_tx(tx, keep_id)
            drop = self._need_person_tx(tx, drop_id)
            p = dict(keep=keep_id, drop=drop_id)
            # clusters
            tx.run(self._q("""
                MATCH (c:FaceCluster {person_id: $drop}) SET c.person_id = $keep
                WITH c OPTIONAL MATCH (c)-[r:BELONGS_TO]->() DELETE r
                WITH DISTINCT c MATCH (k:Person {id: $keep}) MERGE (c)-[:BELONGS_TO]->(k)"""), p).consume()
            # sightings
            tx.run(self._q("""
                MATCH (s:Sighting {person_id: $drop}) SET s.person_id = $keep
                WITH s MATCH (k:Person {id: $keep}) MERGE (s)-[:OF]->(k)"""), p).consume()
            # facts
            tx.run(self._q("""
                MATCH (f:Fact {person_id: $drop}) SET f.person_id = $keep
                WITH f MATCH (k:Person {id: $keep}) MERGE (k)-[:HAS_FACT]->(f)"""), p).consume()
            # relations (dedup; a relation between keep and drop would become a self-loop: dropped)
            tx.run(self._q("""
                MATCH (d:Person {id: $drop})-[r:RELATED]->(o:Person) WHERE NOT o.id IN [$keep, $drop]
                MATCH (k:Person {id: $keep})
                MERGE (k)-[n:RELATED {relation: r.relation}]->(o)
                ON CREATE SET n.source = r.source, n.ts = r.ts, n.seq = r.seq"""), p).consume()
            tx.run(self._q("""
                MATCH (o:Person)-[r:RELATED]->(d:Person {id: $drop}) WHERE NOT o.id IN [$keep, $drop]
                MATCH (k:Person {id: $keep})
                MERGE (o)-[n:RELATED {relation: r.relation}]->(k)
                ON CREATE SET n.source = r.source, n.ts = r.ts, n.seq = r.seq"""), p).consume()
            # doubts that pointed at drop now point at keep
            tx.run(self._q("""
                MATCH (x:Doubt) WHERE $drop IN x.about_ids
                SET x.about_ids = reduce(acc = [], i IN x.about_ids |
                    CASE WHEN (CASE WHEN i = $drop THEN $keep ELSE i END) IN acc THEN acc
                         ELSE acc + (CASE WHEN i = $drop THEN $keep ELSE i END) END)
                WITH x MATCH (k:Person {id: $keep}) MERGE (x)-[:ABOUT]->(k)"""), p).consume()
            aliases = list(keep.get("aliases") or []) + list(drop.get("aliases") or [])
            if drop.get("name"):
                aliases.append(drop["name"])
            tx.run(self._q("""
                MATCH (k:Person {id: $keep}) SET k.aliases = $aliases,
                    k.first_seen = $first, k.last_seen = $last
                WITH k MATCH (d:Person {id: $drop}) DETACH DELETE d"""),
                   keep=keep_id, drop=drop_id,
                   aliases=_dedup(a for a in aliases if a != keep.get("name")),
                   first=min(keep.get("first_seen", 0.0), drop.get("first_seen", 0.0)),
                   last=max(keep.get("last_seen", 0.0), drop.get("last_seen", 0.0))).consume()
        self._write(tx_fn)

    def forget_person(self, person_id):
        """Hard delete person, facts, relations, sightings, faces and clusters.
        Returns snapshot paths that the caller must delete from disk.
        Doubts about the person or their clusters are deleted too, and their
        snapshot paths returned."""
        def tx_fn(tx):
            self._need_person_tx(tx, person_id)
            cids = self._one(tx, "MATCH (c:FaceCluster {person_id: $id}) RETURN collect(c.id) AS c",
                             id=person_id)["c"]
            paths: list[str] = []
            paths += self._one(tx, """
                MATCH (f:Face) WHERE f.cluster_id IN $cids
                WITH f, f.snapshot_path AS sp DETACH DELETE f RETURN collect(sp) AS p""", cids=cids)["p"]
            paths += self._one(tx, """
                MATCH (s:Sighting) WHERE s.person_id = $id OR s.cluster_id IN $cids
                WITH s, s.snapshot_path AS sp DETACH DELETE s RETURN collect(sp) AS p""",
                               id=person_id, cids=cids)["p"]
            paths += self._one(tx, """
                MATCH (x:Doubt) WHERE any(i IN x.about_ids WHERE i = $id OR i IN $cids)
                WITH x, coalesce(x.snapshot_paths, []) AS sp DETACH DELETE x
                RETURN reduce(acc = [], l IN collect(sp) | acc + l) AS p""", id=person_id, cids=cids)["p"]
            tx.run(self._q("MATCH (c:FaceCluster) WHERE c.id IN $cids DETACH DELETE c"), cids=cids).consume()
            tx.run(self._q("MATCH (f:Fact {person_id: $id}) DETACH DELETE f"), id=person_id).consume()
            tx.run(self._q("MATCH (p:Person {id: $id}) DETACH DELETE p"), id=person_id).consume()
            return sorted({p for p in paths if p})
        return self._write(tx_fn)

    # ------------------------------------------------------------------- faces
    def _match_tx(self, tx, emb: list[float], k: int) -> list[FaceMatch]:
        if k <= 0:
            return []
        rows = tx.run(self._q("""
            CALL db.index.vector.queryNodes($index, $kk, $emb) YIELD node
            WITH node, vector.similarity.cosine(node.centroid, $emb) AS score
            RETURN node.id AS id, node.person_id AS person_id, score
            ORDER BY score DESC, id LIMIT $k"""),
            index=self._centroid_index, kk=max(int(k), 10), emb=emb, k=int(k))
        return [FaceMatch(r["id"], r["person_id"], 2.0 * r["score"] - 1.0) for r in rows]

    def add_face(self, embedding, *, quality, ts, snapshot_path, cluster_id=None):
        emb = self._emb(embedding)
        fid = new_id("f")

        def tx_fn(tx):
            cid = cluster_id
            if cid is None:
                best = self._match_tx(tx, emb, 1)
                if best and best[0].score >= self.cluster_threshold:
                    cid = best[0].cluster_id
                else:
                    cid = new_id("fc")
                    tx.run(self._q("""CREATE (:FaceCluster {id: $id, person_id: null, n_faces: 0,
                                      centroid: $emb, created: $now})"""),
                           id=cid, emb=emb, now=time.time()).consume()
            row = self._one(tx, """
                MATCH (c:FaceCluster {id: $cid})
                SET c.centroid = [i IN range(0, size($emb) - 1) |
                                  (c.centroid[i] * c.n_faces + $emb[i]) / (c.n_faces + 1)],
                    c.n_faces = c.n_faces + 1
                CREATE (f:Face {id: $fid, embedding: $emb, quality: $q, ts: $ts,
                                snapshot_path: $sp, cluster_id: $cid})-[:IN]->(c)
                RETURN c.id AS cid""", cid=cid, emb=emb, fid=fid, q=float(quality), ts=float(ts),
                            sp=snapshot_path)
            if row is None:
                raise KeyError(f"unknown face cluster {cid}")
            return cid
        return fid, self._write(tx_fn)

    def match_face(self, embedding, k=3):
        emb = self._emb(embedding)
        return self._read(lambda tx: self._match_tx(tx, emb, k))

    def label_face_cluster(self, cluster_id, person_id):
        def tx_fn(tx):
            self._need_person_tx(tx, person_id)
            row = self._one(tx, """
                MATCH (c:FaceCluster {id: $cid}) SET c.person_id = $pid
                WITH c OPTIONAL MATCH (c)-[r:BELONGS_TO]->() DELETE r
                WITH DISTINCT c MATCH (p:Person {id: $pid}) MERGE (c)-[:BELONGS_TO]->(p)
                RETURN c.id AS id""", cid=cluster_id, pid=person_id)
            if row is None:
                raise KeyError(f"unknown face cluster {cluster_id}")
            tx.run(self._q("""
                MATCH (s:Sighting {cluster_id: $cid}) WHERE s.person_id IS NULL
                SET s.person_id = $pid
                WITH s MATCH (p:Person {id: $pid}) MERGE (s)-[:OF]->(p)"""),
                   cid=cluster_id, pid=person_id).consume()
        self._write(tx_fn)

    def cluster_size(self, cluster_id):
        rows = self._read_rows("MATCH (c:FaceCluster {id: $id}) RETURN c.n_faces AS n", id=cluster_id)
        if not rows:
            raise KeyError(f"unknown face cluster {cluster_id}")
        return rows[0]["n"]

    def list_clusters(self, person_id: Optional[str] = None, limit: int = 1000) -> list[dict]:
        """Superset helper: [{id, person_id, n_faces, created}]."""
        return self._read_rows("""
            MATCH (c:FaceCluster) WHERE $pid IS NULL OR c.person_id = $pid
            RETURN c.id AS id, c.person_id AS person_id, c.n_faces AS n_faces, c.created AS created
            ORDER BY c.created LIMIT $limit""", pid=person_id, limit=int(limit))

    # --------------------------------------------- sightings / facts / relations
    def add_sighting(self, s):
        sid = s.id or new_id("s")

        def tx_fn(tx):
            person_id = s.person_id
            if person_id is None and s.cluster_id:
                row = self._one(tx, "MATCH (c:FaceCluster {id: $id}) RETURN c.person_id AS pid", id=s.cluster_id)
                person_id = row["pid"] if row else None
            tx.run(self._q("""
                MERGE (s:Sighting {id: $id})
                SET s.ts = $ts, s.camera = $camera, s.track_id = $track_id, s.person_id = $pid,
                    s.cluster_id = $cid, s.zones = $zones, s.activity = $activity,
                    s.snapshot_path = $sp
                WITH s OPTIONAL MATCH (s)-[r:OF]->() DELETE r
                WITH DISTINCT s
                OPTIONAL MATCH (p:Person {id: $pid})
                OPTIONAL MATCH (c:FaceCluster {id: $cid})
                FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (s)-[:OF]->(p)
                    SET p.last_seen = CASE WHEN p.last_seen > $ts THEN p.last_seen ELSE $ts END)
                FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | MERGE (s)-[:OF]->(c))"""),
                   id=sid, ts=float(s.ts), camera=s.camera, track_id=int(s.track_id), pid=person_id,
                   cid=s.cluster_id, zones=list(s.zones), activity=s.activity,
                   sp=s.snapshot_path).consume()
        self._write(tx_fn)
        return sid

    def recent_sightings(self, *, person_id=None, cluster_id=None, limit=20):
        rows = self._read_rows("""
            MATCH (s:Sighting)
            WHERE ($pid IS NULL OR s.person_id = $pid) AND ($cid IS NULL OR s.cluster_id = $cid)
            RETURN s{.*} AS s ORDER BY s.ts DESC LIMIT $limit""",
                               pid=person_id, cid=cluster_id, limit=int(limit))
        return [self._sighting(r["s"]) for r in rows]

    def add_fact(self, person_id, key, value, *, source, confidence, ts=None):
        fid = new_id("fact")

        def tx_fn(tx):
            self._need_person_tx(tx, person_id)
            if source not in FACT_SOURCES:
                raise ValueError(f"bad source {source!r}")
            tx.run(self._q("""
                MATCH (p:Person {id: $pid})
                CREATE (p)-[:HAS_FACT]->(:Fact {id: $id, person_id: $pid, key: $key, value: $value,
                                                source: $source, confidence: $conf, ts: $ts, seq: $seq})"""),
                   pid=person_id, id=fid, key=key, value=value, source=source,
                   conf=float(confidence), ts=time.time() if ts is None else float(ts),
                   seq=time.time_ns()).consume()
        self._write(tx_fn)
        return fid

    def facts(self, person_id):
        rows = self._read_rows("MATCH (f:Fact {person_id: $pid}) RETURN f{.*} AS f ORDER BY f.ts, f.seq",
                               pid=person_id)
        return [self._fact(r["f"]) for r in rows]

    def relate(self, a_id, relation, b_id, *, source):
        def tx_fn(tx):
            self._need_person_tx(tx, a_id)
            self._need_person_tx(tx, b_id)
            tx.run(self._q("""
                MATCH (a:Person {id: $a}), (b:Person {id: $b})
                MERGE (a)-[r:RELATED {relation: $rel}]->(b)
                ON CREATE SET r.source = $source, r.ts = $now, r.seq = $seq"""),
                   a=a_id, b=b_id, rel=relation, source=source, now=time.time(),
                   seq=time.time_ns()).consume()
        self._write(tx_fn)

    def relations(self, person_id):
        def tx_fn(tx):
            out = self._rows(tx, """MATCH (p:Person {id: $id})-[r:RELATED]->(o:Person)
                                    RETURN r.relation AS rel, o.id AS other ORDER BY r.seq""", id=person_id)
            inn = self._rows(tx, """MATCH (o:Person)-[r:RELATED]->(p:Person {id: $id})
                                    RETURN r.relation AS rel, o.id AS other ORDER BY r.seq""", id=person_id)
            return [(r["rel"], r["other"], "out") for r in out] + [(r["rel"], r["other"], "in") for r in inn]
        return self._read(tx_fn)

    def person_context(self, person_id):
        p = self.get_person(person_id)
        if p is None:
            raise KeyError(f"unknown person {person_id}")
        return {
            "person": p,
            "facts": self.facts(person_id),
            "relations": self.relations(person_id),
            "recent_sightings": self.recent_sightings(person_id=person_id, limit=10),
            "clusters": [c["id"] for c in self.list_clusters(person_id)],
        }

    # ------------------------------------------------------------------ doubts
    def _write_doubt_tx(self, tx, d: Doubt, seq: Optional[int] = None) -> None:
        tx.run(self._q("""
            MERGE (x:Doubt {id: $id})
            SET x.kind = $kind, x.status = $status, x.about_ids = $about, x.snapshot_paths = $snaps,
                x.question = $question, x.answer = $answer, x.rounds = $rounds,
                x.created = $created, x.updated = $updated, x.last_error = $last_error,
                x.message_ref = $message_ref, x.detail = $detail,
                x.seq = coalesce($seq, x.seq)
            WITH x OPTIONAL MATCH (x)-[r:ABOUT]->() DELETE r
            WITH DISTINCT x
            CALL () {
                MATCH (n:Person) WHERE n.id IN $about RETURN n
                UNION MATCH (n:FaceCluster) WHERE n.id IN $about RETURN n
                UNION MATCH (n:Sighting) WHERE n.id IN $about RETURN n
            }
            MERGE (x)-[:ABOUT]->(n)"""), id=d.id, kind=d.kind, status=d.status, about=list(d.about_ids),
               snaps=list(d.snapshot_paths), question=d.question, answer=d.answer,
               rounds=int(d.rounds), created=float(d.created), updated=float(d.updated),
               last_error=d.last_error, message_ref=d.message_ref,
               detail=json.dumps(d.detail or {}, sort_keys=True, default=str), seq=seq).consume()

    def create_doubt(self, d):
        now = time.time()
        if d.kind not in DOUBT_KINDS or d.status not in DOUBT_STATUSES:
            raise ValueError("bad doubt kind/status")
        d = replace(d, id=d.id or new_id("d"), created=d.created or now, updated=now)
        self._write(lambda tx: self._write_doubt_tx(tx, d, seq=time.time_ns()))
        return d.id

    def get_doubt(self, doubt_id):
        rows = self._read_rows("MATCH (x:Doubt {id: $id}) RETURN x{.*} AS x", id=doubt_id)
        return self._doubt(rows[0]["x"]) if rows else None

    def update_doubt(self, doubt_id, **fields):
        if "status" in fields and fields["status"] not in DOUBT_STATUSES:
            raise ValueError("bad status")
        bad = set(fields) - (_DOUBT_FIELDS - {"id", "updated"})
        if bad:
            raise TypeError(f"update_doubt: cannot set {sorted(bad)}")
        if "kind" in fields and fields["kind"] not in DOUBT_KINDS:
            raise ValueError("bad kind")

        def tx_fn(tx):
            # SET first so the node is write-locked for the read-modify-write.
            row = self._one(tx, "MATCH (x:Doubt {id: $id}) SET x.updated = x.updated RETURN x{.*} AS x",
                            id=doubt_id)
            if row is None:
                raise KeyError(doubt_id)
            d = replace(self._doubt(row["x"]), **fields, updated=time.time())
            self._write_doubt_tx(tx, d)
            return d
        return self._write(tx_fn)

    def list_doubts(self, status=None, limit=50):
        rows = self._read_rows("""
            MATCH (x:Doubt) WHERE $status IS NULL OR x.status = $status
            RETURN x{.*} AS x ORDER BY x.created, x.seq LIMIT $limit""", status=status, limit=int(limit))
        return [self._doubt(r["x"]) for r in rows]

    def find_doubt_by_message(self, message_ref):
        rows = self._read_rows("""
            MATCH (x:Doubt {message_ref: $ref}) RETURN x{.*} AS x ORDER BY x.created, x.seq LIMIT 1""",
                               ref=message_ref)
        return self._doubt(rows[0]["x"]) if rows else None

    # --------------------------------------------------------------- graph_read
    def graph_read(self, cypher, params=None, limit=100):
        """Read-only Cypher for the worker: client-side clause check, READ
        access session + READ transaction, timeout, at most `limit` rows
        (capped at GRAPH_READ_MAX_ROWS). Vectors are elided from the output."""
        from neo4j import READ_ACCESS, unit_of_work
        check_read_only(cypher)
        limit = max(1, min(int(limit), GRAPH_READ_MAX_ROWS))
        query = self._q(cypher)

        @unit_of_work(timeout=self.read_timeout_s)
        def tx_fn(tx):
            result = tx.run(query, params or {})
            return [{k: _jsonable(v) for k, v in rec.items()} for rec in islice(result, limit)]
        with self._driver.session(database=self.database, default_access_mode=READ_ACCESS) as s:
            return s.execute_read(tx_fn)

    # ------------------------------------------------------------ admin helpers
    def stats(self) -> dict[str, int]:
        out = {}
        for label in LABELS:
            out[label] = self._read_rows(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]
        return out

    def export(self) -> dict[str, Any]:
        """All worldcam nodes and edges as JSON-able dicts (vectors included)."""
        def tx_fn(tx):
            nodes = {}
            for label in LABELS:
                nodes[label] = [r["n"] for r in self._rows(
                    tx, f"MATCH (n:{label}) RETURN n{{.*}} AS n ORDER BY n.id")]
            edges = self._rows(tx, f"""
                MATCH (a)-[r]->(b)
                WHERE any(l IN labels(a) WHERE l IN $labels) AND any(l IN labels(b) WHERE l IN $labels)
                RETURN a.id AS src, type(r) AS type, b.id AS dst, properties(r) AS props
                ORDER BY src, type, dst""", labels=[self.prefix + lab for lab in LABELS])
            return {"nodes": nodes, "edges": edges}
        return self._read(tx_fn)

    def wipe(self, batch: int = 5000) -> int:
        """Delete every node of this store's labels (schema kept). Returns count."""
        total = 0
        for label in LABELS:
            while True:
                n = self._write(lambda tx: self._one(
                    tx, f"MATCH (n:{label}) WITH n LIMIT $b DETACH DELETE n RETURN count(*) AS n",
                    b=batch)["n"])
                total += n
                if n < batch:
                    break
        return total


def _jsonable(v):
    from neo4j.graph import Node, Path, Relationship
    if isinstance(v, Node):
        d = {k: (f"<{len(x)}-d vector>" if k in _VECTOR_KEYS and isinstance(x, list) else _jsonable(x))
             for k, x in v.items()}
        d["_labels"] = sorted(v.labels)
        return d
    if isinstance(v, Relationship):
        d = {k: _jsonable(x) for k, x in v.items()}
        d["_type"] = v.type
        return d
    if isinstance(v, Path):
        return [_jsonable(n) for n in v.nodes]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: (f"<{len(x)}-d vector>" if k in _VECTOR_KEYS and isinstance(x, list) else _jsonable(x))
                for k, x in v.items()}
    if hasattr(v, "iso_format"):
        return v.iso_format()
    return v
