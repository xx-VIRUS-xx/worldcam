"""Idempotent Neo4j schema for the worldcam graph (Neo4j 5.26 syntax).

Every label gets a unique-id constraint. Vector indexes:
  face_embedding  on Face.embedding         (cosine, 512)  - per-face search
  cluster_centroid on FaceCluster.centroid  (cosine, 512)  - used by match_face

A `label_prefix` (default "") namespaces labels and index names. The contract
tests use it ("WcTest") so test data never mixes with real data; Neo4j
Community has only one user database, so a separate test DB is not possible.
"""
from __future__ import annotations

import re

LABELS = ("Person", "FaceCluster", "Face", "Sighting", "Fact", "Doubt")
EMBED_DIM = 512
RANGE_INDEXES = (("Sighting", "person_id"), ("Sighting", "cluster_id"), ("Sighting", "ts"),
                 ("Fact", "person_id"), ("FaceCluster", "person_id"), ("Face", "cluster_id"),
                 ("Doubt", "status"), ("Doubt", "message_ref"), ("Person", "name"))
_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,30}$")
_LABEL_RE = re.compile(r":(FaceCluster|Face|Person|Sighting|Fact|Doubt)\b")


def check_prefix(prefix: str) -> str:
    if prefix and not _PREFIX_RE.match(prefix):
        raise ValueError(f"bad label prefix {prefix!r}")
    return prefix


def apply_prefix(cypher: str, prefix: str) -> str:
    """Rewrite `:Person` -> `:<prefix>Person` etc. (labels only; rel types untouched)."""
    if not prefix:
        return cypher
    return _LABEL_RE.sub(lambda m: f":{prefix}{m.group(1)}", cypher)


def index_name(base: str, prefix: str) -> str:
    return f"{prefix.lower()}_{base}" if prefix else base


def schema_statements(prefix: str = "", dim: int = EMBED_DIM) -> list[str]:
    check_prefix(prefix)
    stmts = []
    for label in LABELS:
        lab = prefix + label
        stmts.append(f"CREATE CONSTRAINT {index_name(label.lower() + '_id', prefix)} IF NOT EXISTS "
                     f"FOR (n:{lab}) REQUIRE n.id IS UNIQUE")
    for label, prop in RANGE_INDEXES:
        stmts.append(f"CREATE INDEX {index_name(f'{label.lower()}_{prop}', prefix)} IF NOT EXISTS "
                     f"FOR (n:{prefix}{label}) ON (n.{prop})")
    for name, label, prop in (("face_embedding", "Face", "embedding"),
                              ("cluster_centroid", "FaceCluster", "centroid")):
        stmts.append(
            f"CREATE VECTOR INDEX {index_name(name, prefix)} IF NOT EXISTS "
            f"FOR (n:{prefix}{label}) ON n.{prop} "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dim)}, "
            f"`vector.similarity_function`: 'cosine'}}}}")
    return stmts


def ensure_schema(driver, database: str = "neo4j", prefix: str = "", dim: int = EMBED_DIM,
                  wait_s: float = 60.0) -> list[str]:
    """Create constraints/indexes if missing and wait until they are ONLINE."""
    stmts = schema_statements(prefix, dim)
    with driver.session(database=database) as s:
        for q in stmts:
            s.run(q).consume()
        s.run("CALL db.awaitIndexes($t)", t=int(wait_s)).consume()
    return stmts


def drop_schema(driver, database: str = "neo4j", prefix: str = "") -> None:
    """Drop the constraints/indexes this module creates for `prefix` (tests only)."""
    names = []
    for label in LABELS:
        names.append(("CONSTRAINT", index_name(label.lower() + "_id", prefix)))
    for label, prop in RANGE_INDEXES:
        names.append(("INDEX", index_name(f"{label.lower()}_{prop}", prefix)))
    names += [("INDEX", index_name("face_embedding", prefix)),
              ("INDEX", index_name("cluster_centroid", prefix))]
    with driver.session(database=database) as s:
        for kind, name in names:
            s.run(f"DROP {kind} {name} IF EXISTS").consume()
