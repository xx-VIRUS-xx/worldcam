"""Graph contract shared by identity, the ambiguity engine and the MCP server.

Neo4jGraphStore (graph/store.py) and InMemoryGraphStore (below) both implement
GraphStore and must pass tests/contract_graph.py. Code outside graph/ only ever
sees this module.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Protocol, Sequence

ENTITY_TYPES = ("person", "animal", "object", "other")
PERSON_KINDS = ("owner", "household", "visitor", "unknown")
ANIMAL_KINDS = ("pet", "stray", "wildlife", "visitor", "unknown")
OBJECT_KINDS = ("belonging", "device", "furniture", "package", "vehicle", "unknown")
OTHER_KINDS = ("household", "visitor", "belonging", "device", "package", "vehicle", "unknown")
ENTITY_KINDS = {
    "person": PERSON_KINDS,
    "animal": ANIMAL_KINDS,
    "object": OBJECT_KINDS,
    "other": OTHER_KINDS,
}
FACT_SOURCES = ("owner", "observed", "inferred")
DOUBT_KINDS = ("unknown_face", "low_confidence", "conflict", "metadata_gap", "unknown_entity")
DOUBT_STATUSES = ("open", "asked", "answered", "resolved", "dismissed")


def validate_entity_kind(entity_type: str, kind: Optional[str]) -> None:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"bad entity_type {entity_type!r}")
    if kind is not None and kind not in ENTITY_KINDS[entity_type]:
        raise ValueError(f"bad kind {kind!r} for entity_type {entity_type!r}")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class Person:
    id: str
    name: Optional[str] = None
    kind: str = "unknown"
    aliases: tuple[str, ...] = ()
    first_seen: float = 0.0
    last_seen: float = 0.0
    persona_summary: Optional[str] = None
    persona_updated: Optional[float] = None
    entity_type: str = "person"
    label: Optional[str] = None


@dataclass(frozen=True)
class FaceMatch:
    cluster_id: str
    person_id: Optional[str]      # None = cluster not yet labeled
    score: float                  # cosine similarity, 1.0 = identical


@dataclass(frozen=True)
class Sighting:
    id: str
    ts: float
    camera: str
    track_id: int
    person_id: Optional[str] = None
    cluster_id: Optional[str] = None
    zones: tuple[str, ...] = ()
    activity: Optional[str] = None
    snapshot_path: Optional[str] = None
    entity_label: Optional[str] = None


@dataclass(frozen=True)
class Fact:
    id: str
    person_id: str
    key: str                      # "relation_to_owner", "works_at", "phone_owner", ...
    value: str
    source: str                   # one of FACT_SOURCES
    confidence: float
    ts: float


@dataclass(frozen=True)
class Doubt:
    id: str
    kind: str                     # one of DOUBT_KINDS
    status: str = "open"          # one of DOUBT_STATUSES
    about_ids: tuple[str, ...] = ()   # person / cluster / sighting ids
    snapshot_paths: tuple[str, ...] = ()
    question: Optional[str] = None
    answer: Optional[str] = None
    rounds: int = 0
    created: float = 0.0
    updated: float = 0.0
    last_error: Optional[str] = None
    message_ref: Optional[str] = None  # messenger message id the owner replies to
    detail: dict = field(default_factory=dict)


class GraphStore(Protocol):
    # people / entities
    def upsert_person(self, person_id: Optional[str] = None, *, name: Optional[str] = None,
                      kind: Optional[str] = None, aliases: Sequence[str] = (), ts: Optional[float] = None,
                      entity_type: Optional[str] = None, label: Optional[str] = None) -> str: ...
    def get_person(self, person_id: str) -> Optional[Person]: ...
    def search_people(self, text: str, limit: int = 10) -> list[Person]: ...
    def list_people(self, limit: int = 1000) -> list[Person]: ...
    def set_persona_summary(self, person_id: str, summary: str) -> None: ...
    def merge_people(self, keep_id: str, drop_id: str) -> None: ...
    def forget_person(self, person_id: str) -> list[str]:
        """Hard delete person, facts, relations, sightings, faces, clusters, and
        every doubt about the person or their clusters (doubts hold their crops
        and questions). Returns snapshot paths the caller must delete from disk."""

    # faces
    def add_face(self, embedding: Sequence[float], *, quality: float, ts: float,
                 snapshot_path: Optional[str], cluster_id: Optional[str] = None,
                 new_cluster: bool = False) -> tuple[str, str]:
        """Store a face. If cluster_id is None, join the best cluster above the
        store's cluster threshold or start a new one; new_cluster=True always
        starts a new one (used for CONFLICT faces so they can't inherit either
        disputed person). Returns (face_id, cluster_id)."""
    def match_face(self, embedding: Sequence[float], k: int = 3) -> list[FaceMatch]: ...
    def label_face_cluster(self, cluster_id: str, person_id: str) -> None:
        """Also accepts entity cluster ids (so the worker's one labeling tool covers both)."""
    def cluster_size(self, cluster_id: str) -> int: ...

    # entity views (animals / objects; appearance embeddings from perception/reid.py)
    def add_entity_view(self, label: str, embedding: Sequence[float], *, quality: float, ts: float,
                        snapshot_path: Optional[str], cluster_id: Optional[str] = None,
                        new_cluster: bool = False) -> tuple[str, str]:
        """Store one appearance view of a `label` entity (YOLO class: "dog", "cup", ...).
        If cluster_id is None, join the best cluster WITH THE SAME LABEL above the
        store's entity cluster threshold, else start a new one (new_cluster=True
        always starts one, used for CONFLICT views). Returns (view_id, cluster_id)."""
    def match_entity(self, label: str, embedding: Sequence[float], k: int = 3) -> list[FaceMatch]:
        """Nearest entity clusters of the same label, plain cosine, best first.
        FaceMatch.cluster_id/person_id/score as for faces; a cup never matches a dog."""
    def label_entity_cluster(self, cluster_id: str, person_id: str) -> None: ...
    def entity_cluster_size(self, cluster_id: str) -> int: ...

    # sightings / facts / relations
    def add_sighting(self, s: Sighting) -> str: ...
    def recent_sightings(self, *, person_id: Optional[str] = None, cluster_id: Optional[str] = None,
                         limit: int = 20) -> list[Sighting]: ...
    def add_fact(self, person_id: str, key: str, value: str, *, source: str,
                 confidence: float, ts: Optional[float] = None) -> str: ...
    def facts(self, person_id: str) -> list[Fact]: ...
    def relate(self, a_id: str, relation: str, b_id: str, *, source: str) -> None: ...
    def relations(self, person_id: str) -> list[tuple[str, str, str]]:
        """[(relation, other_person_id, direction 'out'|'in')]"""
    def person_context(self, person_id: str) -> dict[str, Any]:
        """Everything a worker needs: person, facts, relations, recent sightings, clusters."""

    # doubts
    def create_doubt(self, d: Doubt) -> str: ...
    def get_doubt(self, doubt_id: str) -> Optional[Doubt]: ...
    def update_doubt(self, doubt_id: str, **fields: Any) -> Doubt: ...
    def list_doubts(self, status: Optional[str] = None, limit: int = 50) -> list[Doubt]: ...
    def find_doubt_by_message(self, message_ref: str) -> Optional[Doubt]: ...

    # read-only escape hatch for the worker (Neo4j: READ transaction; fake: unsupported)
    def graph_read(self, cypher: str, params: Optional[dict] = None, limit: int = 100) -> list[dict]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryGraphStore:
    """Reference implementation for tests. Not for production: no persistence,
    linear-scan face search."""

    def __init__(self, cluster_threshold: float = 0.45, entity_cluster_threshold: float = 0.88):
        self.cluster_threshold = cluster_threshold
        self.entity_cluster_threshold = entity_cluster_threshold
        self.entity_views: dict[str, dict] = {}     # id -> {label, embedding, cluster_id, quality, ts, snapshot_path}
        self.entity_clusters: dict[str, dict] = {}  # id -> {label, person_id, centroid, n}
        self.people: dict[str, Person] = {}
        self.faces: dict[str, dict] = {}            # id -> {embedding, cluster_id, quality, ts, snapshot_path}
        self.clusters: dict[str, dict] = {}         # id -> {person_id, centroid, n}
        self.sightings: dict[str, Sighting] = {}
        self._facts: dict[str, Fact] = {}
        self._rels: list[tuple[str, str, str, str]] = []  # (a, relation, b, source)
        self.doubts: dict[str, Doubt] = {}

    # people / entities
    def upsert_person(self, person_id=None, *, name=None, kind=None, aliases=(), ts=None,
                      entity_type=None, label=None):
        ts = time.time() if ts is None else ts
        existing = self.people.get(person_id) if person_id else None
        eff_type = entity_type or (existing.entity_type if existing else "person")
        validate_entity_kind(eff_type, kind)
        if existing is not None:
            p = replace(existing, name=name or existing.name, kind=kind or existing.kind,
                        aliases=tuple(dict.fromkeys(existing.aliases + tuple(aliases))),
                        last_seen=max(existing.last_seen, ts),
                        entity_type=eff_type,
                        label=label or existing.label)
        else:
            person_id = person_id or new_id("p")
            p = Person(person_id, name, kind or "unknown", tuple(aliases), ts, ts,
                       entity_type=eff_type, label=label)
        self.people[person_id] = p
        return person_id

    def get_person(self, person_id):
        return self.people.get(person_id)

    def search_people(self, text, limit=10):
        t = text.lower().strip()
        hits = [p for p in self.people.values()
                if (p.name and t in p.name.lower()) or any(t in a.lower() for a in p.aliases)]
        return hits[:limit]

    def list_people(self, limit=1000):
        return sorted(self.people.values(), key=lambda p: p.first_seen)[:limit]

    def set_persona_summary(self, person_id, summary):
        p = self._need_person(person_id)
        self.people[person_id] = replace(p, persona_summary=summary, persona_updated=time.time())

    def merge_people(self, keep_id, drop_id):
        if keep_id == drop_id:
            raise ValueError("cannot merge a person into itself")
        keep, drop = self._need_person(keep_id), self._need_person(drop_id)
        for c in (*self.clusters.values(), *self.entity_clusters.values()):
            if c["person_id"] == drop_id:
                c["person_id"] = keep_id
        for sid, s in list(self.sightings.items()):
            if s.person_id == drop_id:
                self.sightings[sid] = replace(s, person_id=keep_id)
        for fid, f in list(self._facts.items()):
            if f.person_id == drop_id:
                self._facts[fid] = replace(f, person_id=keep_id)
        rels, seen = [], set()
        for a, r, b, src in self._rels:
            a = keep_id if a == drop_id else a
            b = keep_id if b == drop_id else b
            if a == b or (a, r, b) in seen:
                continue  # drop self-loops and duplicates created by the merge
            seen.add((a, r, b))
            rels.append((a, r, b, src))
        self._rels = rels
        for did, d in list(self.doubts.items()):
            if drop_id in d.about_ids:
                ids = tuple(dict.fromkeys(keep_id if i == drop_id else i for i in d.about_ids))
                self.doubts[did] = replace(d, about_ids=ids)
        aliases = keep.aliases + drop.aliases + ((drop.name,) if drop.name else ())
        self.people[keep_id] = replace(keep, aliases=tuple(dict.fromkeys(a for a in aliases if a != keep.name)),
                                       first_seen=min(keep.first_seen, drop.first_seen),
                                       last_seen=max(keep.last_seen, drop.last_seen),
                                       label=keep.label or drop.label)
        del self.people[drop_id]

    def forget_person(self, person_id):
        self._need_person(person_id)
        paths: list[str] = []
        cids = [cid for cid, c in self.clusters.items() if c["person_id"] == person_id]
        ecids = [cid for cid, c in self.entity_clusters.items() if c["person_id"] == person_id]
        for fid, f in list(self.faces.items()):
            if f["cluster_id"] in cids:
                if f["snapshot_path"]:
                    paths.append(f["snapshot_path"])
                del self.faces[fid]
        for vid, v in list(self.entity_views.items()):
            if v["cluster_id"] in ecids:
                if v["snapshot_path"]:
                    paths.append(v["snapshot_path"])
                del self.entity_views[vid]
        for cid in cids:
            del self.clusters[cid]
        for cid in ecids:
            del self.entity_clusters[cid]
        cids = cids + ecids
        for sid, s in list(self.sightings.items()):
            if s.person_id == person_id or s.cluster_id in cids:
                if s.snapshot_path:
                    paths.append(s.snapshot_path)
                del self.sightings[sid]
        self._facts = {k: f for k, f in self._facts.items() if f.person_id != person_id}
        self._rels = [r for r in self._rels if person_id not in (r[0], r[2])]
        gone = {person_id, *cids}
        for did, d in list(self.doubts.items()):
            if gone & set(d.about_ids):
                paths.extend(d.snapshot_paths)
                del self.doubts[did]
        del self.people[person_id]
        return sorted(set(paths))

    # faces
    def add_face(self, embedding, *, quality, ts, snapshot_path, cluster_id=None, new_cluster=False):
        if new_cluster and cluster_id is not None:
            raise ValueError("cluster_id and new_cluster are exclusive")
        emb = [float(x) for x in embedding]
        if cluster_id is None:
            best = [] if new_cluster else self.match_face(emb, k=1)
            if best and best[0].score >= self.cluster_threshold:
                cluster_id = best[0].cluster_id
            else:
                cluster_id = new_id("fc")
                self.clusters[cluster_id] = {"person_id": None, "centroid": emb, "n": 0}
        c = self.clusters[cluster_id]
        n = c["n"]
        c["centroid"] = [(cv * n + e) / (n + 1) for cv, e in zip(c["centroid"], emb)]
        c["n"] = n + 1
        fid = new_id("f")
        self.faces[fid] = {"embedding": emb, "cluster_id": cluster_id, "quality": quality,
                           "ts": ts, "snapshot_path": snapshot_path}
        return fid, cluster_id

    def match_face(self, embedding, k=3):
        scored = [FaceMatch(cid, c["person_id"], cosine(embedding, c["centroid"]))
                  for cid, c in self.clusters.items()]
        return sorted(scored, key=lambda m: m.score, reverse=True)[:k]

    def label_face_cluster(self, cluster_id, person_id):
        if cluster_id in self.entity_clusters:       # entity cluster ids work here too (worker tool)
            return self.label_entity_cluster(cluster_id, person_id)
        self._need_person(person_id)
        self.clusters[cluster_id]["person_id"] = person_id
        for sid, s in list(self.sightings.items()):
            if s.cluster_id == cluster_id and s.person_id is None:
                self.sightings[sid] = replace(s, person_id=person_id)

    def cluster_size(self, cluster_id):
        if cluster_id in self.entity_clusters:
            return self.entity_clusters[cluster_id]["n"]
        return self.clusters[cluster_id]["n"]

    # entity views
    def add_entity_view(self, label, embedding, *, quality, ts, snapshot_path, cluster_id=None,
                        new_cluster=False):
        if new_cluster and cluster_id is not None:
            raise ValueError("cluster_id and new_cluster are exclusive")
        label = label.strip().lower()
        emb = [float(x) for x in embedding]
        if cluster_id is None:
            best = [] if new_cluster else self.match_entity(label, emb, k=1)
            if best and best[0].score >= self.entity_cluster_threshold:
                cluster_id = best[0].cluster_id
            else:
                cluster_id = new_id("ec")
                self.entity_clusters[cluster_id] = {"label": label, "person_id": None, "centroid": emb, "n": 0}
        c = self.entity_clusters[cluster_id]
        n = c["n"]
        c["centroid"] = [(cv * n + e) / (n + 1) for cv, e in zip(c["centroid"], emb)]
        c["n"] = n + 1
        vid = new_id("ev")
        self.entity_views[vid] = {"label": c["label"], "embedding": emb, "cluster_id": cluster_id,
                                  "quality": quality, "ts": ts, "snapshot_path": snapshot_path}
        return vid, cluster_id

    def match_entity(self, label, embedding, k=3):
        label = label.strip().lower()
        scored = [FaceMatch(cid, c["person_id"], cosine(embedding, c["centroid"]))
                  for cid, c in self.entity_clusters.items() if c["label"] == label]
        return sorted(scored, key=lambda m: m.score, reverse=True)[:k]

    def label_entity_cluster(self, cluster_id, person_id):
        self._need_person(person_id)
        self.entity_clusters[cluster_id]["person_id"] = person_id
        for sid, s in list(self.sightings.items()):
            if s.cluster_id == cluster_id and s.person_id is None:
                self.sightings[sid] = replace(s, person_id=person_id)

    def entity_cluster_size(self, cluster_id):
        return self.entity_clusters[cluster_id]["n"]

    # sightings / facts / relations
    def add_sighting(self, s):
        if s.person_id is None and s.cluster_id:
            c = self.clusters.get(s.cluster_id) or self.entity_clusters.get(s.cluster_id) or {}
            if c.get("person_id"):
                s = replace(s, person_id=c["person_id"])
        self.sightings[s.id] = s
        if s.person_id and s.person_id in self.people:
            p = self.people[s.person_id]
            self.people[s.person_id] = replace(p, last_seen=max(p.last_seen, s.ts))
        return s.id

    def recent_sightings(self, *, person_id=None, cluster_id=None, limit=20):
        rows = [s for s in self.sightings.values()
                if (person_id is None or s.person_id == person_id)
                and (cluster_id is None or s.cluster_id == cluster_id)]
        return sorted(rows, key=lambda s: s.ts, reverse=True)[:limit]

    def add_fact(self, person_id, key, value, *, source, confidence, ts=None):
        self._need_person(person_id)
        if source not in FACT_SOURCES:
            raise ValueError(f"bad source {source!r}")
        fid = new_id("fact")
        self._facts[fid] = Fact(fid, person_id, key, value, source, float(confidence),
                                time.time() if ts is None else ts)
        return fid

    def facts(self, person_id):
        return sorted((f for f in self._facts.values() if f.person_id == person_id), key=lambda f: f.ts)

    def relate(self, a_id, relation, b_id, *, source):
        self._need_person(a_id), self._need_person(b_id)
        if (a_id, relation, b_id) not in [(a, r, b) for a, r, b, _ in self._rels]:
            self._rels.append((a_id, relation, b_id, source))

    def relations(self, person_id):
        out = [(r, b, "out") for a, r, b, _ in self._rels if a == person_id]
        inn = [(r, a, "in") for a, r, b, _ in self._rels if b == person_id]
        return out + inn

    def person_context(self, person_id):
        p = self._need_person(person_id)
        return {
            "person": p,
            "facts": self.facts(person_id),
            "relations": self.relations(person_id),
            "recent_sightings": self.recent_sightings(person_id=person_id, limit=10),
            "clusters": [cid for cid, c in (*self.clusters.items(), *self.entity_clusters.items())
                         if c["person_id"] == person_id],
        }

    # doubts
    def create_doubt(self, d):
        now = time.time()
        if d.kind not in DOUBT_KINDS or d.status not in DOUBT_STATUSES:
            raise ValueError("bad doubt kind/status")
        d = replace(d, id=d.id or new_id("d"), created=d.created or now, updated=now)
        self.doubts[d.id] = d
        return d.id

    def get_doubt(self, doubt_id):
        return self.doubts.get(doubt_id)

    def update_doubt(self, doubt_id, **fields):
        if {"id", "updated"} & set(fields):
            raise TypeError("update_doubt: cannot set id/updated")
        if "status" in fields and fields["status"] not in DOUBT_STATUSES:
            raise ValueError("bad status")
        d = replace(self.doubts[doubt_id], **fields, updated=time.time())
        self.doubts[doubt_id] = d
        return d

    def list_doubts(self, status=None, limit=50):
        rows = [d for d in self.doubts.values() if status is None or d.status == status]
        return sorted(rows, key=lambda d: d.created)[:limit]

    def find_doubt_by_message(self, message_ref):
        return next((d for d in self.doubts.values() if d.message_ref == message_ref), None)

    def graph_read(self, cypher, params=None, limit=100):
        raise NotImplementedError("graph_read needs Neo4j")

    def _need_person(self, person_id):
        p = self.people.get(person_id)
        if p is None:
            raise KeyError(f"unknown person {person_id}")
        return p
