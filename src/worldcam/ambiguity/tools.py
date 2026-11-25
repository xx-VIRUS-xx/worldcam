"""Tool logic behind the worldcam MCP server (one method per MCP tool).

Kept free of the `mcp` SDK so tests call it directly. Every method validates
its arguments, enforces the per-run scope, and returns JSON-able data.

Scope: the server is started for exactly one doubt (`scope_doubt_id`). Doubt
tools accept only that id. Writes to existing people are allowed only for
people "in scope": ids in the doubt's about_ids / detail candidates, people the
worker created or touched for this doubt (persisted in
doubt.detail["touched_person_ids"] so follow-up rounds keep them).
label_face_cluster may target any existing person (that person then joins the
scope); merge_people only works on a `conflict` doubt about both people. This limits
what a prompt-injected worker could damage to the doubt it was asked about.

Outbox: ask_owner does not talk to Telegram. It writes the question into the
doubt itself (status 'asked', message_ref None, detail['outbox']={...}); the
engine process flushes it through the Notifier on its next tick. See
docs/ambiguity.md for why.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Optional, Sequence

from worldcam.graph.api import (ENTITY_KINDS, ENTITY_TYPES, FACT_SOURCES, PERSON_KINDS,
                                GraphStore)

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
REL_RE = re.compile(r"^[a-z][a-z_ ]{0,31}$")
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
WORKER_KINDS = tuple(k for k in PERSON_KINDS if k != "owner")   # worker may never create/promote an owner
WORKER_KINDS_BY_TYPE = {
    t: (WORKER_KINDS if t == "person" else ks) for t, ks in ENTITY_KINDS.items()
}
WRITE_CYPHER = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL)\b", re.I)
TOOL_NAMES = ("get_doubt", "person_context", "search_people", "graph_read", "upsert_person",
              "label_face_cluster", "add_fact", "relate", "set_persona_summary", "merge_people",
              "ask_owner", "resolve_doubt", "dismiss_doubt")


class ToolInputError(ValueError):
    """Bad arguments or an action outside this run's scope."""


def jsonable(x: Any) -> Any:
    if is_dataclass(x) and not isinstance(x, type):
        return jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k not in ("outbox",)}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def _id(v: Any, what: str) -> str:
    if not isinstance(v, str) or not ID_RE.match(v):
        raise ToolInputError(f"{what} must match {ID_RE.pattern}")
    return v


def _text(v: Any, what: str, max_len: int, *, allow_newlines: bool = False) -> str:
    if not isinstance(v, str):
        raise ToolInputError(f"{what} must be a string")
    v = v.strip()
    if not v or len(v) > max_len:
        raise ToolInputError(f"{what} must be 1..{max_len} chars")
    if CTRL_RE.search(v) or (not allow_newlines and "\n" in v):
        raise ToolInputError(f"{what} contains control characters")
    return v


def confine_image(path: Any, snapshot_dir: str) -> str:
    """Return the real path if it is an existing image file inside snapshot_dir."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ToolInputError("image path must be a non-empty string")
    root = os.path.realpath(snapshot_dir)
    real = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    if os.path.commonpath([root, real]) != root:
        raise ToolInputError(f"image path outside snapshot dir: {path}")
    if not real.lower().endswith(IMAGE_EXT) or not os.path.isfile(real):
        raise ToolInputError(f"not an existing image file under snapshot dir: {path}")
    return real


class WorldcamTools:
    def __init__(self, store: GraphStore, snapshot_dir: str, scope_doubt_id: str,
                 max_rounds: int = 3, clock=time.time):
        self.store = store
        self.snapshot_dir = snapshot_dir
        self.scope = _id(scope_doubt_id, "scope doubt id")
        self.max_rounds = max_rounds
        self.clock = clock

    # ---- scope helpers ----------------------------------------------------
    def _doubt(self, doubt_id: Any):
        doubt_id = _id(doubt_id, "doubt_id")
        if doubt_id != self.scope:
            raise ToolInputError(f"this run may only act on doubt {self.scope}")
        d = self.store.get_doubt(doubt_id)
        if d is None:
            raise ToolInputError(f"unknown doubt {doubt_id}")
        return d

    def _scope_people(self) -> set[str]:
        d = self.store.get_doubt(self.scope)
        if d is None:
            return set()
        ids = set(d.about_ids) | set(d.detail.get("touched_person_ids", ()))
        if d.detail.get("candidate_person_id"):
            ids.add(d.detail["candidate_person_id"])
        ids |= {c[0] for c in d.detail.get("candidates", ()) if c}
        return ids

    def _touch(self, person_id: str) -> None:
        d = self.store.get_doubt(self.scope)
        touched = list(d.detail.get("touched_person_ids", ()))
        if person_id not in touched:
            self.store.update_doubt(self.scope, detail=dict(d.detail, touched_person_ids=touched + [person_id]))

    def _person_in_scope(self, pid: Any, what: str = "person_id") -> str:
        pid = _id(pid, what)
        if self.store.get_person(pid) is None:
            raise ToolInputError(f"unknown person {pid}")
        if pid not in self._scope_people():
            raise ToolInputError(f"person {pid} is not part of doubt {self.scope}")
        return pid

    # ---- read tools -------------------------------------------------------
    def get_doubt(self, doubt_id: str) -> dict:
        d = self._doubt(doubt_id)
        out = jsonable(d)
        out["sighting_count"] = {cid: len(self.store.recent_sightings(cluster_id=cid, limit=50))
                                 for cid in d.about_ids if cid.startswith("fc")}
        sids = {i for i in d.about_ids if i.startswith("s_")}
        if sids:
            out["sightings"] = [
                jsonable(s) for s in self.store.recent_sightings(limit=100) if s.id in sids
            ]
        return out

    def person_context(self, person_id: str) -> dict:
        pid = _id(person_id, "person_id")
        try:
            return jsonable(self.store.person_context(pid))
        except KeyError:
            raise ToolInputError(f"unknown person {pid}")

    def search_people(self, text: str, limit: int = 10) -> list:
        text = _text(text, "text", 80)
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ToolInputError("limit must be 1..50")
        return jsonable(self.store.search_people(text, limit))

    def graph_read(self, cypher: str, params: Optional[dict] = None, limit: int = 50) -> list:
        cypher = _text(cypher, "cypher", 2000, allow_newlines=True)
        if WRITE_CYPHER.search(cypher):
            raise ToolInputError("graph_read is read-only (no CREATE/MERGE/SET/DELETE/REMOVE/DROP/CALL)")
        if params is not None and not isinstance(params, dict):
            raise ToolInputError("params must be an object")
        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ToolInputError("limit must be 1..200")
        try:
            return jsonable(self.store.graph_read(cypher, params or {}, limit))
        except NotImplementedError as e:
            raise ToolInputError(f"graph_read unavailable on this store: {e}")

    # ---- write tools ------------------------------------------------------
    def upsert_person(self, name: Optional[str] = None, kind: Optional[str] = None,
                      aliases: Sequence[str] = (), person_id: Optional[str] = None,
                      entity_type: Optional[str] = None, label: Optional[str] = None) -> dict:
        existing_type = "person"
        if person_id is not None:
            person_id = self._person_in_scope(person_id)
            existing = self.store.get_person(person_id)
            existing_type = existing.entity_type
            if existing.kind == "owner" and kind not in (None, "owner"):
                raise ToolInputError("cannot change the owner's kind")
        if entity_type is not None:
            if not isinstance(entity_type, str) or entity_type not in ENTITY_TYPES:
                raise ToolInputError(f"entity_type must be one of {ENTITY_TYPES}")
        eff_type = entity_type or existing_type
        if name is not None:
            name = _text(name, "name", 80)
        allowed_kinds = WORKER_KINDS_BY_TYPE[eff_type]
        if kind is not None:
            if eff_type == "person" and person_id is not None and self.store.get_person(person_id).kind == "owner":
                if kind != "owner":
                    raise ToolInputError("cannot change the owner's kind")
            elif kind not in allowed_kinds:
                raise ToolInputError(f"kind must be one of {allowed_kinds} for entity_type={eff_type!r}")
        if label is not None:
            label = _text(label, "label", 40)
        if not isinstance(aliases, (list, tuple)) or len(aliases) > 10:
            raise ToolInputError("aliases must be a list of at most 10 strings")
        aliases = tuple(_text(a, "alias", 80) for a in aliases)
        if person_id is None and not name:
            raise ToolInputError("a new entity needs a name")
        pid = self.store.upsert_person(person_id, name=name, kind=kind, aliases=aliases,
                                       ts=self.clock(), entity_type=entity_type, label=label)
        self._touch(pid)
        d = self.store.get_doubt(self.scope)
        if d is not None and d.kind == "unknown_entity":
            sids = {i for i in d.about_ids if i.startswith("s_")}
            if sids:
                for s in self.store.recent_sightings(limit=100):
                    if s.id in sids and s.person_id is None:
                        self.store.add_sighting(replace(s, person_id=pid))
        return {"person_id": pid, "entity_type": eff_type}

    def label_face_cluster(self, cluster_id: str, person_id: str) -> dict:
        cid = _id(cluster_id, "cluster_id")
        d = self.store.get_doubt(self.scope)
        if cid not in d.about_ids:
            raise ToolInputError(f"cluster {cid} is not part of doubt {self.scope}")
        # Any existing person may be the target (the owner may name someone already
        # known); that person then joins this doubt's scope.
        pid = _id(person_id, "person_id")
        if self.store.get_person(pid) is None:
            raise ToolInputError(f"unknown person {pid}")
        try:
            self.store.label_face_cluster(cid, pid)
        except KeyError as e:
            raise ToolInputError(f"unknown cluster/person: {e}")
        self._touch(pid)
        return {"ok": True}

    def add_fact(self, person_id: str, key: str, value: str, source: str, confidence: float) -> dict:
        pid = self._person_in_scope(person_id)
        if not isinstance(key, str) or not KEY_RE.match(key):
            raise ToolInputError(f"key must match {KEY_RE.pattern}")
        value = _text(value, "value", 300)
        if source not in FACT_SOURCES:
            raise ToolInputError(f"source must be one of {FACT_SOURCES}")
        if source == "owner" and not self.store.get_doubt(self.scope).answer:
            raise ToolInputError("source='owner' needs an owner answer on this doubt")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise ToolInputError("confidence must be a number in [0, 1]")
        fid = self.store.add_fact(pid, key, value, source=source, confidence=float(confidence), ts=self.clock())
        return {"fact_id": fid}

    def relate(self, a_id: str, relation: str, b_id: str, source: str) -> dict:
        a, b = _id(a_id, "a_id"), _id(b_id, "b_id")
        if not isinstance(relation, str) or not REL_RE.match(relation):
            raise ToolInputError(f"relation must match {REL_RE.pattern}")
        if source not in FACT_SOURCES:
            raise ToolInputError(f"source must be one of {FACT_SOURCES}")
        if a == b:
            raise ToolInputError("cannot relate a person to themselves")
        for x in (a, b):
            if self.store.get_person(x) is None:
                raise ToolInputError(f"unknown person {x}")
        scope = self._scope_people()
        if a not in scope and b not in scope:
            raise ToolInputError(f"neither person is part of doubt {self.scope}")
        self.store.relate(a, relation, b, source=source)
        return {"ok": True}

    def set_persona_summary(self, person_id: str, summary: str) -> dict:
        pid = self._person_in_scope(person_id)
        self.store.set_persona_summary(pid, _text(summary, "summary", 1000, allow_newlines=True))
        return {"ok": True}

    def merge_people(self, keep_id: str, drop_id: str) -> dict:
        d = self.store.get_doubt(self.scope)
        keep, drop = _id(keep_id, "keep_id"), _id(drop_id, "drop_id")
        if d.kind != "conflict" or keep not in d.about_ids or drop not in d.about_ids:
            raise ToolInputError("merge_people is only allowed for a conflict doubt about both people")
        for x in (keep, drop):
            if self.store.get_person(x) is None:
                raise ToolInputError(f"unknown person {x}")
        if keep == drop:
            raise ToolInputError("keep_id and drop_id are the same")
        if self.store.get_person(drop).kind == "owner":
            raise ToolInputError("cannot merge the owner away")
        self.store.merge_people(keep, drop)
        return {"ok": True}

    # ---- doubt tools ------------------------------------------------------
    def ask_owner(self, doubt_id: str, text: str, choices: Optional[Sequence[Any]] = None,
                  image_paths: Optional[Sequence[str]] = None) -> dict:
        d = self._doubt(doubt_id)
        if d.status not in ("open", "answered"):
            raise ToolInputError(f"cannot ask in status {d.status}")
        if d.rounds >= self.max_rounds:
            raise ToolInputError(f"max {self.max_rounds} rounds reached; resolve or dismiss instead")
        text = _text(text, "text", 800, allow_newlines=True)
        norm_choices = []
        for c in choices or ():
            if isinstance(c, str):
                c = {"label": c, "value": c}
            if not isinstance(c, dict):
                raise ToolInputError("each choice is a string or {label, value}")
            label = _text(c.get("label"), "choice label", 40)
            value = _text(c.get("value", label), "choice value", 64)
            if len(value.encode()) > 64:
                raise ToolInputError("choice value must be <= 64 bytes")
            norm_choices.append({"label": label, "value": value})
        if len(norm_choices) > 8:
            raise ToolInputError("at most 8 choices")
        paths = d.snapshot_paths[:2] if image_paths is None else image_paths
        if not isinstance(paths, (list, tuple)) or len(paths) > 2:
            raise ToolInputError("at most 2 image paths")
        real = [confine_image(p, self.snapshot_dir) for p in paths]
        outbox = {"text": text, "choices": norm_choices, "image_paths": real, "queued_at": self.clock()}
        history = list(d.detail.get("history", ()))
        if d.question and d.answer:
            history.append({"question": d.question, "answer": d.answer})
        self.store.update_doubt(d.id, status="asked", question=text, rounds=d.rounds + 1,
                                message_ref=None, answer=None,
                                detail=dict(d.detail, outbox=outbox, history=history))
        return {"ok": True, "status": "asked", "rounds": d.rounds + 1,
                "note": "queued; the engine delivers it to the owner. End your run now."}

    def resolve_doubt(self, doubt_id: str, summary: str) -> dict:
        d = self._doubt(doubt_id)
        if d.status != "answered":
            raise ToolInputError("only an answered doubt can be resolved")
        summary = _text(summary, "summary", 500, allow_newlines=True)
        self.store.update_doubt(d.id, status="resolved", detail=dict(d.detail, resolution=summary))
        return {"ok": True, "status": "resolved"}

    def dismiss_doubt(self, doubt_id: str, reason: str) -> dict:
        d = self._doubt(doubt_id)
        if d.status not in ("open", "answered"):
            raise ToolInputError(f"cannot dismiss in status {d.status}")
        reason = _text(reason, "reason", 300)
        self.store.update_doubt(d.id, status="dismissed",
                                detail=dict(d.detail, dismissed_reason=reason, dismissed_at=self.clock()))
        return {"ok": True, "status": "dismissed"}

    def toolset(self, factory_args: Optional[dict] = None):
        """ToolSet over this object. Pass `factory_args` (JSON-able, no secrets) when an
        out-of-process runner must rebuild it through build_toolset()."""
        from worldcam.models.api import ToolSet, ToolSpec
        specs = tuple(ToolSpec(n, desc, {"type": "object", "properties": props, "required": list(req)},
                               _tool_fn(getattr(self, n)), writes=w)
                      for n, (desc, props, req, w) in TOOL_META.items())
        return ToolSet(specs, factory=TOOLSET_FACTORY if factory_args is not None else None,
                       factory_args=factory_args or {})


# ---- ToolSet metadata (one entry per tool: description, properties, required, writes) ----
_S, _N = {"type": "string"}, {"type": "number"}
_CHOICES = {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}}
TOOL_META: dict[str, tuple[str, dict, tuple, bool]] = {
    "get_doubt": ("Read the doubt this run is about: kind, status, question, owner answer, history, about_ids.",
                  {"doubt_id": _S}, ("doubt_id",), False),
    "person_context": ("Person, facts, relations, recent sightings and face clusters for one person.",
                       {"person_id": _S}, ("person_id",), False),
    "search_people": ("Find people by name or alias (substring, case-insensitive).",
                      {"text": _S, "limit": {"type": "integer"}}, ("text",), False),
    "graph_read": ("Parameterised read-only Cypher (READ transaction). Not available on the test file store.",
                   {"cypher": _S, "params": {"type": "object"}, "limit": {"type": "integer"}}, ("cypher",), False),
    "upsert_person": ("Create or update an entity (person, animal, object, other) in this doubt's scope. "
                      "entity_type: person|animal|object|other. kind depends on entity_type (person: "
                      "household|visitor|unknown; animal: pet|stray|wildlife|visitor|unknown; object: "
                      "belonging|device|furniture|package|vehicle|unknown). label: short class tag like "
                      "'dog', 'backpack'.",
                      {"name": _S, "kind": _S, "aliases": {"type": "array", "items": _S}, "person_id": _S,
                       "entity_type": _S, "label": _S},
                      (), True),
    "label_face_cluster": ("Attach a face cluster from this doubt to a person.",
                           {"cluster_id": _S, "person_id": _S}, ("cluster_id", "person_id"), True),
    "add_fact": ("Add a fact. key: snake_case (e.g. relation_to_owner, works_at, visit_pattern). "
                 "source: owner (the owner said it) | observed (camera) | inferred (your guess). confidence 0..1.",
                 {"person_id": _S, "key": _S, "value": _S, "source": _S, "confidence": _N},
                 ("person_id", "key", "value", "source", "confidence"), True),
    "relate": ("Add a relation edge a -[relation]-> b (e.g. colleague, partner, friend).",
               {"a_id": _S, "relation": _S, "b_id": _S, "source": _S}, ("a_id", "relation", "b_id", "source"), True),
    "set_persona_summary": ("Replace the short persona summary of a person in scope.",
                            {"person_id": _S, "summary": _S}, ("person_id", "summary"), True),
    "merge_people": ("Merge drop_id into keep_id (both must be part of this doubt).",
                     {"keep_id": _S, "drop_id": _S}, ("keep_id", "drop_id"), True),
    "ask_owner": ("Queue one question to the owner (sent by the engine). choices: strings or {label, value}; "
                  "image_paths default to the doubt's crops and must be under the snapshot dir. Max 3 rounds.",
                  {"doubt_id": _S, "text": _S, "choices": _CHOICES, "image_paths": {"type": "array", "items": _S}},
                  ("doubt_id", "text"), True),
    "resolve_doubt": ("Mark an answered doubt resolved after writing what the answer taught you.",
                      {"doubt_id": _S, "summary": _S}, ("doubt_id", "summary"), True),
    "dismiss_doubt": ("Dismiss the doubt (nothing useful to ask or learn).",
                      {"doubt_id": _S, "reason": _S}, ("doubt_id", "reason"), True),
}
assert tuple(TOOL_META) == TOOL_NAMES
TOOLSET_FACTORY = "worldcam.ambiguity.tools:build_toolset"


def _tool_fn(method):
    def fn(args: dict):
        if not isinstance(args, dict):
            raise ToolInputError("arguments must be an object")
        try:
            return method(**args)
        except TypeError as e:           # unknown / missing argument names
            raise ToolInputError(f"bad arguments: {e}") from None
    return fn


def build_toolset(store: str, snapshot_dir: str, doubt_id: str, max_rounds: int = 3):
    """Factory ref `worldcam.ambiguity.tools:build_toolset`: the per-doubt ToolSet.
    `store` is 'neo4j' or 'file:<path>' (see file_store.open_store)."""
    from worldcam.ambiguity.file_store import open_store
    tools = WorldcamTools(open_store(store), snapshot_dir, doubt_id, max_rounds=max_rounds)
    return tools.toolset({"store": store, "snapshot_dir": snapshot_dir, "doubt_id": doubt_id,
                          "max_rounds": max_rounds})
