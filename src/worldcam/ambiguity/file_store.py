"""Test-only, file-backed GraphStore: InMemoryGraphStore pickled to one file.

Used so the engine process and the MCP server process (spawned by `claude`)
can share a graph in smoke tests without Neo4j. Every call takes an exclusive
fcntl lock, reloads the file, runs the InMemoryGraphStore method and writes the
file back atomically. Slow and single-host by design. Never use it in production.

pickle is acceptable here because the file is created and read only by this
code on the owner's own machine; do not point it at a file from elsewhere.
"""
from __future__ import annotations

import fcntl
import os
import pickle
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator

from worldcam.graph.api import InMemoryGraphStore

_READ_ONLY = {
    "get_person", "search_people", "match_face", "cluster_size", "recent_sightings",
    "facts", "relations", "person_context", "get_doubt", "list_doubts",
    "find_doubt_by_message", "graph_read",
}
_MUTATING = {
    "upsert_person", "set_persona_summary", "merge_people", "forget_person", "add_face",
    "label_face_cluster", "add_sighting", "add_fact", "relate", "create_doubt",
    "update_doubt",
}


class FileGraphStore:
    def __init__(self, path: str, cluster_threshold: float = 0.45):
        self.path = os.path.abspath(path)
        self._lock_path = self.path + ".lock"
        self._threshold = cluster_threshold
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with self._locked():
                if not os.path.exists(self.path):
                    self._save(InMemoryGraphStore(cluster_threshold))

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with open(self._lock_path, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load(self) -> InMemoryGraphStore:
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def _save(self, mem: InMemoryGraphStore) -> None:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix=".graph.")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(mem, f)
        os.replace(tmp, self.path)

    def snapshot(self) -> InMemoryGraphStore:
        """Return a detached copy of the whole graph (for tests and reports)."""
        with self._locked():
            return self._load()

    def _call(self, _method: str, /, *args: Any, **kwargs: Any) -> Any:
        with self._locked():
            mem = self._load()
            out = getattr(mem, _method)(*args, **kwargs)
            if _method in _MUTATING:
                self._save(mem)
            return out

    def __getattr__(self, name: str) -> Any:
        if name in _READ_ONLY or name in _MUTATING:
            return lambda *a, **kw: self._call(name, *a, **kw)
        raise AttributeError(name)


def open_store(spec: str):
    """`neo4j` -> worldcam.graph.store (imported lazily), `file:<path>` -> FileGraphStore."""
    if spec == "neo4j":
        from worldcam.graph import store as neo  # lazy: needs the [graph] extra and a server
        factory = getattr(neo, "from_env", None) or getattr(neo, "Neo4jGraphStore")
        return factory()
    if spec.startswith("file:") and len(spec) > 5:
        return FileGraphStore(spec[5:])
    raise ValueError(f"unknown store spec {spec!r} (use 'neo4j' or 'file:<path>')")
