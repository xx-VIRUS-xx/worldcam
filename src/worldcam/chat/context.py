"""Rebuild the agent's MemoryContext inside another process.

The `claude -p` backend runs chat tools in a separate MCP server process, which must
reconstruct the same stores. `worldcam-run` registers this factory (plus absolute paths)
in `ctx.extras`; `chat.tools.build_memory_toolset_from_args` calls it there. Trusted
config only: the model never supplies these arguments.
"""
from __future__ import annotations

from ..agent.api import MemoryContext


def build_context(store: str = "neo4j", events_db: str = "captures/brain.db",
                  users_db: str = "captures/users.db", snapshot_dir: str = "captures/snapshots") -> MemoryContext:
    from ..ambiguity.file_store import open_store
    from ..brain.store import SqliteEventStore
    from ..users.store import open_directory

    events = SqliteEventStore(events_db)
    return MemoryContext(graph=open_store(store), events=events, routines=events,
                         users=open_directory(users_db, bootstrap=False),
                         extras={"snapshot_dir": snapshot_dir})
