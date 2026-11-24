"""Neo4j connection settings from `.env` (git-ignored) and the environment.

Real environment variables win over `.env`. The `.env` file is looked up in
the current directory and then each parent, so it works from the repo root,
a subdirectory, or a git worktree under `.claude/worktrees/`.
Never log the password: `Neo4jConfig.__repr__` redacts it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def find_env_file(start: Optional[Path] = None, name: str = ".env") -> Optional[Path]:
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        p = d / name
        if p.is_file():
            return p
    return None


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser: `#` comments, optional `export`, quotes."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if val[:1] in ("'", '"'):
            q = val[0]
            end = val.find(q, 1)
            val = val[1:end] if end > 0 else val[1:]
        else:
            hash_at = val.find(" #")
            if hash_at >= 0:
                val = val[:hash_at]
            val = val.strip()
        out[key] = val
    return out


def load_env(path: Optional[Path] = None) -> dict[str, str]:
    """Merged view: .env values overridden by os.environ (os.environ not modified)."""
    p = path or find_env_file()
    vals = parse_env_file(p) if p else {}
    vals.update({k: v for k, v in os.environ.items()})
    return vals


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    def __repr__(self) -> str:  # never print the secret
        return (f"Neo4jConfig(uri={self.uri!r}, user={self.user!r}, "
                f"password={'***' if self.password else ''!r}, database={self.database!r})")

    __str__ = __repr__

    @classmethod
    def from_env(cls, path: Optional[Path] = None) -> "Neo4jConfig":
        env = load_env(path)
        return cls(uri=env.get("NEO4J_URI") or cls.uri,
                   user=env.get("NEO4J_USER") or cls.user,
                   password=env.get("NEO4J_PASSWORD", ""),
                   database=env.get("NEO4J_DATABASE") or cls.database)
