"""SQLite-backed UserDirectory (``captures/users.db``) with invite codes.

* WAL, file mode 0600, one connection guarded by an RLock (thread-safe), a
  ``busy_timeout`` so the admin CLI can run next to the live process.
* Schema is versioned (``meta.schema_version``); a newer database is refused.
* Behaviour matches ``InMemoryUserDirectory`` (shared contract test), plus:
  - the last active owner can never be deactivated or demoted (``LastOwnerError``);
  - invites: short-lived, single-use codes bound to a role/profiles, stored only as
    a SHA-256 hash. The plaintext code is returned once by ``create_invite``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence

from .api import PERMISSIONS, PROFILES, ROLES, User, can

SCHEMA_VERSION = 1
DEFAULT_DB = "captures/users.db"
INVITE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no look-alikes (0/O, 1/I/L)
INVITE_LEN = 10
MAX_INVITE_TTL_S = 14 * 24 * 3600


class LastOwnerError(ValueError):
    """Refused: it would leave the directory without an active owner."""


class InviteError(ValueError):
    """Invite code unknown, expired or already used (deliberately not told apart)."""


def _hash_code(code: str) -> str:
    norm = "".join(c for c in code.upper() if c.isalnum())
    return hashlib.sha256(norm.encode()).hexdigest()


def _check_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    for p in profiles:
        if p not in PROFILES:
            raise ValueError(f"bad profile {p!r}")
    return tuple(dict.fromkeys(profiles))


def _check_perm(permission: str) -> None:
    if permission not in PERMISSIONS:
        raise ValueError(f"bad permission {permission!r}")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, person_id TEXT,
  profiles TEXT NOT NULL DEFAULT '[]', perm_extra TEXT NOT NULL DEFAULT '[]',
  perm_denied TEXT NOT NULL DEFAULT '[]', active INTEGER NOT NULL DEFAULT 1,
  created REAL NOT NULL, seq INTEGER);
CREATE TABLE IF NOT EXISTS user_channels(
  user_id TEXT NOT NULL REFERENCES users(id), channel TEXT NOT NULL, sender_id TEXT NOT NULL,
  PRIMARY KEY(user_id, channel));
CREATE INDEX IF NOT EXISTS ix_user_channels_lookup ON user_channels(channel, sender_id);
CREATE TABLE IF NOT EXISTS invites(
  code_hash TEXT PRIMARY KEY, role TEXT NOT NULL, profiles TEXT NOT NULL,
  created REAL NOT NULL, expires REAL NOT NULL, used_at REAL, used_by TEXT);
"""


class SqliteUserDirectory:
    def __init__(self, path: str | os.PathLike = DEFAULT_DB, *,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            os.close(fd)
            os.chmod(self.path, 0o600)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None,
                                   timeout=10.0)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA busy_timeout=10000")
            self._migrate()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __repr__(self) -> str:
        return f"SqliteUserDirectory({str(self.path)!r})"

    # ------------------------------------------------------------ plumbing ---
    @contextlib.contextmanager
    def _tx(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    def _migrate(self) -> None:
        with self._tx() as db:
            for stmt in filter(None, (s.strip() for s in _SCHEMA.split(";"))):
                db.execute(stmt)
            row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(f"users.db schema {row['value']} is newer than this code "
                                   f"({SCHEMA_VERSION}); upgrade worldcam")

    def _user(self, db, row) -> User:
        chans = {r["channel"]: r["sender_id"] for r in db.execute(
            "SELECT channel, sender_id FROM user_channels WHERE user_id=?", (row["id"],))}
        return User(row["id"], row["name"], row["role"], chans, row["person_id"],
                    tuple(json.loads(row["profiles"])),
                    frozenset(json.loads(row["perm_extra"])),
                    frozenset(json.loads(row["perm_denied"])),
                    bool(row["active"]), row["created"])

    def _row(self, db, user_id):
        row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if row is None:
            raise KeyError(user_id)
        return row

    def _other_active_owners(self, db, user_id: str) -> int:
        return db.execute("SELECT COUNT(*) FROM users WHERE role='owner' AND active=1 AND id<>?",
                          (user_id,)).fetchone()[0]

    # ----------------------------------------------------------- directory ---
    def _add(self, db, name, role, channel, sender_id, profiles, person_id) -> User:
        if role not in ROLES:
            raise ValueError(f"bad role {role!r}")
        profiles = _check_profiles(profiles)
        sender_id = str(sender_id)
        if db.execute("SELECT 1 FROM user_channels c JOIN users u ON u.id=c.user_id "
                      "WHERE c.channel=? AND c.sender_id=? AND u.active=1",
                      (channel, sender_id)).fetchone():
            raise ValueError("already whitelisted")
        uid = uuid.uuid4().hex[:10]
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM users").fetchone()[0]
        db.execute("INSERT INTO users(id,name,role,person_id,profiles,created,seq) "
                   "VALUES(?,?,?,?,?,?,?)",
                   (uid, name, role, person_id, json.dumps(profiles), self._clock(), seq))
        db.execute("INSERT INTO user_channels VALUES(?,?,?)", (uid, channel, sender_id))
        return self._user(db, self._row(db, uid))

    def add(self, name, role, channel, sender_id, *, profiles=(), person_id=None) -> User:
        with self._tx() as db:
            return self._add(db, name, role, channel, sender_id, profiles, person_id)

    def get(self, user_id) -> Optional[User]:
        with self._lock:
            row = self._db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return self._user(self._db, row) if row else None

    def by_channel(self, channel, sender_id) -> Optional[User]:
        with self._lock:
            row = self._db.execute(
                "SELECT u.* FROM users u JOIN user_channels c ON c.user_id=u.id "
                "WHERE c.channel=? AND c.sender_id=? AND u.active=1 ORDER BY u.seq LIMIT 1",
                (channel, str(sender_id))).fetchone()
            return self._user(self._db, row) if row else None

    def list(self, include_inactive=False) -> list[User]:
        with self._lock:
            q = "SELECT * FROM users" + ("" if include_inactive else " WHERE active=1") + " ORDER BY seq"
            return [self._user(self._db, r) for r in self._db.execute(q).fetchall()]

    def _set(self, db, user_id, **cols) -> User:
        sets = ", ".join(f"{k}=?" for k in cols)
        db.execute(f"UPDATE users SET {sets} WHERE id=?", (*cols.values(), user_id))
        return self._user(db, self._row(db, user_id))

    def set_role(self, user_id, role) -> User:
        if role not in ROLES:
            raise ValueError(f"bad role {role!r}")
        with self._tx() as db:
            row = self._row(db, user_id)
            if (row["role"] == "owner" and role != "owner" and row["active"]
                    and self._other_active_owners(db, user_id) == 0):
                raise LastOwnerError("cannot demote the last active owner")
            return self._set(db, user_id, role=role)

    def set_profiles(self, user_id, profiles) -> User:
        profiles = _check_profiles(profiles)
        with self._tx() as db:
            self._row(db, user_id)
            return self._set(db, user_id, profiles=json.dumps(profiles))

    def set_person(self, user_id, person_id) -> User:
        with self._tx() as db:
            self._row(db, user_id)
            return self._set(db, user_id, person_id=person_id)

    def grant(self, user_id, permission) -> User:
        _check_perm(permission)
        with self._tx() as db:
            row = self._row(db, user_id)
            extra = set(json.loads(row["perm_extra"])) | {permission}
            denied = set(json.loads(row["perm_denied"])) - {permission}
            return self._set(db, user_id, perm_extra=json.dumps(sorted(extra)),
                             perm_denied=json.dumps(sorted(denied)))

    def revoke(self, user_id, permission) -> User:
        _check_perm(permission)
        with self._tx() as db:
            row = self._row(db, user_id)
            if row["role"] == "owner" and permission in ("manage_users", "chat"):
                raise LastOwnerError(f"cannot revoke {permission} from an owner")
            extra = set(json.loads(row["perm_extra"])) - {permission}
            denied = set(json.loads(row["perm_denied"])) | {permission}
            return self._set(db, user_id, perm_extra=json.dumps(sorted(extra)),
                             perm_denied=json.dumps(sorted(denied)))

    def deactivate(self, user_id) -> None:
        with self._tx() as db:
            row = self._row(db, user_id)
            if (row["role"] == "owner" and row["active"]
                    and self._other_active_owners(db, user_id) == 0):
                raise LastOwnerError("cannot deactivate the last active owner")
            self._set(db, user_id, active=0)

    def with_permission(self, permission, profile=None) -> list[User]:
        return [u for u in self.list() if can(u, permission)
                and (profile is None or profile in u.profiles)]

    # -------------------------------------------------------------- invites ---
    def create_invite(self, role: str, profiles: Sequence[str] = (),
                      ttl: float = 24 * 3600) -> str:
        """Returns the plaintext code (shown once; only its hash is stored).
        An invite can never create an owner."""
        if role not in ROLES or role == "owner":
            raise ValueError(f"invites cannot grant role {role!r}")
        profiles = _check_profiles(profiles)
        if not 0 < ttl <= MAX_INVITE_TTL_S:
            raise ValueError("ttl must be between 1 second and 14 days")
        code = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(INVITE_LEN))
        now = self._clock()
        with self._tx() as db:
            db.execute("DELETE FROM invites WHERE expires < ?", (now - 7 * 24 * 3600,))
            db.execute("INSERT INTO invites(code_hash,role,profiles,created,expires) "
                       "VALUES(?,?,?,?,?)", (_hash_code(code), role, json.dumps(profiles),
                                             now, now + ttl))
        return code

    def redeem(self, code: str, channel: str, sender_id: str, name: str) -> User:
        """Atomically consume a valid invite and whitelist the sender. Raises
        InviteError for unknown/expired/used codes (not distinguished)."""
        if not isinstance(code, str) or not (4 <= len(code) <= 64):
            raise InviteError("invalid or expired invite")
        now = self._clock()
        with self._tx() as db:
            row = db.execute("SELECT * FROM invites WHERE code_hash=?",
                             (_hash_code(code),)).fetchone()
            if row is None or row["used_at"] is not None or row["expires"] < now:
                raise InviteError("invalid or expired invite")
            user = self._add(db, (name or "").strip()[:80] or "guest", row["role"], channel,
                             sender_id, json.loads(row["profiles"]), None)
            db.execute("UPDATE invites SET used_at=?, used_by=? WHERE code_hash=?",
                       (now, user.id, row["code_hash"]))
            return user

    def pending_invites(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM invites WHERE used_at IS NULL AND "
                                    "expires>=?", (self._clock(),)).fetchone()[0]


def bootstrap_owner(directory, owner_chat_id: Optional[str], *, name: str = "Owner",
                    channel: str = "telegram") -> Optional[User]:
    """If the directory has no users at all and an owner chat id is configured,
    create the owner (subscribed to every profile). Idempotent."""
    if not owner_chat_id or not str(owner_chat_id).strip():
        return None
    if directory.list(include_inactive=True):
        return None
    return directory.add(name, "owner", channel, str(owner_chat_id).strip(), profiles=PROFILES)


def open_directory(path: str | os.PathLike = DEFAULT_DB, *, env_path: str | os.PathLike = ".env",
                   bootstrap: bool = True) -> SqliteUserDirectory:
    """Open the SQLite directory and bootstrap the owner from TELEGRAM_OWNER_CHAT_ID."""
    d = SqliteUserDirectory(path)
    if bootstrap:
        from ..messenger.telegram import get_setting
        bootstrap_owner(d, get_setting("TELEGRAM_OWNER_CHAT_ID", env_path))
    return d
