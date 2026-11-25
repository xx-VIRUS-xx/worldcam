"""UserDirectory contract (InMemory + SQLite parity), invites, owner protection, CLI."""
from __future__ import annotations

import os
import stat
import threading

import pytest

from worldcam.users.api import InMemoryUserDirectory, can, can_confirm
from worldcam.users.store import (
    InviteError, LastOwnerError, SqliteUserDirectory, bootstrap_owner, open_directory,
)


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture(params=["memory", "sqlite"])
def d(request, tmp_path):
    if request.param == "memory":
        return InMemoryUserDirectory()
    return SqliteUserDirectory(tmp_path / "users.db")


def test_contract_add_get_lookup(d):
    u = d.add("Ann", "household", "telegram", 111, profiles=["cctv"])
    assert d.get(u.id) == u and u.channel_ids == {"telegram": "111"}
    assert d.by_channel("telegram", "111").id == u.id
    assert d.by_channel("telegram", 111).id == u.id
    assert d.by_channel("telegram", "999") is None and d.by_channel("voice", "111") is None
    assert d.get("nope") is None
    with pytest.raises(ValueError):
        d.add("Dup", "guest", "telegram", "111")
    with pytest.raises(ValueError):
        d.add("Bad", "king", "telegram", "5")


def test_contract_roles_profiles_permissions(d):
    u = d.add("Ann", "guest", "telegram", "1")
    assert not can_confirm(d.get(u.id), "identity")
    u = d.set_role(u.id, "household")
    assert can_confirm(u, "identity") and u.role == "household"
    u = d.set_profiles(u.id, ["companion", "cctv"])
    assert set(u.profiles) == {"companion", "cctv"}
    u = d.grant(u.id, "answer_medical")
    assert can(u, "answer_medical") and can_confirm(u, "medical")
    u = d.revoke(u.id, "answer_medical")
    assert not can(u, "answer_medical")
    u = d.revoke(u.id, "answer_identity")
    assert not can_confirm(u, "identity")
    u = d.grant(u.id, "answer_identity")
    assert can_confirm(u, "identity")
    with pytest.raises(KeyError):
        d.set_role("nope", "guest")


def test_contract_deactivate_and_listing(d):
    a = d.add("A", "household", "telegram", "1", profiles=["medical"])
    b = d.add("B", "caregiver", "telegram", "2", profiles=["medical"])
    assert [u.id for u in d.list()] == [a.id, b.id]
    d.deactivate(b.id)
    assert d.by_channel("telegram", "2") is None
    assert [u.id for u in d.list()] == [a.id]
    assert {u.id for u in d.list(include_inactive=True)} == {a.id, b.id}
    assert d.get(b.id).active is False
    # id of a deactivated user can be re-added (new record)
    assert d.add("B2", "guest", "telegram", "2").name == "B2"


def test_contract_with_permission(d):
    d.add("A", "household", "telegram", "1", profiles=["cctv"])
    c = d.add("C", "caregiver", "telegram", "3", profiles=["medical"])
    d.add("G", "guest", "telegram", "4")
    assert [u.id for u in d.with_permission("receive_medical_alerts")] == [c.id]
    assert d.with_permission("receive_medical_alerts", profile="cctv") == []
    assert {u.name for u in d.with_permission("chat")} == {"A", "C", "G"}


# ------------------------------------------------------------ sqlite only ---
@pytest.fixture
def s(tmp_path):
    clock = Clock()
    db = SqliteUserDirectory(tmp_path / "users.db", clock=clock)
    db.clock = clock
    return db


def test_file_mode_wal_and_persistence(tmp_path):
    p = tmp_path / "sub" / "users.db"
    db = SqliteUserDirectory(p)
    u = db.add("Ann", "owner", "telegram", "1", profiles=["cctv"])
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert db._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.close()
    again = SqliteUserDirectory(p)
    assert again.get(u.id) == u


def test_schema_version_refused_when_newer(tmp_path):
    db = SqliteUserDirectory(tmp_path / "u.db")
    db._db.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    db.close()
    with pytest.raises(RuntimeError):
        SqliteUserDirectory(tmp_path / "u.db")


def test_last_owner_protected(s):
    o = s.add("Boss", "owner", "telegram", "1")
    with pytest.raises(LastOwnerError):
        s.deactivate(o.id)
    with pytest.raises(LastOwnerError):
        s.set_role(o.id, "guest")
    with pytest.raises(LastOwnerError):
        s.revoke(o.id, "manage_users")
    assert s.get(o.id).active and s.get(o.id).role == "owner"
    o2 = s.add("Co", "owner", "telegram", "2")
    s.set_role(o.id, "household")           # allowed: another owner remains
    with pytest.raises(LastOwnerError):
        s.deactivate(o2.id)


def test_invite_roundtrip_single_use_and_hashed(s):
    code = s.create_invite("caregiver", ["medical"], ttl=3600)
    rows = s._db.execute("SELECT * FROM invites").fetchall()
    assert len(rows) == 1 and code not in "".join(str(tuple(r)) for r in rows)
    u = s.redeem(code.lower(), "telegram", "555", "Nurse Joy")   # case/format tolerant
    assert (u.role, u.profiles, u.name) == ("caregiver", ("medical",), "Nurse Joy")
    assert s.by_channel("telegram", "555").id == u.id
    with pytest.raises(InviteError):
        s.redeem(code, "telegram", "556", "Other")                # single use
    assert s.by_channel("telegram", "556") is None


def test_invite_expiry_wrong_code_and_no_owner(s):
    code = s.create_invite("guest", ttl=60)
    with pytest.raises(InviteError):
        s.redeem("WRONGCODE1", "telegram", "5", "x")
    s.clock.t += 61
    with pytest.raises(InviteError):
        s.redeem(code, "telegram", "5", "x")
    assert s.by_channel("telegram", "5") is None
    with pytest.raises(ValueError):
        s.create_invite("owner")
    with pytest.raises(ValueError):
        s.create_invite("guest", ttl=0)
    with pytest.raises(ValueError):
        s.create_invite("guest", profiles=["nope"])
    with pytest.raises(InviteError):
        s.redeem("", "telegram", "5", "x")


def test_invite_not_consumed_when_sender_already_whitelisted(s):
    s.add("Ann", "guest", "telegram", "7")
    code = s.create_invite("household")
    with pytest.raises(ValueError):
        s.redeem(code, "telegram", "7", "Ann")
    assert s.pending_invites() == 1   # rolled back, still usable


def test_bootstrap_owner(tmp_path, monkeypatch):
    db = SqliteUserDirectory(tmp_path / "u.db")
    assert bootstrap_owner(db, None) is None and db.list() == []
    o = bootstrap_owner(db, " 4242 ")
    assert o.role == "owner" and o.channel_ids == {"telegram": "4242"}
    assert set(o.profiles) == {"cctv", "companion", "medical"}
    assert bootstrap_owner(db, "9999") is None and len(db.list()) == 1   # only when empty
    monkeypatch.setenv("TELEGRAM_OWNER_CHAT_ID", "77")
    d2 = open_directory(tmp_path / "u2.db", env_path=tmp_path / "none.env")
    assert d2.by_channel("telegram", "77").role == "owner"


def test_concurrency_smoke(s):
    s.add("Boss", "owner", "telegram", "0")
    errors = []

    def worker(n):
        try:
            for i in range(25):
                u = s.add(f"u{n}-{i}", "guest", "telegram", f"{n}{i:03d}")
                s.grant(u.id, "view_images")
                s.by_channel("telegram", f"{n}{i:03d}")
                s.list()
        except Exception as e:   # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(n,)) for n in range(1, 7)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not errors and len(s.list()) == 151

    # racing redemptions of one code: exactly one wins
    code = s.create_invite("guest")
    wins = []

    def redeem(n):
        try:
            wins.append(s.redeem(code, "telegram", f"r{n}", "x"))
        except InviteError:
            pass

    ts = [threading.Thread(target=redeem, args=(n,)) for n in range(8)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert len(wins) == 1


def test_cli(tmp_path, capsys, monkeypatch):
    from worldcam.cli import users_admin as cli
    monkeypatch.setenv("TELEGRAM_OWNER_CHAT_ID", "42")
    db = str(tmp_path / "u.db")
    assert cli.main(["--db", db, "--env", str(tmp_path / "x"), "list"]) == 0
    assert "owner" in capsys.readouterr().out
    assert cli.main(["--db", db, "add", "--name", "Ann", "--role", "household",
                     "--telegram-id", "5", "--profiles", "cctv"]) == 0
    assert cli.main(["--db", db, "invite", "--role", "caregiver", "--profiles", "medical"]) == 0
    out = capsys.readouterr().out
    assert "/join " in out and "Invite code:" in out
    assert cli.main(["--db", db, "deactivate", "nosuchid"]) == 1
    owner = SqliteUserDirectory(db).by_channel("telegram", "42")
    assert cli.main(["--db", db, "deactivate", owner.id]) == 1       # last owner
    assert "last active owner" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--db", db, "invite", "--role", "owner"])
