import time

from worldcam.agent.budget import (BudgetConfig, InMemoryAttentionBudget, SqliteAttentionBudget,
                                   budget_from_users)
from worldcam.users.api import InMemoryUserDirectory


def local(h, day=10, m=0):
    return time.mktime((2026, 11, day, h, m, 0, 0, 0, -1))


NOON = local(12)


def mk(cls=InMemoryAttentionBudget, **cfg):
    users = InMemoryUserDirectory()
    owner = users.add("Olga", "owner", "fake", "1")
    guest = users.add("Gus", "guest", "fake", "4")
    b = budget_from_users(users, config=BudgetConfig(**cfg)) if cls is InMemoryAttentionBudget else \
        SqliteAttentionBudget(":memory:", BudgetConfig(**cfg), lambda uid: (users.get(uid) or owner).role)
    return b, owner, guest


def test_hour_and_day_caps_by_role():
    b, owner, guest = mk()
    assert b.limits(guest.id) == (1, 3)
    assert b.allow(guest.id, now=NOON)
    b.record(guest.id, "question", NOON)
    assert not b.allow(guest.id, now=NOON + 60)          # hourly cap 1
    assert b.allow(guest.id, now=NOON + 3700)
    b.record(guest.id, "question", NOON + 3700)
    b.record(guest.id, "question", NOON + 7400)
    assert not b.allow(guest.id, now=NOON + 11100)       # daily cap 3
    r = b.remaining(guest.id, NOON + 11100)
    assert r["day"] == 0 and r["hard_day"] == 3


def test_quiet_hours_default_and_per_user():
    b, owner, guest = mk()
    assert not b.allow(owner.id, now=local(23, m=30))
    assert not b.allow(owner.id, now=local(3))
    assert b.allow(owner.id, now=local(8))
    b.set_quiet_hours(owner.id, 13, 15)
    assert b.allow(owner.id, now=local(3))
    assert not b.allow(owner.id, now=local(14))
    b.set_quiet_hours(owner.id, None, None)
    assert b.allow(owner.id, now=local(14))


def test_priority3_bypasses_quiet_and_caps_but_not_hard_ceiling():
    b, owner, guest = mk()
    t = local(2)
    assert b.allow(guest.id, priority=3, now=t)          # quiet hours bypassed
    for i in range(3):
        b.record(guest.id, "alert", t + i)
    assert not b.allow(guest.id, priority=1, now=t + 10)
    assert b.allow(guest.id, priority=3, now=t + 10)     # day cap bypassed
    for i in range(3):
        b.record(guest.id, "alert", t + 20 + i)          # 6 = hard ceiling (2 x 3)
    assert not b.allow(guest.id, priority=3, now=t + 30)


def test_ignore_backoff_is_exponential_and_answer_resets():
    b, owner, _ = mk()
    b.record(owner.id, "ignored", NOON)
    assert not b.allow(owner.id, now=NOON + 3600)        # 2 h cooldown
    assert b.allow(owner.id, now=NOON + 2 * 3600 + 1)
    b.record(owner.id, "ignored", NOON + 3 * 3600)
    assert b.remaining(owner.id, NOON + 3 * 3600)["ignore_streak"] == 2
    assert not b.allow(owner.id, now=NOON + 3 * 3600 + 3 * 3600)   # 4 h now
    assert b.allow(owner.id, now=NOON + 3 * 3600 + 4 * 3600 + 1)
    assert b.allow(owner.id, priority=3, now=NOON + 3 * 3600 + 60)  # urgent ignores cooldown
    b.record(owner.id, "answered", NOON + 8 * 3600)
    assert b.remaining(owner.id, NOON + 8 * 3600)["ignore_streak"] == 0
    assert b.allow(owner.id, now=NOON + 8 * 3600 + 1)


def test_profile_limit_tightens_role_limit():
    b, owner, _ = mk(profile_limits={"companion": (1, 2)})
    b.record(owner.id, "question", NOON)
    assert b.allow(owner.id, now=NOON + 60)                      # role cap 6/h
    assert not b.allow(owner.id, profile="companion", now=NOON + 60)


def test_sqlite_budget_persists(tmp_path):
    users = InMemoryUserDirectory()
    g = users.add("G", "guest", "fake", "4")
    path = str(tmp_path / "b.db")
    b = budget_from_users(users, path=path)
    b.record(g.id, "question", NOON)
    b.record(g.id, "ignored", NOON)
    b.set_quiet_hours(g.id, 10, 11)
    b.close()
    b2 = budget_from_users(users, path=path)
    assert b2.remaining(g.id, NOON + 60)["hour"] == 0
    assert b2.remaining(g.id, NOON + 60)["cooldown_until"] is not None
    assert b2.quiet_hours(g.id) == (10, 11)


def test_sqlite_same_policy():
    b, owner, guest = mk(SqliteAttentionBudget)
    assert b.allow(guest.id, now=NOON)
    b.record(guest.id, "question", NOON)
    assert not b.allow(guest.id, now=NOON + 1)
    assert not b.allow(guest.id, now=local(23, m=30))
