from datetime import datetime, timezone

import pytest

from worldcam.brain.api import InMemoryEventStore, Routine, new_event
from worldcam.brain.routines import apply_mining, mine_routines, routine_status
from worldcam.brain.store import SqliteEventStore

UTC = timezone.utc


def T(day, hh, mm=0):
    """UTC timestamp for 2026-11-<day> hh:mm. 2026-11-02 is a Monday."""
    return datetime(2026, 11, day, hh, mm, tzinfo=UTC).timestamp()


def med(ts, sid="p1", i=[0]):
    i[0] += 1
    return new_event("activity_changed", id=f"m{i[0]}", ts=ts, subject_id=sid, activity="taking_medication")


def entered(ts, sid="p1", i=[0]):
    i[0] += 1
    return new_event("entered", id=f"n{i[0]}", ts=ts, subject_id=sid)


def regular(days=range(2, 12), minute_jitter=(0, 7, -5, 12, -9, 3, 0, -11, 8, 2)):
    return [med(T(d, 9, 0) + j * 60) for d, j in zip(days, minute_jitter)]


def test_regular_daily_routine_mined():
    rts = mine_routines(regular(), tz=UTC)
    assert len(rts) == 1
    r = rts[0]
    assert r.kind == "daily_at" and r.source == "inferred"
    assert abs(r.params["minute_of_day"] - 540) <= 3
    assert r.params["activity"] == "taking_medication"
    assert r.support == 10 and r.confidence > 0.7
    assert "09:0" in r.description


def test_support_below_three_days_not_mined():
    assert mine_routines(regular(days=range(2, 4), minute_jitter=(0, 1)), tz=UTC) == []
    # many events but on only 2 distinct days
    evs = [med(T(2, 9)), med(T(2, 9, 5)), med(T(3, 9))]
    assert mine_routines(evs, tz=UTC) == []


def test_irregular_not_mined():
    hours = [3, 14, 8, 21, 11, 17, 5, 19]
    evs = [med(T(2 + i, h)) for i, h in enumerate(hours)]
    assert mine_routines(evs, tz=UTC) == []


def test_drifting_schedule_rejected():
    evs = [med(T(2 + i, 9) + i * 25 * 60) for i in range(10)]     # +25 min every day
    assert mine_routines(evs, tz=UTC) == []


def test_weekday_only_routine():
    # Mon-Fri 08:30 for two weeks, nothing on weekends, but the subject is seen on weekends too
    evs, days = [], [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
    for d in days:
        evs.append(med(T(d, 8, 30)))
    for d in (7, 8, 14, 15):
        evs.append(new_event("seen", id=f"s{d}", ts=T(d, 12), subject_id="p1"))
    rts = [r for r in mine_routines(evs, tz=UTC) if r.params.get("activity")]
    assert len(rts) == 1 and rts[0].kind == "weekday_at"
    assert rts[0].params["weekdays"] == [0, 1, 2, 3, 4]


def test_presence_window():
    evs = []
    for d in range(2, 9):
        evs.append(entered(T(d, 8, 0) + (d % 3) * 60))
        evs.append(new_event("left", id=f"l{d}", ts=T(d, 17, 0) - (d % 2) * 120, subject_id="p1"))
    rts = [r for r in mine_routines(evs, tz=UTC) if r.kind == "presence_window"]
    assert len(rts) == 1
    p = rts[0].params
    assert abs(p["start_minute"] - 8 * 60) <= 3 and abs(p["end_minute"] - 17 * 60) <= 3


def test_two_clusters_morning_and_evening():
    evs = [med(T(d, 8, 0)) for d in range(2, 8)] + [med(T(d, 20, 0) + 600) for d in range(2, 8)]
    # first event per day only counts toward the first cluster -> evening ones are 2nd of the day
    rts = mine_routines(evs, tz=UTC)
    assert len(rts) == 1 and abs(rts[0].params["minute_of_day"] - 480) <= 2


def test_user_routines_never_overwritten():
    st = InMemoryEventStore()
    for e in regular():
        st.add_event(e)
    user = Routine(id="u1", subject_id="p1", kind="daily_at", description="meds at 09:00 (owner)",
                   params={"event_kind": "activity_changed", "activity": "taking_medication",
                           "minute_of_day": 540, "tolerance_min": 30}, source="user", confidence=1.0)
    st.upsert_routine(user)
    other = Routine(id="u2", subject_id="p1", kind="interval", description="drink every 3h",
                    params={"event_kind": "seen", "interval_min": 180}, source="user")
    st.upsert_routine(other)
    now = T(12, 12)
    mined = apply_mining(st, st, now=now, tz=UTC)
    assert mined == []                                  # duplicate of the user routine -> skipped
    assert st.routines["u1"] == user and "u2" in st.routines
    # without the duplicate the miner writes its own and still keeps the user ones
    st.delete_routine("u1")
    mined = apply_mining(st, st, now=now, tz=UTC)
    assert len(mined) == 1 and "u2" in st.routines
    again = apply_mining(st, st, now=now, tz=UTC)
    assert [r.id for r in again] == [r.id for r in mined]            # idempotent
    assert sum(r.source == "inferred" for r in st.routines.values()) == 1


def test_stale_inferred_removed_on_remine():
    st = SqliteEventStore()
    for e in regular():
        st.add_event(e)
    apply_mining(st, st, now=T(12, 12), tz=UTC)
    assert len(st.list_routines()) == 1
    st.purge_before(T(30, 0))                           # all history gone, subject has no events now
    st.add_event(med(T(13, 3), sid="p1"))               # one event: subject still mined, but no routine
    apply_mining(st, st, now=T(14, 0), tz=UTC)
    assert st.list_routines() == []


def _st_with_routine(kind="daily_at", **extra):
    st = SqliteEventStore()
    p = {"event_kind": "activity_changed", "activity": "taking_medication", "minute_of_day": 540,
         "tolerance_min": 45, **extra}
    r = Routine(id="r1", subject_id="p1", kind=kind, description="meds", params=p, source="inferred")
    st.upsert_routine(r)
    return st, r


def test_status_on_time_due_overdue():
    st, r = _st_with_routine()
    f = lambda now: routine_status(st, [r], now, tz=UTC)[0]
    assert f(T(5, 7, 0)).state == "on_time"          # not yet due
    assert f(T(5, 9, 0)).state == "due"
    assert f(T(5, 9, 45)).state == "due"
    s = f(T(5, 10, 30))
    assert s.state == "overdue" and s.last_done is None and s.expected_at == T(5, 9)
    st.add_event(med(T(5, 9, 10)))
    s = f(T(5, 10, 30))
    assert s.state == "on_time" and s.last_done == T(5, 9, 10)
    assert f(T(6, 10, 30)).state == "overdue"       # yesterday's dose does not count today


def test_status_weekday_not_applicable_and_store_method():
    st, r = _st_with_routine("weekday_at", weekdays=[0, 1, 2, 3, 4])
    assert routine_status(st, [r], T(8, 12), tz=UTC)[0].state == "not_applicable"   # Sunday
    assert routine_status(st, [r], T(9, 12), tz=UTC)[0].state == "overdue"          # Monday
    assert isinstance(st.status(T(9, 12), "p1"), list)


def test_status_presence_window():
    st = SqliteEventStore()
    r = Routine(id="w", subject_id="p1", kind="presence_window", description="present 08-17",
                params={"start_minute": 480, "end_minute": 1020, "tolerance_min": 30}, source="inferred")
    f = lambda now: routine_status(st, [r], now, tz=UTC)[0].state
    assert f(T(5, 3)) == "not_applicable"
    assert f(T(5, 8, 10)) == "due"
    assert f(T(5, 10)) == "overdue"
    st.add_event(entered(T(5, 9)))
    assert f(T(5, 10)) == "on_time"
    st.add_event(new_event("left", id="lf", ts=T(5, 9, 30), subject_id="p1"))
    assert f(T(5, 10)) == "overdue"
