from datetime import datetime, timezone

from worldcam.brain.api import InMemoryEventStore, Routine, new_event
from worldcam.brain.retention import RetentionPolicy, delete_files, forget_subject
from worldcam.brain.store import SqliteEventStore
from worldcam.brain.summary import build_day_summary, polish_summary
from worldcam.models.api import FakeModelClient

UTC = timezone.utc
DAY = "2026-11-25"


def T(hh, mm=0, day=25):
    return datetime(2026, 11, day, hh, mm, tzinfo=UTC).timestamp()


def E(i, ts, kind, **kw):
    return new_event(kind, id=f"e{i}", ts=ts, **kw)


def events():
    return [
        E(1, T(7, 30), "entered", subject_id="p1", subject_label="Rahul", zone="kitchen", profiles=("medical",)),
        E(2, T(8, 0), "zone_changed", subject_id="p1", subject_label="Rahul", zone="sofa", profiles=("medical",)),
        E(3, T(8, 30), "activity_changed", subject_id="p1", activity="eating", profiles=("medical",)),
        E(4, T(9, 0), "left", subject_id="p1", subject_label="Rahul", zone="sofa", profiles=("medical",)),
        E(5, T(11, 0), "entered", subject_label="unknown person", zone="door", profiles=("cctv",),
          detail={"cluster_id": "c5"}),
        E(6, T(11, 5), "left", subject_label="unknown person", zone="door", profiles=("cctv",),
          detail={"cluster_id": "c5"}),
        E(7, T(15, 0), "alert", subject_id="p1", activity="fall_suspected", profiles=("medical",)),
        E(8, T(7, 30, day=24), "entered", subject_id="p1", profiles=("medical",)),     # other day
    ]


def meds_routine():
    return Routine(id="r1", subject_id="p1", kind="daily_at", description="usually taking medication around 09:00",
                   params={"event_kind": "activity_changed", "activity": "taking_medication",
                           "minute_of_day": 540, "tolerance_min": 45}, support=6, confidence=0.9)


def test_summary_fixed_day_exact():
    s = build_day_summary(events(), [meds_routine()], DAY, tz=UTC)
    st = s.stats
    assert st["first_seen"] == "07:30" and st["last_seen"] == "11:05"
    assert st["visitors"] == ["Rahul"] and st["unknown_visits"] == 1
    assert st["zone_dwell_min"] == {"sofa": 60.0, "kitchen": 30.0, "door": 5.0}
    assert st["activities"] == {"eating": 1} and st["alerts"] == 1
    assert [m["routine_id"] for m in st["missed_routines"]] == ["r1"]
    assert s.event_ids == ("e1", "e2", "e3", "e4", "e5", "e6", "e7")
    assert s.text == (f"{DAY}: Activity seen from 07:30 to 11:05. Seen: Rahul. 1 unidentified visit(s). "
                      "Time by zone: sofa 60 min, kitchen 30 min, door 5 min. Activities: eating x1. "
                      "Missed routine: usually taking medication around 09:00 (expected ~09:00). "
                      "Alert: fall_suspected at 15:00.")
    assert build_day_summary(events(), [meds_routine()], DAY, tz=UTC) == s      # deterministic


def test_summary_routine_done_is_not_missed_and_profile_filter():
    evs = events() + [E(9, T(9, 10), "activity_changed", subject_id="p1", activity="taking_medication")]
    s = build_day_summary(evs, [meds_routine()], DAY, tz=UTC)
    assert s.stats["missed_routines"] == []
    sc = build_day_summary(events(), [], DAY, profile="cctv", tz=UTC)
    assert sc.stats["unknown_visits"] == 1 and sc.stats["visitors"] == [] and sc.profile == "cctv"
    assert build_day_summary([], [], DAY, tz=UTC).text == f"{DAY}: no events recorded."


def test_polish_with_fake_model_and_fallbacks():
    base = build_day_summary(events(), [meds_routine()], DAY, tz=UTC)
    good = FakeModelClient(lambda r: "Rahul was seen from 07:30 to 11:05 and ate once. He missed his 09:00 medication.")
    p = polish_summary(base, good)
    assert p.text.startswith("Rahul was seen") and p.stats["polished"] and p.stats["facts_text"] == base.text
    assert good.calls[0].task == "summarize" and base.text in good.calls[0].prompt
    assert p.stats["first_seen"] == base.stats["first_seen"]
    # invented number -> deterministic text kept
    assert polish_summary(base, FakeModelClient(lambda r: "He was seen 12 times at 23:59.")) == base
    # empty answer / exception -> deterministic text kept
    assert polish_summary(base, FakeModelClient(lambda r: "")) == base

    def boom(r):
        raise RuntimeError("no creds")
    assert polish_summary(base, FakeModelClient(boom)) == base


def _store():
    st = SqliteEventStore()
    day = 86400
    now = 100 * day
    st.add_event(E(1, now - 40 * day, "seen", subject_id="a", profiles=("cctv",), snapshot_paths=("/s/1",)))
    st.add_event(E(2, now - 10 * day, "seen", subject_id="a", profiles=("cctv",), snapshot_paths=("/s/2",)))
    st.add_event(E(3, now - 40 * day + 5, "seen", subject_id="b", profiles=("medical",), snapshot_paths=("/s/3",)))
    st.add_event(E(4, now - 3 * day, "seen", subject_id="b", profiles=("medical",), snapshot_paths=("/s/4",)))
    st.add_event(E(5, now - 20 * day, "seen", subject_id="c", snapshot_paths=("/s/5",)))   # no profile
    return st, now


def test_retention_per_profile_returns_paths():
    st, now = _store()
    pol = RetentionPolicy({"cctv": 30, "medical": 90}, default_days=14)
    paths = pol.apply(st, now)
    assert paths == ["/s/1", "/s/5"]        # cctv 40d>30 gone; no-profile 20d>14 gone; medical 40d kept
    assert sorted(e.id for e in st.query()) == ["e2", "e3", "e4"]
    assert pol.apply(st, now) == []
    assert RetentionPolicy().apply(st, now, {"cctv": 1, "medical": 1}) == ["/s/2", "/s/3", "/s/4"]


def test_retention_fallback_without_delete_event():
    mem = InMemoryEventStore()                 # no delete_event: leading-run purge only
    mem.add_event(E(1, 10, "seen", subject_id="a", snapshot_paths=("/m/1",)))
    mem.add_event(E(2, 20, "seen", subject_id="a", snapshot_paths=("/m/2",)))
    mem.add_event(E(3, 9_000_000, "seen", subject_id="a"))
    assert RetentionPolicy(default_days=1).apply(mem, 9_000_100) == ["/m/1", "/m/2"]
    assert mem.count() == 1


def test_forget_subject_and_file_deletion(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    st = SqliteEventStore()
    st.add_event(E(1, 1, "seen", subject_id="a", snapshot_paths=(str(a),)))
    st.add_event(E(2, 2, "seen", subject_id="b", snapshot_paths=(str(b),)))
    st.upsert_routine(meds_routine())
    assert forget_subject(st, "p1") == []
    assert st.list_routines("p1") == []                      # routines of a forgotten subject go too
    assert forget_subject(st, "a") == [str(a)] and a.exists()    # no delete requested
    assert forget_subject(st, "b", delete=True, root=str(tmp_path / "other")) == [str(b)]
    assert b.exists()                                        # outside root: refused
    assert delete_files([str(b)], root=str(tmp_path)) == [str(b)] and not b.exists()
