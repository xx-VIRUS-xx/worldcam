from types import SimpleNamespace

from worldcam.brain.api import InMemoryEventStore
from worldcam.brain.recorder import EventRecorder, RecorderConfig
from worldcam.identity.resolver import Decision
from worldcam.pipeline import Identity, Track, TrackState, TrackUpdate

BOX = (0.1, 0.1, 0.3, 0.5)


def trk(tid, t0, t1, state=TrackState.ACTIVE, label="person", zones=("desk",), stat=None):
    return Track(track_id=tid, label=label, state=state, box=BOX, first_ts=t0, last_ts=t1,
                 zones=tuple(zones), stationary_since=stat)


def upd(active=(), ended=()):
    return TrackUpdate(active=list(active), ended=list(ended))


def rec(**kw):
    st = InMemoryEventStore()
    cfg = RecorderConfig(**kw)
    return EventRecorder(st, cfg), st


def kinds(evs):
    return [e.kind for e in evs]


def test_entered_only_after_confirmation_and_identity_wait():
    r, st = rec()
    assert r.on_track_update(upd([trk(1, 0, 0.2, TrackState.NEW)]), ts=0.2) == []
    assert r.on_track_update(upd([trk(1, 0, 1.5)]), ts=1.5) == []         # confirmed, but waiting for identity
    out = r.on_track_update(upd([trk(1, 0, 3.2)]), ts=3.2)                # identity_wait_s passed
    assert kinds(out) == ["entered"]
    e = out[0]
    assert e.subject_id is None and e.subject_label == "unknown person" and e.zone == "desk"
    assert e.detail["unidentified"] is True and e.detail["track_id"] == 1
    assert e.profiles == ("cctv", "companion", "medical")
    # continuing frames do not re-enter
    assert r.on_track_update(upd([trk(1, 0, 4)]), ts=4) == []
    assert st.count() == 1


def test_matched_identity_enters_immediately_with_person_and_snapshot():
    r, st = rec()
    r.on_track_update(upd([trk(1, 0, 1.2)]), ts=1.2)
    res = SimpleNamespace(identity=Identity(1, "Rahul", 0.9), decision=Decision.MATCHED,
                          person_id="p_rahul", cluster_id="c1", crop_path="/snaps/r.jpg")
    out = r.on_resolve(res, ts=1.3)
    assert kinds(out) == ["entered"]
    e = out[0]
    assert e.subject_id == "p_rahul" and e.subject_label == "Rahul"
    assert e.snapshot_paths == ("/snaps/r.jpg",) and e.confidence == 0.9


def test_unknown_cluster_identity_recorded_in_detail():
    r, st = rec()
    r.on_track_update(upd([trk(1, 0, 1.2)]), ts=1.2)
    res = SimpleNamespace(identity=Identity(1, "unknown", 0.2), decision=Decision.UNKNOWN_CLUSTER,
                          person_id=None, cluster_id="c9", crop_path=None)
    e = r.on_resolve(res, ts=1.3)[0]
    assert e.subject_id is None and e.detail["cluster_id"] == "c9"


def test_left_after_grace_and_reentry_cancels_left():
    r, st = rec(leave_grace_s=10)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4.1)
    assert st.count(kind="entered") == 1
    assert r.on_track_update(upd(ended=[trk(1, 0, 20, TrackState.ENDED)]), ts=20) == []   # still in grace
    out = r.flush(31)
    assert kinds(out) == ["left"] and out[0].ts == 20          # ts = last time seen
    # returns inside the grace period with a NEW track id of a known person: no left, no entered
    r2, st2 = rec(leave_grace_s=10)
    res = lambda tid: SimpleNamespace(identity=Identity(tid, "Rahul", 0.9), decision=Decision.MATCHED,
                                      person_id="p1", cluster_id=None, crop_path=None)
    r2.on_track_update(upd([trk(1, 0, 2)]), ts=2)
    r2.on_resolve(res(1), ts=2)
    r2.on_track_update(upd(ended=[trk(1, 0, 10, TrackState.ENDED)]), ts=10)
    r2.on_track_update(upd([trk(2, 12, 14)]), ts=14)
    r2.on_resolve(res(2), ts=14)
    assert r2.flush(20) == []
    assert kinds(st2.query(newest_first=False)) == ["entered"]


def test_unseen_for_absence_s_treated_as_ended():
    r, st = rec(absence_s=30, leave_grace_s=5)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4.2)
    r.flush(20)
    assert st.count(kind="left") == 0
    out = r.flush(40)       # unseen 36 s -> ended; last_ts + grace elapsed
    assert kinds(out) == ["left"]
    assert r.stats()["tracks"] == 0 and r.stats()["presences"] == 0


def test_zone_changed_debounced():
    r, st = rec(zone_min_s=2)
    r.on_track_update(upd([trk(1, 0, 4, zones=("door",))]), ts=4)
    r.on_track_update(upd([trk(1, 0, 4.2, zones=("door",))]), ts=4.2)
    assert r.on_track_update(upd([trk(1, 0, 5, zones=("desk",))]), ts=5) == []
    assert r.on_track_update(upd([trk(1, 0, 5.5, zones=("door",))]), ts=5.5) == []       # flicker back
    assert r.on_track_update(upd([trk(1, 0, 6, zones=("desk",))]), ts=6) == []
    out = r.on_track_update(upd([trk(1, 0, 8.5, zones=("desk",))]), ts=8.5)
    assert kinds(out) == ["zone_changed"]
    assert out[0].zone == "desk" and out[0].detail["from_zone"] == "door"


def test_stationary_once_per_episode():
    r, st = rec(stationary_after_s=60)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4)
    r.on_track_update(upd([trk(1, 0, 4.2)]), ts=4.2)
    s = lambda now: trk(1, 0, now, TrackState.STATIONARY, stat=10)
    assert r.on_track_update(upd([s(40)]), ts=40) == []
    assert kinds(r.on_track_update(upd([s(75)]), ts=75)) == ["stationary"]
    assert r.on_track_update(upd([s(80)]), ts=80) == []
    r.on_track_update(upd([trk(1, 0, 90)]), ts=90)                      # moved
    assert kinds(r.on_track_update(upd([s(200)]), ts=200)) == ["stationary"]


def test_seen_rate_limited():
    r, st = rec(seen_every_s=300)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4)
    r.on_track_update(upd([trk(1, 0, 4.2)]), ts=4.2)
    n = 0
    for t in range(5, 700, 5):
        n += kinds(r.on_track_update(upd([trk(1, 0, t)]), ts=t)).count("seen")
    assert n == 2     # ~every 300 s over ~700 s


def test_hard_cap_events_per_minute_per_subject():
    r, st = rec(max_events_per_min=3, zone_min_s=0, seen_every_s=0.0)
    r.on_track_update(upd([trk(1, 0, 4)]), ts=4)
    r.on_track_update(upd([trk(1, 0, 4.1)]), ts=4.1)
    for i in range(40):                       # zone flapping every second
        z = "a" if i % 2 else "b"
        r.on_track_update(upd([trk(1, 0, 5 + i, zones=(z,))]), ts=5 + i)
        r.on_track_update(upd([trk(1, 0, 5 + i + .5, zones=(z,))]), ts=5 + i + .5)
    for lo in range(0, 60, 10):
        assert st.count(since=lo, until=lo + 60) <= 3
    assert r.stats()["dropped"] > 0


def test_unlisted_labels_ignored_and_custom_profile_map():
    r, st = rec(profile_map={"dog": ("companion",)})
    assert r.on_track_update(upd([trk(1, 0, 5, label="person")]), ts=5) == []
    out = r.on_track_update(upd([trk(2, 0, 2, label="dog")]), ts=2)
    out += r.on_track_update(upd([trk(2, 0, 3, label="dog")]), ts=3)
    assert kinds(out) == ["entered"] and out[0].profiles == ("companion",) and out[0].subject_label == "dog"


def test_entity_sighting_enters_and_leaves():
    r, st = rec(leave_grace_s=5, absence_s=20)
    out = r.on_sighting("ent_cat1", "cat", ts=100, zone="sofa", snapshot="/s/cat.jpg", name="Tom")
    assert kinds(out) == ["entered"] and out[0].subject_id == "ent_cat1" and out[0].subject_label == "Tom"
    assert r.on_sighting("ent_cat1", "cat", ts=110, zone="sofa") == []
    assert kinds(r.flush(140)) == ["left"]


def test_bounded_memory_under_many_tracks():
    r, st = rec(max_subjects=20, max_tracks=30, absence_s=1e9)
    for i in range(500):
        r.on_track_update(upd([trk(i, i, i + 2, label="person")]), ts=i + 2)
        r.on_track_update(upd([trk(i, i, i + 4, label="person")]), ts=i + 4)
    s = r.stats()
    assert s["tracks"] <= 30 and s["presences"] <= 20


def test_thread_safe_smoke():
    import threading
    r, st = rec()
    def work(k):
        for t in range(50):
            r.on_track_update(upd([trk(k, 0, t)]), ts=t)
    th = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert st.count(kind="entered") == 4
