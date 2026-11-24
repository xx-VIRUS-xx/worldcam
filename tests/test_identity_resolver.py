"""IdentityResolver decision paths + per-track caching, driven by synthetic
embeddings against InMemoryGraphStore. No model, no camera."""
import math

import numpy as np
import pytest

from worldcam.graph.api import InMemoryGraphStore, cosine
from worldcam.identity.resolver import Decision, IdentityResolver
from worldcam.pipeline import BestSnapshot, FaceObservation, Track, TrackState

D = 512
RNG = np.random.default_rng(7)


def unit():
    v = RNG.normal(size=D)
    return v / np.linalg.norm(v)


def at_cos(base, s):
    """Unit vector with cosine exactly `s` to `base`."""
    o = unit()
    o -= o.dot(base) * base
    o /= np.linalg.norm(o)
    return s * base + math.sqrt(1 - s * s) * o


def track(tid=1, ts=100.0, zones=("door",)):
    return Track(tid, "person", TrackState.ACTIVE, (0.1, 0.1, 0.5, 0.9), ts, ts, zones=zones,
                 best=BestSnapshot(seq=1, ts=ts, box=(0.1, 0.1, 0.5, 0.9), score=0.9))


def face(vec, tid=1, ts=100.0, q=0.8, path="/tmp/f.jpg"):
    return FaceObservation(tid, ts, (0.2, 0.15, 0.3, 0.3), tuple(float(x) for x in vec), q, path)


def labeled(store, base, name):
    pid = store.upsert_person(name=name, kind="household")
    _, cid = store.add_face(base, quality=0.9, ts=1.0, snapshot_path=None)
    store.label_face_cluster(cid, pid)
    return pid, cid


@pytest.fixture
def store():
    return InMemoryGraphStore()


def test_matched(store):
    a = unit()
    pid, cid = labeled(store, a, "Prabhat")
    r = IdentityResolver(store).resolve(track(), [face(at_cos(a, 0.7))])
    assert r.decision is Decision.MATCHED
    assert r.identity.name == "Prabhat" and r.identity.conf == pytest.approx(0.7, abs=1e-3)
    assert r.person_id == pid and r.cluster_id == cid and r.cluster_size == 2
    s = store.sightings[r.sighting_id]
    assert s.person_id == pid and s.cluster_id == cid and s.zones == ("door",)
    assert s.snapshot_path == "/tmp/f.jpg" and r.crop_path == "/tmp/f.jpg"


def test_matched_forces_person_cluster_even_below_store_threshold():
    # store clusters at 0.6 but match_t is 0.45: a 0.5 match must still join the person's cluster
    st = InMemoryGraphStore(cluster_threshold=0.6)
    a = unit()
    _, cid = labeled(st, a, "A")
    r = IdentityResolver(st).resolve(track(), [face(at_cos(a, 0.5))])
    assert r.decision is Decision.MATCHED and r.cluster_id == cid


def test_low_confidence(store):
    a = unit()
    pid, cid = labeled(store, a, "Prabhat")
    r = IdentityResolver(store).resolve(track(), [face(at_cos(a, 0.40))])
    assert r.decision is Decision.LOW_CONFIDENCE
    assert r.identity.name == "unknown" and r.person_id is None
    assert r.candidates[0].person_id == pid and r.candidates[0].name == "Prabhat"
    assert r.candidates[0].score == pytest.approx(0.40, abs=1e-3)
    assert r.cluster_id != cid            # unsure faces don't pollute the labeled cluster
    assert store.sightings[r.sighting_id].person_id is None


def test_unknown_cluster_reports_cluster_and_size(store):
    u = unit()
    _, cid = store.add_face(u, quality=0.9, ts=1.0, snapshot_path=None)
    res = IdentityResolver(store)
    r = res.resolve(track(1), [face(at_cos(u, 0.8), tid=1)])
    assert r.decision is Decision.UNKNOWN_CLUSTER and r.cluster_id == cid and r.cluster_size == 2
    r = res.resolve(track(2), [face(at_cos(u, 0.8), tid=2)])
    assert r.cluster_id == cid and r.cluster_size == 3
    assert r.identity.name == "unknown"


def test_unknown_cluster_wins_over_weak_labeled(store):
    a, u = unit(), unit()
    labeled(store, a, "A")
    store.add_face(u, quality=0.9, ts=1.0, snapshot_path=None)
    f = 0.38 * a + 0.8 * u
    r = IdentityResolver(store).resolve(track(), [face(f / np.linalg.norm(f))])
    assert r.decision is Decision.UNKNOWN_CLUSTER
    assert {c.person_id is None for c in r.candidates} == {True, False}   # both offered to the engine


def test_new_face_empty_store(store):
    r = IdentityResolver(store).resolve(track(), [face(unit())])
    assert r.decision is Decision.NEW_FACE and r.cluster_size == 1 and r.candidates == ()


def test_new_face_far_from_everything(store):
    a = unit()
    _, cid = labeled(store, a, "A")
    r = IdentityResolver(store).resolve(track(), [face(at_cos(a, 0.2))])
    assert r.decision is Decision.NEW_FACE and r.cluster_id != cid
    assert r.identity.conf == pytest.approx(0.2, abs=1e-3)


def test_conflict_two_persons_close(store):
    a = unit()
    b = at_cos(a, 0.0)
    pa, _ = labeled(store, a, "A")
    pb, _ = labeled(store, b, "B")
    c = at_cos(a, 0.0)
    c -= c.dot(b) * b
    c /= np.linalg.norm(c)
    f = 0.5 * a + 0.48 * b + math.sqrt(1 - 0.25 - 0.48 ** 2) * c
    assert cosine(f, a) > 0.45 and cosine(f, b) > 0.45
    r = IdentityResolver(store).resolve(track(), [face(f)])
    assert r.decision is Decision.CONFLICT and r.detail["reason"] == "two_persons_close"
    assert {c.person_id for c in r.candidates[:2]} == {pa, pb}
    assert r.person_id is None and r.identity.name == "unknown"


def test_no_conflict_when_margin_is_large(store):
    a = unit()
    b = at_cos(a, 0.0)
    labeled(store, a, "A")
    labeled(store, b, "B")
    f = 0.75 * a + 0.36 * b
    f += math.sqrt(1 - f.dot(f)) * at_cos(a, 0.0)  # roughly unit; exact value irrelevant
    r = IdentityResolver(store).resolve(track(), [face(f / np.linalg.norm(f))])
    assert r.decision is Decision.MATCHED and r.identity.name == "A"


def test_reverify_identity_change_is_conflict(store):
    a = unit()
    b = at_cos(a, 0.0)
    pa, _ = labeled(store, a, "A")
    pb, _ = labeled(store, b, "B")
    res = IdentityResolver(store, reverify_s=60)
    assert res.resolve(track(ts=100), [face(at_cos(a, 0.8), ts=100)]).person_id == pa
    r = res.resolve(track(ts=200), [face(at_cos(b, 0.8), ts=200)])
    assert r.decision is Decision.CONFLICT and r.detail["previous_person_id"] == pa
    assert {c.person_id for c in r.candidates} >= {pa, pb}
    assert r.person_id is None and r.identity.name == "unknown"   # the track is not flipped to B


def test_no_face_then_face(store):
    res = IdentityResolver(store)
    r = res.resolve(track(), [])
    assert r.decision is Decision.NO_FACE and r.identity.name == "unknown" and not store.sightings
    assert res.resolve(track(), []).cached
    r = res.resolve(track(), [face(unit())])
    assert r.decision is Decision.NEW_FACE and not r.cached


def test_low_quality_faces_ignored(store):
    r = IdentityResolver(store, min_quality=0.5).resolve(track(), [face(unit(), q=0.2)])
    assert r.decision is Decision.NO_FACE and not store.faces


def test_best_quality_face_is_used(store):
    a = unit()
    labeled(store, a, "A")
    r = IdentityResolver(store).resolve(track(), [face(unit(), q=0.3), face(at_cos(a, 0.8), q=0.9)])
    assert r.decision is Decision.MATCHED and r.quality == 0.9


# ---- per-track caching -----------------------------------------------------

def test_cache_resolves_each_track_once(store):
    a = unit()
    labeled(store, a, "A")
    res = IdentityResolver(store, reverify_s=600)
    r1 = res.resolve(track(ts=100), [face(at_cos(a, 0.8), ts=100)])
    n_faces, n_sight = len(store.faces), len(store.sightings)
    for t in (110, 300, 699):
        r = res.resolve(track(ts=t), [face(at_cos(a, 0.8), ts=t)])
        assert r.cached and r.identity == r1.identity and r.sighting_id == r1.sighting_id
    assert (len(store.faces), len(store.sightings)) == (n_faces, n_sight)
    assert res.cached(1).decision is Decision.MATCHED


def test_cache_reverifies_after_interval(store):
    a = unit()
    labeled(store, a, "A")
    res = IdentityResolver(store, reverify_s=600)
    r1 = res.resolve(track(ts=100), [face(at_cos(a, 0.8), ts=100)])
    r2 = res.resolve(track(ts=700), [face(at_cos(a, 0.8), ts=700)])
    assert not r2.cached and r2.sighting_id != r1.sighting_id and r2.decision is Decision.MATCHED


def test_cache_upgrades_unsure_result_on_better_face(store):
    a = unit()
    labeled(store, a, "A")
    res = IdentityResolver(store)
    r = res.resolve(track(), [face(at_cos(a, 0.40), q=0.3)])
    assert r.decision is Decision.LOW_CONFIDENCE
    assert res.resolve(track(), [face(at_cos(a, 0.8), q=0.4)]).cached         # not better enough
    r = res.resolve(track(), [face(at_cos(a, 0.8), q=0.6)])
    assert not r.cached and r.decision is Decision.MATCHED


def test_cache_is_per_track_and_forget_and_invalidate(store):
    u = unit()
    res = IdentityResolver(store)
    r1 = res.resolve(track(1), [face(u, tid=1)])
    r2 = res.resolve(track(2), [face(at_cos(u, 0.9), tid=2)])
    assert not r2.cached and r2.cluster_id == r1.cluster_id
    res.forget(1)
    assert res.cached(1) is None and res.cached(2) is not None
    # owner labels the cluster -> engine invalidates -> next call says MATCHED
    pid = store.upsert_person(name="Rahul")
    store.label_face_cluster(r1.cluster_id, pid)
    assert res.invalidate(cluster_id=r1.cluster_id) == 1
    r = res.resolve(track(2), [face(at_cos(u, 0.9), tid=2)])
    assert r.decision is Decision.MATCHED and r.identity.name == "Rahul"


def test_cache_is_bounded(store):
    res = IdentityResolver(store, max_cache=3)
    for tid in range(5):
        res.resolve(track(tid), [])
    assert res.cached(0) is None and res.cached(4) is not None


def test_bad_thresholds():
    with pytest.raises(ValueError):
        IdentityResolver(InMemoryGraphStore(), match_t=0.3, ask_t=0.4)
