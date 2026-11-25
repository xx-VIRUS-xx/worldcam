"""EntityObserver: recognise animals/objects before asking (no camera, no models, no network).

A fake embedder maps a crop (an 8x8 image whose first byte is a key) to a chosen unit vector,
so the scenarios control the cosine between sightings exactly.
"""
import numpy as np
import pytest

from worldcam.ambiguity.entities import DEFAULT_ENTITY_LABELS, EntityObserver
from worldcam.ambiguity.triggers import DoubtDetector, TriggerConfig
from worldcam.graph.api import InMemoryGraphStore
from worldcam.perception.reid import ReidThresholds
from worldcam.pipeline import BestSnapshot, Track, TrackState

DIM = 64
TH = ReidThresholds(match_t=0.90, ask_t=0.80, cluster_t=0.88)


def unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)


def mix(a, b, cos):
    """Unit vector with cosine `cos` to a (b supplies an orthogonal-ish direction)."""
    o = b - np.dot(a, b) * a
    o /= np.linalg.norm(o)
    return cos * a + np.sqrt(1 - cos * cos) * o


class FakeEmbedder:
    thresholds = TH

    def __init__(self):
        self.vecs = {}
        self.calls = 0

    def embed(self, crop):
        self.calls += 1
        return self.vecs[int(crop[0, 0, 0])]


def crop(key):
    c = np.zeros((8, 8, 3), np.uint8)
    c[0, 0, 0] = key
    return c


class Rig:
    def __init__(self, tmp_path, **cfg):
        self.store = InMemoryGraphStore()
        self.emb = FakeEmbedder()
        self.det = DoubtDetector(self.store, TriggerConfig(**cfg))
        self.obs = EntityObserver(self.store, self.det, embedder=self.emb, snapshot_dir=str(tmp_path),
                                  labels=("cup", "dog", "cat"), cooldown_s=300.0)
        self.seq = 0

    def see(self, key, ts, label="cup", track_id=1):
        self.seq += 1
        ts += 1000.0                                  # ts 0 would read as 'unset'
        best = BestSnapshot(seq=self.seq, ts=ts, box=(0, 0, 1, 1), score=0.9, crop=crop(key))
        t = Track(track_id, label, TrackState.STATIONARY, (0, 0, 1, 1), ts, ts, zones=("desk",), best=best)
        return self.obs.observe(t)

    def answer(self, did, name, label="cup", entity_type="object", kind="belonging"):
        """What the worker does after the owner answers: label the cluster, resolve the doubt."""
        d = self.store.get_doubt(did)
        cid = next(i for i in d.about_ids if i.startswith("ec_"))
        pid = self.store.upsert_person(name=name, kind=kind, entity_type=entity_type, label=label)
        self.store.label_face_cluster(cid, pid)       # the one labeling tool covers entity clusters
        self.store.update_doubt(did, status="resolved", answer=name)
        return pid, cid


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.emb.vecs[1] = unit(1)
    r.emb.vecs[2] = unit(2)
    r.emb.vecs[3] = unit(3)
    return r


def test_default_labels_are_few_and_cups_are_opt_in():
    assert set(DEFAULT_ENTITY_LABELS) == {"cat", "dog", "backpack", "handbag", "suitcase", "laptop"}


def test_repeat_ask_regression_answered_entity_is_not_asked_again(rig):
    """Reproduced bug: question 1 answered, a new question opened 400 s later."""
    assert rig.see(1, 0.0) is None                    # one detection is noise
    did = rig.see(1, 400.0)                           # second sighting of the same cup -> ask once
    assert did is not None and rig.store.get_doubt(did).kind == "unknown_entity"
    pid, cid = rig.answer(did, "my blue mug")
    assert rig.see(1, 800.0) is None                  # 400 s later, same object: NO new doubt
    assert rig.see(1, 5000.0) is None
    assert rig.store.list_doubts() == [rig.store.get_doubt(did)]
    s = rig.store.recent_sightings(person_id=pid, limit=10)
    assert len(s) >= 3 and all(x.cluster_id == cid for x in s)   # sightings of the KNOWN entity
    assert rig.obs.stats["matched"] == 2


def test_one_doubt_per_cluster_while_open(rig):
    rig.see(1, 0.0)
    did = rig.see(1, 400.0)
    assert did
    assert rig.see(1, 800.0) is None and rig.see(1, 1200.0) is None
    assert len(rig.store.list_doubts()) == 1


def test_different_object_same_label_gets_its_own_doubt(rig):
    rig.see(1, 0.0)
    d1 = rig.see(1, 400.0)
    rig.answer(d1, "blue mug")
    # a visibly different cup (cosine to mug 1 well below ask_t)
    assert rig.see(2, 1000.0, track_id=2) is None
    d2 = rig.see(2, 1500.0, track_id=2)
    assert d2 and d2 != d1
    assert rig.store.get_doubt(d2).about_ids[0] != rig.store.get_doubt(d1).about_ids[0]


def test_label_scoping_cup_never_matches_dog(rig):
    rig.see(1, 0.0, label="dog")
    did = rig.see(1, 400.0, label="dog")
    rig.answer(did, "Bruno", label="dog", entity_type="animal", kind="pet")
    # the identical embedding under another label is a different entity
    assert rig.see(1, 1000.0, label="cat", track_id=2) is None
    assert len({c["label"] for c in rig.store.entity_clusters.values()}) == 2
    dog_clusters = [c for c in rig.store.entity_clusters.values() if c["label"] == "dog"]
    assert len(dog_clusters) == 1


def test_low_confidence_between_thresholds_names_candidate(rig):
    rig.see(1, 0.0)
    d1 = rig.see(1, 400.0)
    pid, cid = rig.answer(d1, "blue mug")
    v = mix(unit(1), unit(9), 0.85)                   # in [ask_t, match_t)
    rig.emb.vecs[4] = v
    assert rig.see(4, 1000.0, track_id=2) is None     # first sighting of the new cluster: noise
    did = rig.see(4, 1500.0, track_id=2)              # second: ask, naming the candidate
    d = rig.store.get_doubt(did)
    assert d.kind == "low_confidence" and d.detail["candidate_person_id"] == pid
    assert d.detail["entity_label"] == "cup" and d.about_ids[1] == pid
    assert rig.see(4, 2000.0, track_id=2) is None     # deduped by cluster
    # the doubtful sighting was NOT silently attributed to the known entity
    assert all(s.person_id is None for s in rig.store.recent_sightings(cluster_id=d.about_ids[0]))


def test_conflict_when_two_known_entities_both_match(rig):
    base = unit(1)
    rig.emb.vecs[1] = base
    rig.emb.vecs[2] = mix(base, unit(7), 0.80)        # a second, similar-looking cup
    for key, name, t in ((1, "mug A", 0.0), (2, "mug B", 2000.0)):
        rig.see(key, t, track_id=key)
        did = rig.see(key, t + 400.0, track_id=key)
        assert did, key
        rig.answer(did, name)
    # a view almost equidistant from both known mugs
    mid = base + rig.emb.vecs[2]
    mid /= np.linalg.norm(mid)
    rig.emb.vecs[5] = mid
    assert float(mid @ base) >= TH.ask_t and abs(float(mid @ base) - float(mid @ rig.emb.vecs[2])) < 0.03
    assert rig.see(5, 5000.0, track_id=9) is None
    did = rig.see(5, 5500.0, track_id=9)
    d = rig.store.get_doubt(did)
    assert d.kind == "conflict" and len([i for i in d.about_ids if i.startswith("p_")]) == 2


def test_cluster_grows_and_thresholds_use_the_cluster_centroid(rig):
    rig.see(1, 0.0)
    rig.see(1, 400.0)
    cid = next(iter(rig.store.entity_clusters))
    assert rig.store.entity_cluster_size(cid) == 2
    near = mix(unit(1), unit(8), 0.95)                # same cup from a slightly different angle
    rig.emb.vecs[6] = near
    rig.see(6, 1000.0, track_id=3)
    assert rig.store.entity_cluster_size(cid) == 3 and len(rig.store.entity_clusters) == 1


def test_cooldown_for_stationary_repeats(rig):
    rig.see(1, 0.0)
    n = rig.emb.calls
    assert rig.see(1, 100.0) is None                  # same track within cooldown: not even embedded
    assert rig.emb.calls == n
    assert rig.obs.stats["cooldown_skips"] == 1
    assert rig.see(1, 200.0, track_id=2) is None      # a new track of the same cluster within cooldown
    assert len(rig.store.recent_sightings(limit=10)) == 1


def test_daily_cap_on_new_entity_doubts(tmp_path):
    r = Rig(tmp_path, entity_max_new_doubts_per_day=2)
    made = []
    for i in range(1, 5):
        r.emb.vecs[i] = unit(100 + i)
        r.see(i, i * 1000.0, track_id=i)
        made.append(r.see(i, i * 1000.0 + 400.0, track_id=i))
    assert [bool(x) for x in made] == [True, True, False, False]


def test_ignore_cooldown_is_per_cluster(rig):
    rig.see(1, 0.0)
    did = rig.see(1, 400.0)
    d = rig.store.get_doubt(did)
    rig.store.update_doubt(did, status="dismissed", detail={**d.detail, "ignored_by_owner": True, "dismissed_at": 500.0})
    assert rig.see(1, 1000.0) is None
    assert rig.see(1, 90000.0) is None                 # within the 30 day ignore cooldown
    rig.emb.vecs[2] = unit(2)
    rig.see(2, 2000.0, track_id=4)
    assert rig.see(2, 2400.0, track_id=4) is not None  # a different object is still asked about


def test_forget_removes_entity_data(rig):
    rig.see(1, 0.0)
    did = rig.see(1, 400.0)
    pid, cid = rig.answer(did, "blue mug")
    rig.see(1, 900.0)
    rig.store.forget_person(pid)
    assert rig.store.entity_clusters == {} and rig.store.entity_views == {}
    assert rig.store.recent_sightings() == [] and rig.store.list_doubts() == []


def test_embedder_failure_is_swallowed(tmp_path):
    r = Rig(tmp_path)                                  # key 7 has no vector -> KeyError inside embed
    assert r.see(7, 0.0) is None
    assert r.obs.stats["embed_errors"] == 1
