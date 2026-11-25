from worldcam.ambiguity.triggers import DoubtDetector, IdentitySignal, TriggerConfig
from worldcam.graph.api import InMemoryGraphStore, Sighting, new_id

import pytest


def sight(store, *, cluster_id=None, person_id=None, ts=1000.0):
    store.add_sighting(Sighting(new_id("s"), ts, "door", 1, person_id=person_id, cluster_id=cluster_id))


@pytest.fixture
def store():
    return InMemoryGraphStore()


def test_unknown_face_needs_three_sightings_then_dedups(store):
    det = DoubtDetector(store)
    sig = IdentitySignal("unknown_cluster", track_id=1, cluster_id="fc_a", crop_path="/x/a.jpg", ts=1)
    for i in range(2):
        sight(store, cluster_id="fc_a")
        assert det.observe(sig, now=10) is None
    sight(store, cluster_id="fc_a")
    did = det.observe(sig, now=10)
    d = store.get_doubt(did)
    assert d.kind == "unknown_face" and d.about_ids == ("fc_a",) and d.snapshot_paths == ("/x/a.jpg",)
    sight(store, cluster_id="fc_a")
    assert det.observe(sig, now=11) is None            # open doubt -> dedup
    for st in ("asked", "answered"):
        store.update_doubt(did, status=st)
        assert det.observe(sig, now=12) is None
    store.update_doubt(did, status="resolved")
    assert det.observe(sig, now=13) is not None       # only active doubts block


def test_new_face_counts_like_unknown_cluster(store):
    det = DoubtDetector(store)
    sight(store, cluster_id="fc_n")
    assert det.observe(IdentitySignal("new_face", cluster_id="fc_n"), now=1) is None


def test_ignored_cluster_has_cooldown(store):
    det = DoubtDetector(store, TriggerConfig(ignore_cooldown_days=1))
    for _ in range(3):
        sight(store, cluster_id="fc_i")
    did = det.observe(IdentitySignal("unknown_cluster", cluster_id="fc_i"), now=0)
    store.update_doubt(did, status="dismissed", detail={"ignored_by_owner": True, "dismissed_at": 0})
    assert det.observe(IdentitySignal("unknown_cluster", cluster_id="fc_i"), now=3600) is None
    assert det.observe(IdentitySignal("unknown_cluster", cluster_id="fc_i"), now=2 * 86400) is not None


def test_low_confidence_band(store):
    pid = store.upsert_person(name="Prabhat", kind="household")
    det = DoubtDetector(store)
    assert det.observe(IdentitySignal("low_confidence", cluster_id="fc_1", candidates=((pid, 0.30),))) is None
    assert det.observe(IdentitySignal("low_confidence", cluster_id="fc_1", candidates=((pid, 0.50),))) is None
    did = det.observe(IdentitySignal("low_confidence", cluster_id="fc_1", candidates=((pid, 0.40),)))
    d = store.get_doubt(did)
    assert d.kind == "low_confidence" and d.about_ids == ("fc_1", pid) and d.detail["candidate_person_id"] == pid
    assert det.observe(IdentitySignal("low_confidence", cluster_id="fc_1", candidates=((pid, 0.41),))) is None


def test_conflict_needs_two_people_and_dedups(store):
    a, b = store.upsert_person(name="A"), store.upsert_person(name="B")
    det = DoubtDetector(store)
    assert det.observe(IdentitySignal("conflict", candidates=((a, 0.6),))) is None
    did = det.observe(IdentitySignal("conflict", track_id=4, candidates=((a, 0.6), (b, 0.55))))
    assert store.get_doubt(did).kind == "conflict"
    assert det.observe(IdentitySignal("conflict", candidates=((b, 0.6), (a, 0.55)))) is None


def test_metadata_gap_after_k_sightings(store):
    pid = store.upsert_person(name=None, kind="visitor")
    det = DoubtDetector(store, TriggerConfig(metadata_min_sightings=5))
    sig = IdentitySignal("matched", person_id=pid)
    for _ in range(4):
        sight(store, person_id=pid)
    assert det.observe(sig) is None
    sight(store, person_id=pid)
    did = det.observe(sig)
    assert store.get_doubt(did).detail["missing"] == ["name", "relation_to_owner"]
    assert det.observe(sig) is None


def test_metadata_gap_not_for_complete_person_or_owner(store):
    pid = store.upsert_person(name="Rahul", kind="visitor")
    store.add_fact(pid, "relation_to_owner", "colleague", source="owner", confidence=0.9)
    owner = store.upsert_person(name=None, kind="owner")
    for _ in range(6):
        sight(store, person_id=pid)
        sight(store, person_id=owner)
    det = DoubtDetector(store)
    assert det.observe(IdentitySignal("matched", person_id=pid)) is None
    assert det.observe(IdentitySignal("matched", person_id=owner)) is None


def test_bad_signal_kind():
    with pytest.raises(ValueError):
        IdentitySignal("weird")
    with pytest.raises(ValueError):
        IdentitySignal("unknown_entity", entity_type="alien")


def test_unknown_entity_and_animal_metadata_gap(store):
    det = DoubtDetector(store, TriggerConfig(entity_min_sightings=1, metadata_min_sightings=3,
                                             ignore_cooldown_days=1))
    store.add_sighting(Sighting("s_1", 10.0, "door", 1, entity_label="backpack", snapshot_path="/x/b.jpg"))
    sig = IdentitySignal("unknown_entity", track_id=1, crop_path="/x/b.jpg", ts=10.0,
                         entity_type="object", entity_label="backpack", entity_key="ek_object_backpack_door",
                         sighting_id="s_1", zones=("door",))
    did = det.observe(sig, now=10.0)
    assert did is not None
    d = store.get_doubt(did)
    assert d.kind == "unknown_entity" and d.about_ids == ("s_1",)
    # Dedup while active
    assert det.observe(sig, now=11.0) is None
    # Cooldown when ignored
    store.update_doubt(did, status="dismissed", detail={**d.detail, "ignored_by_owner": True, "dismissed_at": 11.0})
    assert det.observe(sig, now=100.0) is None
    assert det.observe(sig, now=2 * 86400) is not None

    # Animal metadata_gap checks animal required facts (species_or_breed, relation_to_owner)
    dog_id = store.upsert_person(name="Bruno", kind="pet", entity_type="animal", label="dog")
    for i in range(3):
        sight(store, person_id=dog_id, ts=float(i))
    gap_id = det.observe(IdentitySignal("matched", person_id=dog_id, entity_type="animal", entity_label="dog"))
    assert gap_id is not None
    assert store.get_doubt(gap_id).detail["missing"] == ["species_or_breed", "relation_to_owner"]
