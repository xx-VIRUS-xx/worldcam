"""Owner enrolment, the self-interview, and the animal false-positive filter."""
import numpy as np

from worldcam.agent.profiles import ProfileSet, load_profiles
from worldcam.agent.uncertainty import OWNER_INTERVIEW, UncertaintyManager
from worldcam.cli.enroll import adopt_cluster, link_owner, pick_user, store_faces
from worldcam.graph.api import Doubt, InMemoryGraphStore
from worldcam.perception.filters import FilteredDetector
from worldcam.pipeline import Detection
from worldcam.types import Frame
from worldcam.users.api import InMemoryUserDirectory


def _world():
    store, users = InMemoryGraphStore(), InMemoryUserDirectory()
    owner = users.add("Pat", "owner", "telegram", 1, profiles=("companion",))
    return store, users, owner


def test_link_owner_creates_person_and_links_user_idempotently():
    store, users, owner = _world()
    pid = link_owner(store, users, owner)
    assert store.get_person(pid).kind == "owner" and store.get_person(pid).name == "Pat"
    assert users.get(owner.id).person_id == pid
    assert link_owner(store, users, users.get(owner.id)) == pid            # no second person
    assert len(store.list_people()) == 1


def test_adopt_cluster_labels_and_closes_open_questions():
    store, users, owner = _world()
    pid = link_owner(store, users, owner)
    rng = np.random.default_rng(1)
    v = rng.normal(size=512)
    _, cid = store.add_face(v, quality=0.5, ts=1.0, snapshot_path=None)
    did = store.create_doubt(Doubt("", "unknown_face", about_ids=(cid,)))
    other = store.create_doubt(Doubt("", "unknown_face", about_ids=("fc_other",)))
    assert adopt_cluster(store, pid, cid) == [did]
    assert store.get_doubt(did).status == "resolved" and store.get_doubt(other).status == "open"
    assert store.match_face(v, k=1)[0].person_id == pid                   # you are recognised from now on


def test_store_faces_enrols_a_cluster_for_the_person():
    store, users, owner = _world()
    pid = link_owner(store, users, owner)
    rng = np.random.default_rng(2)
    base = rng.normal(size=512)

    class F:
        def __init__(self, e):
            self.embedding, self.quality, self.ts, self.crop_path = e, 0.6, 1.0, None
    faces = [F(base + rng.normal(scale=0.05, size=512)) for _ in range(6)]
    clusters = store_faces(store, pid, faces)
    assert len(clusters) == 1 and store.match_face(base, k=1)[0].person_id == pid


def test_pick_user_requires_an_owner():
    store, users, owner = _world()
    users.add("Gus", "guest", "telegram", 2)
    assert pick_user(users, None).id == owner.id
    try:
        pick_user(users, "Gus")
        assert False, "guest must not enrol"
    except SystemExit:
        pass


def _um(store, users, profiles=("companion",)):
    return UncertaintyManager(store, users, profiles=ProfileSet(load_profiles(list(profiles))))


def test_interview_waits_for_account_link_then_asks_one_question_at_a_time():
    store, users, owner = _world()
    pid = store.upsert_person(None, name="Pat", kind="owner")
    um = _um(store, users)
    assert [g for g in um.knowledge_gaps(1000.0) if g.kind == "self_profile"] == []   # not linked yet
    users.set_person(owner.id, pid)
    g = next(g for g in um.knowledge_gaps(1000.0) if g.kind == "self_profile")
    assert g.topic == "own_facts" and g.extra["fact_key"] == "areas"   # the name is known, so the interview starts with the rooms
    d = um.gap_to_doubt(g, 1000.0)
    assert d.detail["gap_kind"] == "self_profile" and d.detail["fact_key"] == g.extra["fact_key"]
    store.create_doubt(d)
    assert [x for x in um.knowledge_gaps(1001.0) if x.kind == "self_profile"] == []   # one at a time
    assert um.ask_target(d, 1000.0).id == owner.id                                    # about_self works


def test_interview_advances_after_an_answer_and_respects_spacing_and_profiles():
    store, users, owner = _world()
    pid = store.upsert_person(None, name="Pat", kind="owner")
    users.set_person(owner.id, pid)
    um = _um(store, users, ("companion",))
    t0 = 1_800_000_000.0
    first = um.gap_to_doubt(next(g for g in um.knowledge_gaps(t0) if g.kind == "self_profile"), t0)
    did = store.create_doubt(first)
    store.add_fact(pid, first.detail["fact_key"], "desk and bed", source="owner", confidence=0.9)
    store.update_doubt(did, status="resolved")
    assert [g for g in um.knowledge_gaps(t0 + 60.0) if g.kind == "self_profile"] == []     # too soon
    nxt = [g for g in um.knowledge_gaps(t0 + 5 * 3600.0) if g.kind == "self_profile"]
    assert nxt and nxt[0].extra["fact_key"] != first.detail["fact_key"]
    # medical questions only for the medical profile, and only after consent
    keys = {g.extra["fact_key"] for g in nxt}
    assert "care_routines" not in keys and "consent_health_tracking" not in keys


def test_medical_interview_needs_consent_first():
    store, users, owner = _world()
    pid = store.upsert_person(None, name="Pat", kind="owner")
    users.set_person(owner.id, pid)
    um = _um(store, users, ("medical",))
    for k in ("areas", "daily_rhythm", "privacy_preferences"):
        store.add_fact(pid, k, "x", source="owner", confidence=0.9)
    g = next(g for g in um.knowledge_gaps(0.0) if g.kind == "self_profile")
    assert g.extra["fact_key"] == "consent_health_tracking"
    store.add_fact(pid, "consent_health_tracking", "no thanks", source="owner", confidence=0.9)
    assert [x for x in um.knowledge_gaps(0.0) if x.kind == "self_profile"] == []      # declined: never asked
    store.add_fact(pid, "consent_health_tracking", "yes please", source="owner", confidence=0.9)


class _Inner:
    def __init__(self, dets):
        self.dets, self.device = dets, "mps"

    def detect(self, frame, regions=None):
        return self.dets


def _frame():
    return Frame(1, 0.0, np.zeros((10, 10, 3), np.uint8))


def test_filter_drops_low_confidence_animals_and_animals_inside_people():
    person = Detection("person", 0.9, (0.2, 0.1, 0.8, 0.9))
    face_cat = Detection("cat", 0.9, (0.3, 0.2, 0.6, 0.5))              # inside the person: a body part
    blanket = Detection("cat", 0.45, (0.0, 0.6, 0.2, 0.9))              # low confidence, elsewhere
    real_dog = Detection("dog", 0.8, (0.85, 0.6, 1.0, 0.95))            # confident, outside the person
    laptop = Detection("laptop", 0.4, (0.0, 0.0, 0.1, 0.1))             # objects keep the normal bar
    f = FilteredDetector(_Inner([person, face_cat, blanket, real_dog, laptop]), {"cat": 0.6, "dog": 0.6})
    assert [d.label for d in f.detect(_frame())] == ["person", "dog", "laptop"]
    assert f.dropped == {"low_conf": 1, "inside_person": 1} and f.device == "mps"
