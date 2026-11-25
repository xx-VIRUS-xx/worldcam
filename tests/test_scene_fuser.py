import json

from worldcam.brain.api import EVENT_KINDS, InMemoryEventStore
from worldcam.perception.fuser import RuleActivityFuser, load_rules
from worldcam.pipeline import ActivityHypothesis, Track, TrackState

UP = (0.4, 0.2, 0.5, 0.8)
FLAT = (0.2, 0.7, 0.6, 0.95)


def trk(zones=("desk",), box=UP, state=TrackState.ACTIVE, tid=1):
    return Track(tid, "person", state, box, 0.0, 1.0, zones=zones)


def hyp(a, c=0.8):
    return ActivityHypothesis(a, (), c)


f = RuleActivityFuser()


def test_coding_needs_desk_laptop_dwell():
    assert f.fuse(hyp("coding"), trk(), ["laptop"], 200) == ("coding", 0.9)
    assert "ok:" in f.last.reasons[1] and f.last.corroborated
    for t, objs, dw in [(trk(("sofa",)), ["laptop"], 200), (trk(), [], 200), (trk(), ["laptop"], 30)]:
        a, c = f.fuse(hyp("coding"), t, objs, dw)
        assert a == "unknown" and c <= 0.3
        assert any(r.startswith("missing:") for r in f.last.reasons)
        assert f.last.vlm_activity == "coding"


def test_taking_medication_needs_signal_and_vlm_agreement():
    assert f.fuse(hyp("taking_medication", 0.9), trk(()), ["pill_bottle"], 5)[0] == "taking_medication"
    assert f.fuse(hyp("taking_medication", 0.9), trk(()), [], 5)[0] == "unknown"
    a, c = f.fuse(hyp("taking_medication", 0.4), trk(()), ["pill_bottle"], 5)
    assert a == "unknown" and "vlm_conf_below:0.6" in f.last.reasons
    assert f.fuse(hyp("taking_medication", 0.9), trk(("medication",)), [], 5)[0] == "taking_medication"
    assert f.fuse(hyp("taking_medication", 1.0), trk(()), ["medication"], 5)[1] <= 0.75   # max_conf


def test_fall_requires_horizontal_box():
    assert f.fuse(hyp("fall_suspected", 0.9), trk(box=FLAT), [], 5)[0] == "fall_suspected"
    assert f.fuse(hyp("fall_suspected", 0.9), trk(box=UP), [], 5)[0] == "unknown"


def test_contradiction_lowers_confidence():
    a, c = f.fuse(hyp("sitting", 0.8), trk(box=FLAT), [], 5)
    assert a == "sitting" and c == 0.4 and not f.last.corroborated
    assert any(r.startswith("contradiction") for r in f.last.reasons)
    a, c = f.fuse(hyp("walking", 0.8), trk(state=TrackState.STATIONARY), [], 5)
    assert c == 0.4


def test_soft_rule_caps_instead_of_rejecting():
    a, c = f.fuse(hyp("working_at_desk", 0.8), trk(("sofa",)), [], 100)
    assert a == "working_at_desk" and abs(c - 0.48) < 1e-9


def test_unknown_and_unruled_pass_through():
    assert f.fuse(hyp("unknown", 0.2), trk(), [], 0) == ("unknown", 0.2)
    assert f.fuse(hyp("cooking", 0.7), trk(), [], 0) == ("cooking", 0.7)
    assert "no_rule:vlm_only" in f.last.reasons


def test_custom_rules_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"cooking": {"requires": [{"zone_any": ["kitchen"]}], "on_fail": "reject"}}))
    g = RuleActivityFuser(load_rules(p))
    assert g.fuse(hyp("cooking"), trk(("kitchen",)), [], 0)[0] == "cooking"
    assert g.fuse(hyp("cooking"), trk(("desk",)), [], 0)[0] == "unknown"


def test_event_source_and_cap():
    g = RuleActivityFuser()
    res = g.fuse_detailed(hyp("coding", 0.8), trk(), ["laptop"], 200)
    ev = g.to_event(res, trk(), camera="cam1", subject_id="p1", profiles=("companion",))
    assert ev.kind == "activity_changed" and ev.kind in EVENT_KINDS
    assert ev.source == "camera" and ev.confidence == 0.9 and ev.zone == "desk"
    assert ev.detail["reasons"] and ev.detail["vlm_activity"] == "coding"
    res2 = g.fuse_detailed(hyp("cooking", 0.9), trk(), [], 0)       # VLM-only
    ev2 = g.to_event(res2, trk())
    assert ev2.source == "inferred" and ev2.confidence == 0.6
    InMemoryEventStore().add_event(ev), InMemoryEventStore().add_event(ev2)   # valid for the store


def test_event_only_on_change():
    g = RuleActivityFuser()
    t = trk()
    r1 = g.fuse_detailed(hyp("cooking"), t, [], 0)
    assert g.event_if_changed(r1, t) is not None
    assert g.event_if_changed(r1, t) is None
    r2 = g.fuse_detailed(hyp("reading"), t, [], 100)
    assert g.event_if_changed(r2, t).activity == "reading"
    r3 = g.fuse_detailed(hyp("unknown", 0.1), t, [], 0)
    assert g.event_if_changed(r3, t) is None
