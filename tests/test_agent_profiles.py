import copy
import json
import time

import pytest

from worldcam.agent.profiles import (ProfileError, ProfileSet, builtin_names, load_builtin, load_profile,
                                     merge_profiles, profile_from_dict, render, rule_matches_event,
                                     validate_profile_dict)
from worldcam.brain.api import new_event


def local(h, day=10):
    return time.mktime((2026, 11, day, h, 0, 0, 0, 0, -1))


def test_builtins_load_and_validate():
    assert builtin_names() == ["cctv", "companion", "medical"]
    for n in builtin_names():
        p = load_builtin(n)
        assert p.name == n and p.goals and p.description
        assert all(r["to_permission"].startswith("receive_") for r in p.alert_rules)
    med = load_builtin("medical")
    assert not med.send_images_to_cloud and "caregiver" in med.allowed_roles
    assert {r["priority"] for r in med.alert_rules} <= {2, 3}
    text = json.dumps(med.__dict__, default=str).lower()
    assert "diagnos" in text and "may be wrong" in text      # states its limits


def test_validation_errors():
    bad = {"name": "x", "alert_rules": [{"when": {"bogus": 1}, "to_permission": "nope", "priority": 9, "text": ""}],
           "checkins": [{"every_hours": 0, "prompt": ""}], "allowed_roles": ["king"], "surprise": 1}
    errs = validate_profile_dict(bad)
    assert len(errs) >= 7
    with pytest.raises(ProfileError):
        profile_from_dict(bad)


def test_load_from_file_and_custom(tmp_path):
    d = json.loads(open(__import__("worldcam.agent.profiles", fromlist=["x"]).BUILTIN_DIR / "cctv.json").read())
    d["name"] = "shop"
    f = tmp_path / "shop.json"
    f.write_text(json.dumps(d))
    assert load_profile(f).name == "shop"


def test_merge_union_and_strictest_privacy():
    a, b, c = (load_builtin(n) for n in ("cctv", "medical", "companion"))
    a2 = profile_from_dict({**json.loads(json.dumps({"name": "cloudy"})), "send_images_to_cloud": True,
                            "retention_days": 7, "goals": ["g1", "g1"], "notice_labels": ["person", "zebra"]})
    m = merge_profiles([a, b, c, a2])
    assert m.name == "cctv+medical+companion+cloudy"
    assert set(a.notice_labels) | set(b.notice_labels) | {"zebra"} <= set(m.notice_labels)
    assert len(m.notice_labels) == len(set(m.notice_labels))
    assert m.send_images_to_cloud is False                      # one profile forbids it
    assert merge_profiles([a2]).send_images_to_cloud is True
    assert m.retention_days == 7
    assert len(m.alert_rules) == len(a.alert_rules) + len(b.alert_rules)
    assert {r["profile"] for r in m.alert_rules} == {"cctv", "medical"}
    assert merge_profiles([profile_from_dict({"name": "off", "enabled": False})]).enabled is False


def test_rule_matching_unknown_person_at_door_at_night():
    rule = next(r for r in load_builtin("cctv").alert_rules if r["id"].endswith("unknown_person_door_night"))
    ev = lambda h, **kw: new_event("seen", ts=local(h), subject_label="unknown person", zone="front_door",
                                   confidence=0.8, **kw)
    assert rule_matches_event(rule, ev(2))
    assert not rule_matches_event(rule, ev(14))                                   # daytime
    assert not rule_matches_event(rule, new_event("seen", ts=local(2), subject_id="p1", subject_label="Rahul",
                                                  zone="front_door", confidence=0.9))   # known
    assert not rule_matches_event(rule, new_event("seen", ts=local(2), subject_label="unknown", zone="kitchen",
                                                  confidence=0.9))                       # elsewhere
    assert not rule_matches_event(rule, new_event("alert", ts=local(2)))


def test_render_is_safe():
    assert render("at {zone} {nope}", zone="door") == "at door ?"
    assert render("bad {", zone="x") == "bad {"


def test_profileset_rules_from_active_only():
    ps = ProfileSet([load_builtin("cctv"), load_builtin("medical")])
    assert ps.names() == ["cctv", "medical"] and len(ps.checkins) == 2
