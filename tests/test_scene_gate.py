import numpy as np

from worldcam.agent.api import Profile
from worldcam.perception.eventgate import ActivityEventGate, looks_on_floor, posture
from worldcam.pipeline import BestSnapshot, Track, TrackState, TrackUpdate

UPRIGHT = (0.4, 0.2, 0.5, 0.8)
FLAT = (0.2, 0.7, 0.6, 0.95)   # wide + low


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def trk(tid=1, box=UPRIGHT, state=TrackState.ACTIVE, zones=(), first=0.0, last=1.0, best=False, label="person"):
    b = BestSnapshot(1, 0.0, box, 0.9, np.zeros((4, 4, 3), np.uint8)) if best else None
    return Track(tid, label, state, box, first, last, zones=zones, best=b)


def upd(*tracks, ended=()):
    return TrackUpdate(list(tracks), list(ended))


def gate(clock=None, profiles=("companion",), **kw):
    ps = [Profile(name=n, description="") for n in profiles]
    return ActivityEventGate(ps, clock=clock or Clock(), **kw)


def test_posture_heuristic():
    assert posture(UPRIGHT) == "upright"
    assert posture(FLAT) == "horizontal"
    assert looks_on_floor(FLAT) and not looks_on_floor(UPRIGHT)
    assert not looks_on_floor((0.2, 0.1, 0.6, 0.35))   # wide but high in frame


def test_fires_on_entered_then_quiet():
    g = gate()
    assert g.should_fire(upd(trk()), [], {})
    assert g.last_decision.reasons == ("entered",)
    for _ in range(50):                       # no state change: never fires on a per-frame basis
        assert not g.should_fire(upd(trk()), [], {})


def test_ignores_unwatched_labels():
    assert not gate().should_fire(upd(trk(label="laptop")), [], {})


def test_state_change_triggers():
    c = Clock()
    g = gate(c, track_cooldown_s=0)
    assert g.should_fire(upd(trk()), [], {})
    assert g.should_fire(upd(trk(zones=("desk",))), [], {})
    assert g.last_decision.reasons == ("new_zone",)
    assert not g.should_fire(upd(trk(zones=("desk",))), [], {})
    assert not g.should_fire(upd(trk(zones=())), [], {})            # leaving a zone is not a trigger
    assert not g.should_fire(upd(trk(state=TrackState.STATIONARY)), [], {})
    assert g.should_fire(upd(trk(state=TrackState.ACTIVE)), [], {})
    assert g.last_decision.reasons == ("became_active",)
    assert g.should_fire(upd(trk(best=True)), [], {})
    assert g.last_decision.reasons == ("best_snapshot",)
    assert not g.should_fire(upd(trk(best=True)), [], {})           # once


def test_dwell_thresholds_once_each():
    g = gate(track_cooldown_s=0, dwell_thresholds_s=(30, 120))
    g.should_fire(upd(trk(last=1)), [], {})
    assert g.should_fire(upd(trk(last=31)), [], {})
    assert g.last_decision.reasons == ("dwell_30s",)
    assert not g.should_fire(upd(trk(last=60)), [], {})
    assert g.should_fire(upd(trk(last=130)), [], {})
    assert g.last_decision.reasons == ("dwell_120s",)


def test_cooldown_per_track():
    c = Clock()
    g = gate(c, track_cooldown_s=20)
    assert g.should_fire(upd(trk()), [], {})
    assert not g.should_fire(upd(trk(zones=("desk",))), [], {})     # suppressed by cooldown
    assert g.stats["suppressed_cooldown"] == 1
    assert g.should_fire(upd(trk(2)), [], {})                       # other track not affected
    c.t += 25
    assert g.should_fire(upd(trk(zones=("desk", "door"))), [], {})


def test_ended_track_forgotten():
    g = gate()
    t = trk()
    g.should_fire(upd(t), [], {})
    g.should_fire(upd(ended=[t]), [], {})
    assert g.should_fire(upd(trk()), [], {})      # same id reappears = entered again


def test_global_budget_and_window_reset():
    c = Clock()
    g = gate(c, global_budget_per_hour=3, track_cooldown_s=0)
    assert [g.should_fire(upd(trk(i)), [], {}) for i in range(5)] == [True, True, True, False, False]
    assert g.stats["suppressed_budget"] == 2 and g.remaining()["global"] == 0
    c.t += 3601
    assert g.should_fire(upd(trk(9)), [], {})


def test_profile_budget():
    g = gate(profiles=("companion", "cctv"), profile_budget_per_hour={"cctv": 1}, track_cooldown_s=0)
    assert g.should_fire(upd(trk(1)), [], {})
    assert not g.should_fire(upd(trk(2)), [], {})
    assert g.remaining()["profiles"] == {"companion": 39, "cctv": 0}


def test_priority_floor_overrides_budget_and_cooldown():
    c = Clock()
    g = gate(c, profiles=("medical",), global_budget_per_hour=1, track_cooldown_s=1000,
             priority_budget_per_hour=2, floor_cooldown_s=60)
    assert g.should_fire(upd(trk(1)), [], {})
    assert not g.should_fire(upd(trk(2)), [], {})                   # budget exhausted
    assert g.should_fire(upd(trk(1, FLAT)), [], {})                 # priority bypasses
    assert g.last_decision.priority and "possible_floor" in g.last_decision.reasons
    assert not g.should_fire(upd(trk(1, FLAT)), [], {})             # floor cooldown
    c.t += 61
    assert g.should_fire(upd(trk(1, FLAT)), [], {})
    c.t += 61
    assert not g.should_fire(upd(trk(1, FLAT)), [], {})             # priority hard ceiling (2/h)


def test_floor_ignored_without_priority_profile():
    g = gate(profiles=("companion",), track_cooldown_s=0)
    g.should_fire(upd(trk(1)), [], {})
    assert not g.should_fire(upd(trk(1, FLAT)), [], {})


def test_memory_bounded():
    g = gate(max_tracks=5, track_cooldown_s=0, global_budget_per_hour=10**6, default_profile_budget=10**6)
    for i in range(50):
        g.should_fire(upd(trk(i)), [], {})
    assert len(g._mem) == 5
