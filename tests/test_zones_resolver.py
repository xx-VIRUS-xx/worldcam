import json

import cv2
import numpy as np
import pytest

from worldcam.cli.scene_replay import main as replay_main
from worldcam.perception.zones import PolygonZoneResolver, ZoneConfigError, load_zones, parse_zones
from worldcam.pipeline import Track, TrackState, Zone

DESK = Zone("desk", ((0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)), "cam1")
DOOR = Zone("door", ((0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)), "cam1")
OTHER = Zone("desk", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), "cam2")


def trk(box, tid=1):
    return Track(tid, "person", TrackState.ACTIVE, box, 0.0, 1.0)


def test_foot_point_not_center():
    r = PolygonZoneResolver("cam1")
    # center (0.25, 0.3) is above the desk, but the foot (0.25, 0.6) is inside
    assert r.assign(trk((0.2, 0.0, 0.3, 0.6)), [DESK, DOOR]) == ("desk",)


def test_camera_filter():
    r = PolygonZoneResolver("cam1")
    assert r.assign(trk((0.9, 0.2, 0.95, 0.9)), [OTHER, DOOR]) == ("door",)
    assert PolygonZoneResolver("cam2").assign(trk((0.9, 0.2, 0.95, 0.9)), [DESK, DOOR, OTHER]) == ("desk",)


def test_hysteresis_no_flicker():
    r = PolygonZoneResolver("cam1", margin=0.03)
    inside = trk((0.2, 0.3, 0.3, 0.6))
    assert r.assign(inside, [DESK]) == ("desk",)
    # foot drifts 0.01 above the edge: still in (within margin)
    assert r.assign(trk((0.2, 0.3, 0.3, 0.49)), [DESK]) == ("desk",)
    # far outside: leaves
    assert r.assign(trk((0.2, 0.1, 0.3, 0.4)), [DESK]) == ()
    # and re-entering requires being strictly inside, not just near
    assert r.assign(trk((0.2, 0.1, 0.3, 0.49)), [DESK]) == ()
    assert r.assign(trk((0.2, 0.1, 0.3, 0.55)), [DESK]) == ("desk",)


def test_tracks_independent_and_release():
    r = PolygonZoneResolver("cam1", margin=0.03)
    r.assign(trk((0.2, 0.3, 0.3, 0.6), 1), [DESK])
    assert r.assign(trk((0.2, 0.3, 0.3, 0.49), 2), [DESK]) == ()   # track 2 never entered
    r.release(1)
    assert r.assign(trk((0.2, 0.3, 0.3, 0.49), 1), [DESK]) == ()


def test_state_bounded():
    r = PolygonZoneResolver(max_tracks=3)
    for i in range(10):
        r.assign(trk((0.2, 0.3, 0.3, 0.6), i), [DESK])
    assert len(r._current) == 3


def test_loader_ok(tmp_path):
    p = tmp_path / "z.json"
    p.write_text(json.dumps({"cam": {"desk": [[0, 0], [1, 0], [1, 1]]}}))
    z = load_zones(p)
    assert z[0].camera == "cam" and z[0].name == "desk" and len(z[0].polygon) == 3


@pytest.mark.parametrize("bad", [
    [], {}, {"c": []}, {"c": {"z": [[0, 0], [1, 1]]}},
    {"c": {"z": [[0, 0], [1, 0], [2, 1]]}},
    {"c": {"z": [[0, 0], [1, 0], ["a", 1]]}},
    {"c": {"z": [[0, 0], [1, 0], [True, 1]]}},
    {"c": {"z": [[0, 0], [0.5, 0.5], [1, 1]]}},      # degenerate
    {"c": {"": [[0, 0], [1, 0], [1, 1]]}},
    {"c": {"z": [[0, 0], [1, 0], [1]]}},
])
def test_loader_rejects(bad):
    with pytest.raises(ZoneConfigError):
        parse_zones(bad)


def test_loader_bad_json(tmp_path):
    p = tmp_path / "z.json"
    p.write_text("{nope")
    with pytest.raises(ZoneConfigError):
        load_zones(p)


def test_example_file_valid():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert len(load_zones(root / "zones.example.json")) == 3


def test_overlay_cli(tmp_path):
    img = tmp_path / "f.jpg"
    cv2.imwrite(str(img), np.zeros((120, 160, 3), np.uint8))
    z = tmp_path / "z.json"
    z.write_text(json.dumps({"default": {"desk": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]}}))
    out = tmp_path / "o.jpg"
    assert replay_main(["--overlay", str(img), "--zones", str(z), "--out", str(out)]) == 0
    assert out.exists() and cv2.imread(str(out)).any()
    assert replay_main(["--overlay", str(tmp_path / "missing.jpg"), "--zones", str(z)]) == 2
