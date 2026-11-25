"""worldcam-run wiring, no camera / models / network: the identity stage with
fake analyzer, the real IdentityResolver + DoubtDetector on InMemoryGraphStore."""
import time

import numpy as np

from worldcam.ambiguity.triggers import DoubtDetector, TriggerConfig
from worldcam.cli.run import IdentityWorker, build_parser, signal_from_result
from worldcam.graph.api import InMemoryGraphStore
from worldcam.identity.resolver import IdentityResolver
from worldcam.pipeline import BestSnapshot, FaceObservation, Track, TrackState


def _track(tid, seq, ts=1.0):
    crop = np.zeros((64, 32, 3), np.uint8)
    best = BestSnapshot(seq=seq, ts=ts, box=(0.1, 0.1, 0.3, 0.6), score=1.0, crop=crop,
                        crop_box=(0.08, 0.05, 0.32, 0.65))
    return Track(tid, "person", TrackState.ACTIVE, best.box, 0.0, ts, best=best)


class FakeAnalyzer:
    """Every track gets the same face (one stranger)."""

    def __init__(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=512)
        self.emb = tuple(float(x) for x in v / np.linalg.norm(v))
        self.calls = 0

    def analyze(self, crop, track, ts):
        self.calls += 1
        return [FaceObservation(track.track_id, ts, (0.15, 0.12, 0.2, 0.2), self.emb, 0.9,
                                f"snap/t{track.track_id}.jpg")]


def _drain(w, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        with w._cv:
            if not w._pending:
                break
        time.sleep(0.01)
    time.sleep(0.05)


def test_stranger_seen_three_times_opens_one_doubt():
    store = InMemoryGraphStore()
    resolver = IdentityResolver(store)
    detector = DoubtDetector(store, TriggerConfig(unknown_min_sightings=3))
    an = FakeAnalyzer()
    w = IdentityWorker(an, resolver, detector)
    w.start()
    try:
        for tid in (1, 2, 3, 4):              # 4 separate visits by the same stranger
            w.submit(_track(tid, seq=10 * tid, ts=float(tid)))
            _drain(w)
            w.submit(_track(tid, seq=10 * tid, ts=float(tid)), ended=True)
            _drain(w)
    finally:
        w.stop()
        w.join(2)
    doubts = store.list_doubts(limit=100)
    assert [d.kind for d in doubts] == ["unknown_face"]   # deduped by cluster
    assert len(store.clusters) == 1 and store.cluster_size(next(iter(store.clusters))) == 4
    assert an.calls == 4                                   # ended re-submit of same seq not re-analyzed
    assert w.stats["errors"] == 0


def test_submit_keeps_only_newest_snapshot_per_track():
    w = IdentityWorker(FakeAnalyzer(), IdentityResolver(InMemoryGraphStore()), None)  # not started
    w.submit(_track(7, seq=1))
    w.submit(_track(7, seq=2))
    w.submit(_track(8, seq=1))
    assert set(w._pending) == {7, 8} and w._pending[7][0].best.seq == 2
    assert w.stats["replaced"] == 1


def test_no_face_and_cached_results_give_no_signal():
    store = InMemoryGraphStore()
    r = IdentityResolver(store)
    res = r.resolve(_track(1, 1), [])
    assert res.decision.value == "no_face" and signal_from_result(res) is None


def test_parser_defaults():
    a = build_parser().parse_args([])
    assert a.source == "0" and a.store == "neo4j" and a.notifier == "telegram" and a.worker == "auto"


def test_run_requires_gemini_key_when_worker_gemini(monkeypatch, capsys):
    from worldcam.cli.run import main
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("worldcam.ambiguity.gemini.load_env", lambda *a, **kw: {})
    assert main(["--worker", "gemini", "--source", "synthetic"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
