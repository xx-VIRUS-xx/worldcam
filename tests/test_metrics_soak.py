import csv

from worldcam import metrics
from worldcam.cli import preview
from worldcam.metrics import STATS_FIELDS, SoakStats, current_rss_mb, peak_rss_mb


def test_rss_helpers_return_plausible_mib():
    cur, peak = current_rss_mb(), peak_rss_mb()
    assert cur is not None and 5 < cur < 4096
    assert peak is not None and peak >= cur * 0.5  # same order of magnitude, MiB not bytes


def test_current_rss_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ps")
    monkeypatch.setattr(metrics.subprocess, "run", boom)
    monkeypatch.setattr(metrics.sys, "platform", "darwin")
    assert current_rss_mb() is None


class _G:
    frames_read = 0
    frames_consumed = 0
    dropped = 0
    read_failures = 0


class _Src:
    reconnects = 2


def test_soak_stats_gaps_fps_and_open_stall():
    s = SoakStats(now=100.0)
    g = _G()
    for i in range(10):  # 10 frames, 0.1 s apart
        s.on_frame(100.0 + i * 0.1, now=100.0 + i * 0.1 + 0.01)
    g.frames_read = 10
    row = s.snapshot(g, _Src(), now=101.0)
    assert set(row) == set(STATS_FIELDS)
    assert row["fps"] == 10.0 and row["read_fps"] == 10.0
    assert abs(row["gap_max_ms"] - 100.0) < 1
    assert abs(row["age_max_ms"] - 10.0) < 1
    assert row["reconnects"] == 2
    # No frames for the whole next interval: the stall is reported live.
    row2 = s.snapshot(g, object(), now=103.0)
    assert row2["fps"] == 0.0
    assert row2["gap_max_ms"] > 2000
    assert row2["reconnects"] == 0  # getattr default
    summ = s.summary(now=103.0)
    assert summ["fps_min"] == 0.0 and summ["fps_max"] == 10.0 and summ["fps_mean"] == 5.0
    assert summ["gap_max_all_ms"] > 2000


def test_preview_headless_prints_stats_and_writes_csv(tmp_path, capsys):
    out = tmp_path / "s.csv"
    preview.main(["--source", "synthetic", "--no-display", "--seconds", "2.2",
                  "--stats-every", "1", "--stats-csv", str(out)])
    text = capsys.readouterr().out
    lines = [ln for ln in text.splitlines() if ln.startswith("stats ")]
    assert len(lines) >= 2
    assert "summary " in text and "rss_growth_mb=" in text
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(lines)
    assert float(rows[0]["fps"]) > 20  # synthetic runs at 30 fps
    # headless consumer keeps up (allow scheduler jitter on a busy machine)
    assert int(rows[-1]["dropped"]) <= 0.2 * int(rows[-1]["frames_read"])
