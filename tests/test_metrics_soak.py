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


def test_rss_growth_is_windowed_and_skips_first_interval(monkeypatch):
    # init (pre-open) 40, camera-open spike 160, then steady ~90 with noise, end ~92.
    samples = iter([40.0, 160.0, 88, 92, 90, 89, 91, 90, 95, 85, 93, 91, 92, 90])
    monkeypatch.setattr(metrics, "current_rss_mb", lambda: next(samples))
    s = SoakStats(now=0.0, rss_window=3)
    g = _G()
    for i in range(1, 14):
        s.snapshot(g, _Src(), now=float(i))
    summ = s.summary(now=13.0)
    assert summ["rss_init_mb"] == 40.0          # not used for growth
    assert summ["rss_start_mb"] == 90.0         # mean(88, 92, 90): spike skipped
    assert summ["rss_end_mb"] == 91.0           # mean(93, 91, ... last 3) = (91+92+90)/3
    assert summ["rss_growth_mb"] == 1.0
    assert summ["rss_max_mb"] == 160.0


def test_short_trailing_interval_not_counted_in_fps(monkeypatch):
    monkeypatch.setattr(metrics, "current_rss_mb", lambda: 50.0)
    s = SoakStats(now=0.0, min_interval=5.0)
    g = _G()
    for i in range(300):
        s.on_frame(i / 30, now=i / 30)
    s.snapshot(g, _Src(), now=10.0)       # full interval, 30 fps
    s.snapshot(g, _Src(), now=10.5)       # 0.5 s trailing, 0 frames
    summ = s.summary(now=10.5)
    assert summ["intervals"] == 1
    assert summ["fps_min"] == 30.0
