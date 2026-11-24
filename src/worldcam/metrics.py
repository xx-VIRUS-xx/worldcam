from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque

try:  # not available on Windows
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None  # type: ignore[assignment]


class RateMeter:
    """Rolling FPS over the last `window` events."""

    def __init__(self, window: int = 60):
        self._t: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._t.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


# ---- memory --------------------------------------------------------------
#
# Why not resource.getrusage alone: ru_maxrss is the PEAK resident set size
# (macOS: bytes, Linux: KiB). A peak only ever goes up, so it cannot show the
# current RSS or whether memory is being returned; a soak test needs both.
# We still report it (peak_rss_mb) as a cross-check.
#
# Current RSS without psutil (deps are limited to opencv-python + numpy):
#   * Linux: /proc/self/statm, resident pages * page size. A plain file read.
#   * macOS (and other BSDs): `ps -o rss= -p <pid>` (KiB). There is no /proc,
#     and the alternative, mach task_info via ctypes, relies on hand-copied
#     struct layouts that are easy to get wrong. ps ships with the base OS,
#     is purely local (no network), costs a few ms, and is called once per
#     stats interval, so its overhead is irrelevant. Any failure -> None.

_MB = 1024 * 1024


def peak_rss_mb() -> float | None:
    """Peak RSS of this process in MiB (ru_maxrss), or None if unavailable."""
    if _resource is None:
        return None
    peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / _MB  # bytes on macOS
    return peak / 1024  # KiB on Linux


def current_rss_mb() -> float | None:
    """Current RSS of this process in MiB, or None if it cannot be measured."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as fh:
                resident_pages = int(fh.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / _MB
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip()
        return int(out) / 1024  # ps reports KiB
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# ---- soak / periodic stats ----------------------------------------------

STATS_FIELDS = [
    "elapsed_s", "fps", "read_fps", "frames_read", "frames_consumed", "dropped",
    "read_failures", "gap_max_ms", "gap_max_all_ms", "age_max_ms",
    "rss_mb", "peak_rss_mb", "reconnects",
]


def _round(v: float | None, nd: int = 1) -> float | None:
    return None if v is None else round(v, nd)


class SoakStats:
    """Interval statistics for long runs. O(1) memory: only running
    aggregates are kept, never per-frame or per-interval history.

    fps        = new frames the consumer received in the interval / interval.
    read_fps   = frames the grabber read from the source in the interval.
    frame gap  = Frame.ts difference between consecutive NEW frames the
                 consumer received. A stall still in progress is included at
                 snapshot time as (now - last frame ts), so a dead source shows
                 up in the interval where it happens, not only after recovery.
    age        = consumer receive time - Frame.ts.
    """

    def __init__(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        self.t0 = now
        self.interval_start = now
        self._last_ts: float | None = None
        self._last_read = 0
        self._int_frames = 0
        self._int_gap = 0.0
        self._int_age = 0.0
        self.gap_max_all = 0.0
        self.age_max_all = 0.0
        self.fps_min = float("inf")
        self.fps_max = 0.0
        self._fps_sum = 0.0
        self._fps_n = 0
        self.rss_start = current_rss_mb()
        self.rss_end = self.rss_start
        self.rss_max = self.rss_start
        self.reconnects = 0

    def on_frame(self, frame_ts: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._last_ts is not None:
            gap = frame_ts - self._last_ts
            self._int_gap = max(self._int_gap, gap)
            self.gap_max_all = max(self.gap_max_all, gap)
        self._last_ts = frame_ts
        age = now - frame_ts
        self._int_age = max(self._int_age, age)
        self.age_max_all = max(self.age_max_all, age)
        self._int_frames += 1

    def snapshot(self, grabber, source, now: float | None = None) -> dict:
        """Close the current interval and return one row (keys = STATS_FIELDS)."""
        now = time.monotonic() if now is None else now
        dt = max(now - self.interval_start, 1e-9)
        if self._last_ts is not None:  # stall in progress
            open_gap = now - self._last_ts
            self._int_gap = max(self._int_gap, open_gap)
            self.gap_max_all = max(self.gap_max_all, open_gap)
        fps = self._int_frames / dt
        read_fps = (grabber.frames_read - self._last_read) / dt
        if dt >= 0.5:  # a tiny trailing interval would skew fps min/mean/max
            self.fps_min = min(self.fps_min, fps)
            self.fps_max = max(self.fps_max, fps)
            self._fps_sum += fps
            self._fps_n += 1
        rss = current_rss_mb()
        if rss is not None:
            self.rss_end = rss
            self.rss_max = rss if self.rss_max is None else max(self.rss_max, rss)
        self.reconnects = int(getattr(source, "reconnects", 0) or 0)
        row = {
            "elapsed_s": round(now - self.t0, 1),
            "fps": round(fps, 2),
            "read_fps": round(read_fps, 2),
            "frames_read": grabber.frames_read,
            "frames_consumed": grabber.frames_consumed,
            "dropped": grabber.dropped,
            "read_failures": grabber.read_failures,
            "gap_max_ms": round(self._int_gap * 1000, 1),
            "gap_max_all_ms": round(self.gap_max_all * 1000, 1),
            "age_max_ms": round(self._int_age * 1000, 1),
            "rss_mb": _round(rss),
            "peak_rss_mb": _round(peak_rss_mb()),
            "reconnects": self.reconnects,
        }
        self.interval_start = now
        self._last_read = grabber.frames_read
        self._int_frames = 0
        self._int_gap = 0.0
        self._int_age = 0.0
        return row

    def summary(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        growth = None
        if self.rss_start is not None and self.rss_end is not None:
            growth = self.rss_end - self.rss_start
        n = self._fps_n
        return {
            "elapsed_s": round(now - self.t0, 1),
            "rss_start_mb": _round(self.rss_start),
            "rss_end_mb": _round(self.rss_end),
            "rss_max_mb": _round(self.rss_max),
            "peak_rss_mb": _round(peak_rss_mb()),
            "rss_growth_mb": _round(growth),
            "gap_max_all_ms": round(self.gap_max_all * 1000, 1),
            "age_max_all_ms": round(self.age_max_all * 1000, 1),
            "fps_min": round(self.fps_min, 2) if n else None,
            "fps_mean": round(self._fps_sum / n, 2) if n else None,
            "fps_max": round(self.fps_max, 2) if n else None,
            "intervals": n,
            "reconnects": self.reconnects,
        }


def format_row(row: dict) -> str:
    return " ".join(f"{k}={'-' if v is None else v}" for k, v in row.items())
