from __future__ import annotations

import os
import time
from typing import Optional

import cv2

from ..types import Frame
from .base import FrameSource

# Per-call cap for FFmpeg open and read, in ms. Measured on mediamtx (see
# docs/rtsp.md): with a 3000 ms setting a stalled stream returned from read()
# after 3.0-3.1 s; with no timeout it blocked 10 s (publisher stall) to 30 s
# (server stall, OpenCV's built-in default). 4000 keeps every call under ~5 s.
DEFAULT_TIMEOUT_MS = 4000

# open() keeps retrying (with backoff) for this long before raising.
DEFAULT_OPEN_DEADLINE_S = 10.0

# TCP avoids the packet-loss smearing UDP gives on Wi-Fi; nobuffer/low_delay
# cut FFmpeg's internal buffering; `timeout` (microseconds) is the RTSP socket
# I/O timeout, a second layer under the open/read timeouts passed as open
# params. Read by OpenCV when the capture is opened.
_LOW_LATENCY_OPTS = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    f"|timeout;{DEFAULT_TIMEOUT_MS * 1000}"
)


def _backoff_delay(fail: int, max_backoff: float) -> float:
    """Seconds to wait after the `fail`-th consecutive failure: 0.5, 1, 2, 4,
    8, ... capped at max_backoff."""
    return min(max_backoff, 0.25 * 2 ** min(fail, 16))


def redact_url(url: str) -> str:
    """URL safe to log: userinfo and query string are masked.

    The credentials end at the LAST '@' (a password may itself contain '@' or
    ':' when not percent-encoded). If the path also contains '@' this
    over-redacts, which is the safe direction.
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        scheme, rest = "", url
    if "@" in rest:
        rest = "***@" + rest.rsplit("@", 1)[1]
    # IP-camera apps often put credentials in the query (?user=..&pwd=..).
    base, q, _ = rest.partition("?")
    if q:
        rest = base + "?***"
    return f"{scheme}{sep}{rest}"


class StreamSource(FrameSource):
    """RTSP/HTTP stream: CCTV, or a phone running an IP-camera app.

    Reconnects with capped exponential backoff. read() returns None (never
    raises) while it is down so the grabber thread keeps running. Every FFmpeg
    open/read is bounded by `timeout_ms`.

    open() retries with the same backoff until `open_deadline` seconds have
    passed (a camera or phone app briefly unreachable at startup should not
    kill the program), then raises RuntimeError naming the redacted URL, the
    attempt count and the elapsed time. open_deadline=0 means one attempt.

    Public state:
        reconnects: successful reconnects after the initial open.
        connected:  True while a capture is open.

    FFmpeg options come from OPENCV_FFMPEG_CAPTURE_OPTIONS. If the user has
    already set that variable it is left alone (os.environ.setdefault), so it
    fully replaces ours, including rtsp_transport and the socket `timeout`.
    The open/read timeouts passed as VideoCapture params still apply.
    """

    def __init__(self, url: str, max_backoff: float = 8.0,
                 timeout_ms: int = DEFAULT_TIMEOUT_MS,
                 open_deadline: float = DEFAULT_OPEN_DEADLINE_S):
        self.url = url
        self.open_deadline = float(open_deadline)
        self.max_backoff = max_backoff
        self.timeout_ms = int(timeout_ms)
        self.reconnects = 0
        self._ever_connected = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self._fail = 0
        self._next_try = 0.0

    @property
    def connected(self) -> bool:
        return self._cap is not None

    def _open_params(self) -> list[int]:
        if self.timeout_ms <= 0:
            return []
        return [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms]

    def _connect(self) -> bool:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _LOW_LATENCY_OPTS)
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, self._open_params())
        except Exception:  # cv2.error or a bad URL; never let it escape read()
            return False
        try:
            if not cap.isOpened():
                cap.release()
                return False
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            return False
        self._cap = cap
        if self._ever_connected:
            self.reconnects += 1
        self._ever_connected = True
        return True

    def _drop(self, now: float) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._fail += 1
        self._next_try = now + _backoff_delay(self._fail, self.max_backoff)

    def open(self) -> None:
        t0 = time.monotonic()
        deadline = t0 + max(0.0, self.open_deadline)
        attempts = 0
        fail = 0
        while True:
            attempts += 1
            if self._connect():
                return
            fail += 1
            now = time.monotonic()
            # No new attempt starts after the deadline; the last one may run
            # up to timeout_ms past it.
            wait = _backoff_delay(fail, self.max_backoff)
            if now + wait > deadline:
                break
            time.sleep(wait)
        raise RuntimeError(
            f"cannot open stream {self._redacted()}: {attempts} attempt(s) over "
            f"{time.monotonic() - t0:.1f}s (open_deadline={self.open_deadline:g}s, "
            f"timeout_ms={self.timeout_ms}). Check the camera/app is running and "
            f"reachable, and the URL path and credentials."
        )

    def _redacted(self) -> str:
        # Never log credentials embedded in rtsp://user:pass@host/...
        return redact_url(self.url)

    def read(self) -> Optional[Frame]:
        now = time.monotonic()
        if self._cap is None:
            if now < self._next_try:
                time.sleep(min(0.05, self._next_try - now))
                return None
            if not self._connect():
                self._drop(time.monotonic())
            # Either way return now: one read() call does at most one
            # blocking FFmpeg operation (open OR read), so no call exceeds
            # ~timeout_ms. The next call reads from the new capture.
            return None
        t_read = time.monotonic()
        try:
            ok, img = self._cap.read()
        except Exception:
            ok, img = False, None
        ts = time.monotonic()
        # When FFmpeg's read timeout fires, cap.read() still returns ok=True
        # with an OLD frame drained from the decoder (measured: ~2 such
        # reads, then ~8 instant ones, before ok=False). Treat a read that
        # hit the timeout as a stall: drop the stale frame and reconnect now.
        stalled = self.timeout_ms > 0 and (ts - t_read) >= 0.9 * self.timeout_ms / 1000.0
        if not ok or img is None or stalled:
            self._drop(ts)
            return None
        # Reset backoff only once frames flow, so a server that accepts the
        # connection but never delivers still backs off.
        self._fail = 0
        self._seq += 1
        return Frame(self._seq, ts, img)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def describe(self) -> dict:
        d = {"url": self._redacted(), "reconnects": self.reconnects,
             "timeout_ms": self.timeout_ms}
        if self._cap is not None:
            d["actual"] = (
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )
        return d
