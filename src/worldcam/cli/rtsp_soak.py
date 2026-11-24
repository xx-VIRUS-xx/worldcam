"""RTSP outage experiment: StreamSource inside LatestFrameGrabber against a
local mediamtx + ffmpeg publisher, with scripted failures.

    python -m worldcam.cli.rtsp_soak --clip clip.mp4 [--port 8554]

Starts its own mediamtx (temp config, RTSP on 127.0.0.1:<port> only, every
other listener off) and an ffmpeg publisher looping the clip, then runs:

  pubkill   SIGKILL the publisher, wait --down s, restart it
  pubstall  SIGSTOP the publisher (TCP stays open, no data), --down s, kill+restart
  srvstall  SIGSTOP mediamtx (connections hang), --down s, SIGCONT
  srvkill   SIGKILL mediamtx, wait --srv-down s, restart server + publisher

For each event it prints the longest single read() call, the grabber thread
state, time from recovery to the first new frame, reconnects, and the times of
reconnect attempts while down (to show backoff). Every process it starts is
killed on exit. Local-only: binds and connects to 127.0.0.1.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from ..capture import LatestFrameGrabber  # noqa: E402
from ..sources import rtsp as rtsp_mod  # noqa: E402
from ..sources.base import FrameSource  # noqa: E402
from ..sources.rtsp import StreamSource  # noqa: E402

_MTX_CFG = """logLevel: error
rtsp: yes
rtspAddress: 127.0.0.1:{port}
rtspTransports: [tcp]
rtpAddress: 127.0.0.1:{rtp}
rtcpAddress: 127.0.0.1:{rtcp}
rtmp: no
hls: no
webrtc: no
srt: no
api: no
metrics: no
pprof: no
playback: no
paths:
  all_others:
"""


class TimedSource(FrameSource):
    """Wraps a StreamSource and records each read() duration, frame times and
    connect attempts. Lock-protected; read by the main thread."""

    def __init__(self, inner: StreamSource):
        self.inner = inner
        self.lock = threading.Lock()
        self.durations: list[tuple[float, float]] = []  # (start, seconds)
        self.frame_ts: list[float] = []
        self.attempts: list[tuple[float, bool]] = []
        orig = inner._connect

        def connect() -> bool:
            t = time.monotonic()
            ok = orig()
            with self.lock:
                self.attempts.append((t, ok))
            return ok

        inner._connect = connect  # type: ignore[method-assign]

    def open(self) -> None:
        self.inner.open()

    def close(self) -> None:
        self.inner.close()

    def read(self):
        t = time.monotonic()
        with self.lock:
            self.inflight = t
        f = self.inner.read()
        d = time.monotonic() - t
        with self.lock:
            self.inflight = None
            self.durations.append((t, d))
            if f is not None:
                self.frame_ts.append(f.ts)
        return f

    inflight: float | None = None

    def max_read(self, t0: float, t1: float) -> float:
        """Longest read() that overlapped [t0, t1], counting one still blocked."""
        with self.lock:
            done = (d for s, d in self.durations if s < t1 and s + d >= t0)
            m = max(done, default=0.0)
            if self.inflight is not None:
                m = max(m, time.monotonic() - self.inflight)
            return m

    def first_frame_after(self, t: float):
        with self.lock:
            return next((x for x in self.frame_ts if x > t), None)

    def attempts_between(self, t0: float, t1: float):
        with self.lock:
            return [(a, ok) for a, ok in self.attempts if t0 <= a < t1]


class Lab:
    def __init__(self, clip: str, port: int):
        self.clip = clip
        self.port = port
        self.url = f"rtsp://127.0.0.1:{port}/soak"
        self.dir = tempfile.mkdtemp(prefix="worldcam-rtsp-")
        self.cfg = os.path.join(self.dir, "mediamtx.yml")
        with open(self.cfg, "w") as fh:
            fh.write(_MTX_CFG.format(port=port, rtp=port + 10000, rtcp=port + 10001))
        self.srv: subprocess.Popen | None = None
        self.pub: subprocess.Popen | None = None

    def start_server(self) -> None:
        # cwd: mediamtx writes self-signed auto.crt/auto.key into its cwd.
        self.srv = subprocess.Popen([shutil.which("mediamtx") or "mediamtx", self.cfg],
                                    cwd=self.dir, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        time.sleep(0.7)

    def start_publisher(self) -> None:
        self.pub = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "quiet", "-re", "-stream_loop", "-1",
             "-i", self.clip, "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", self.url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _kill(p: subprocess.Popen | None) -> None:
        if p is None or p.poll() is not None:
            return
        try:
            p.send_signal(signal.SIGCONT)
            p.kill()
            p.wait(timeout=5)
        except Exception:
            pass

    def kill_publisher(self) -> None:
        self._kill(self.pub)

    def kill_server(self) -> None:
        self._kill(self.srv)

    def shutdown(self) -> None:
        self.kill_publisher()
        self.kill_server()
        shutil.rmtree(self.dir, ignore_errors=True)


def wait_frame_after(ts: TimedSource, t: float, limit: float) -> float | None:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        f = ts.first_frame_after(t)
        if f is not None:
            return f - t
        time.sleep(0.01)
    return None


def report(name: str, ts: TimedSource, g: LatestFrameGrabber, t_event: float,
           t_recover: float, ttff: float | None) -> None:
    t_end = time.monotonic()
    att = ts.attempts_between(t_event, t_recover + (ttff or 0.0) + 0.001)
    with ts.lock:
        during = sum(1 for x in ts.frame_ts if t_event + 0.5 < x <= t_recover)
    rel = " ".join(f"{a - t_event:.1f}{'' if ok else 'x'}" for a, ok in att)
    print(f"[{name}] max_read={ts.max_read(t_event, t_end):.2f}s "
          f"thread_alive={g._thread is not None and g._thread.is_alive()} "
          f"ttff={'none' if ttff is None else f'{ttff:.2f}s'} "
          f"reconnects={ts.inner.reconnects} connected={ts.inner.connected} "
          f"frames_while_down={during}", flush=True)
    print(f"[{name}]   connect attempts, s after event (x=failed), recovery at "
          f"{t_recover - t_event:.1f}: {rel or '-'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--down", type=float, default=10.0, help="publisher outage, s")
    ap.add_argument("--srv-down", type=float, default=20.0, help="server outage, s")
    ap.add_argument("--steady", type=float, default=5.0, help="run time between events, s")
    ap.add_argument("--timeout-ms", type=int, default=rtsp_mod.DEFAULT_TIMEOUT_MS)
    ap.add_argument("--no-timeouts", action="store_true",
                    help="baseline: no open/read params and no `timeout` capture option")
    ap.add_argument("--events", default="pubkill,pubstall,srvstall,srvkill",
                    help="comma list; empty string for none")
    ap.add_argument("--late-publisher", type=float, default=0.0,
                    help="start the publisher this many s AFTER the source opens")
    a = ap.parse_args(argv)

    if a.no_timeouts:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        a.timeout_ms = 0
    import cv2
    lab = Lab(os.path.abspath(a.clip), a.port)
    print(f"opencv={cv2.__version__} url={lab.url} timeout_ms={a.timeout_ms} "
          f"capture_opts={os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', rtsp_mod._LOW_LATENCY_OPTS)}",
          flush=True)
    g = None
    timer = None
    try:
        lab.start_server()
        if a.late_publisher > 0:
            # Camera unreachable at startup: open() must retry, not fail.
            timer = threading.Timer(a.late_publisher, lab.start_publisher)
            timer.daemon = True
            timer.start()
        else:
            lab.start_publisher()
            time.sleep(2.0)
        src = StreamSource(lab.url, timeout_ms=a.timeout_ms)
        ts = TimedSource(src)
        t_open = time.monotonic()
        g = LatestFrameGrabber(ts).start()
        if a.late_publisher > 0:
            print(f"[startup] publisher started {a.late_publisher:g}s after open(); open() "
                  f"returned after {time.monotonic() - t_open:.2f}s, attempts="
                  f"{len(ts.attempts)}", flush=True)
        t0 = time.monotonic()
        if wait_frame_after(ts, 0.0, 10) is None:
            print("no frames at start; aborting", flush=True)
            return 1
        t1 = time.monotonic()
        time.sleep(a.steady)
        t2 = time.monotonic()
        with ts.lock:
            n = sum(1 for x in ts.frame_ts if t1 < x <= t2)
        print(f"[warmup] {n / (t2 - t1):.1f} fps "
              f"max_read={ts.max_read(t0, time.monotonic()):.2f}s", flush=True)

        for ev in filter(None, a.events.split(",")):
            t_ev = time.monotonic()
            if ev == "pubkill":
                lab.kill_publisher()
                time.sleep(a.down)
                t_rec = time.monotonic()
                lab.start_publisher()
            elif ev == "pubstall":
                lab.pub.send_signal(signal.SIGSTOP)
                time.sleep(a.down)
                lab.kill_publisher()
                t_rec = time.monotonic()
                lab.start_publisher()
            elif ev == "srvstall":
                lab.srv.send_signal(signal.SIGSTOP)
                time.sleep(a.down)
                t_rec = time.monotonic()
                lab.srv.send_signal(signal.SIGCONT)
            elif ev == "srvkill":
                lab.kill_publisher()
                lab.kill_server()
                time.sleep(a.srv_down)
                t_rec = time.monotonic()
                lab.start_server()
                lab.start_publisher()
            else:
                raise SystemExit(f"unknown event {ev}")
            ttff = wait_frame_after(ts, t_rec, 30)
            if ttff is None and ev == "srvstall":
                # The publisher may have been dropped by the frozen server; restart it.
                lab.kill_publisher()
                lab.start_publisher()
                ttff_r = wait_frame_after(ts, t_rec, 30)
                print(f"[{ev}]   publisher restarted; ttff incl. restart="
                      f"{'none' if ttff_r is None else f'{ttff_r:.2f}s'}", flush=True)
            report(ev, ts, g, t_ev, t_rec, ttff)
            time.sleep(a.steady)

        print(f"[final] reconnects={src.reconnects} frames={g.frames_read} "
              f"read_failures={g.read_failures} max_read_overall="
              f"{max((d for _, d in ts.durations), default=0):.2f}s "
              f"thread_alive={g._thread.is_alive()}", flush=True)
        return 0
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        if g is not None:
            g.stop()
        lab.shutdown()


if __name__ == "__main__":
    sys.exit(main())
