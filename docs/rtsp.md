# RTSP robustness (`StreamSource`)

Measured on macOS, Apple M2, OpenCV 5.1.0 (FFmpeg backend), mediamtx v1.21.4,
local loopback. The test clip is ffmpeg `testsrc2` at 1280x720 and 30 fps, 20 s long, H.264 with GOP 30 and a
timestamp overlay. It is published in a loop by `ffmpeg -re -stream_loop -1 ... -c copy -f rtsp`.

## Behaviour

| Item | Behaviour |
|---|---|
| Per-call bound | Every FFmpeg open and read is capped at `timeout_ms` (default 4000). Each `read()` call does at most one blocking FFmpeg operation (an open *or* a read), so no `read()` call runs longer than about 4.1 s. |
| Timeout mechanism | Two layers. (1) `CAP_PROP_OPEN_TIMEOUT_MSEC` and `CAP_PROP_READ_TIMEOUT_MSEC` are passed as open params: `cv2.VideoCapture(url, cv2.CAP_FFMPEG, [..])`. (2) `timeout;4000000` (microseconds, the RTSP socket I/O timeout) is added to `OPENCV_FFMPEG_CAPTURE_OPTIONS`. |
| User-set env var | `OPENCV_FFMPEG_CAPTURE_OPTIONS` is applied with `os.environ.setdefault`, so if you set it yourself it **replaces all of ours**: `rtsp_transport;tcp`, `nobuffer`, `low_delay` and `timeout`. The open/read params still apply, so reads stay bounded. If you set the variable, include `rtsp_transport;tcp` yourself. |
| Stall detection | When the FFmpeg read timeout fires, `cap.read()` still returns `ok=True` with an **old frame drained from the decoder**: 2 slow reads, then about 8 instant ones, and only then `ok=False` (measured below). `StreamSource` therefore treats any read that took at least 90% of `timeout_ms` as a stall: it drops that frame and reconnects. A camera slower than about 0.27 fps would be treated as stalled; raise `timeout_ms` for such cameras. |
| While down | `read()` returns `None` and never raises. `cv2` exceptions from `VideoCapture()` or `cap.read()` are caught. |
| Backoff | After the n-th consecutive failure the next attempt waits `min(max_backoff, 0.25 * 2**n)`: 0.5, 1, 2, 4, 8, 8, ... s with the default `max_backoff=8`. The counter resets only after a frame is decoded, so a server that accepts the connection but sends nothing still backs off. |
| `reconnects` | Public int: the number of successful re-opens after the initial `open()`. A re-open that succeeds and then stalls again still counts. |
| `connected` | Property: True while a capture is open. |
| `open()` at startup | Retries with the same backoff for up to `open_deadline` s (default 10), then raises `RuntimeError`. The error gives the redacted URL, the attempt count, the elapsed time and what to check. No attempt starts after the deadline, but the last one can run up to `timeout_ms` past it. `open_deadline=0` means a single attempt. |
| Redaction | `redact_url()` masks everything up to the **last** `@`, so passwords containing `@`, `:` or `/` don't leak. The old code split on the first `@` and leaked `ss@host` for `admin:p@ss@host`. It also masks the query string (`?user=..&pwd=..`), as used by IP-camera apps. If the path contains `@`, the host is masked as well (over-redaction is the safe failure). |

### Why `open()` retries rather than failing fast

A phone IP-camera app or a CCTV camera that is briefly unreachable at startup
used to make `LatestFrameGrabber.start()` raise, so the program did not start.
After `open()` succeeds, `read()` already rides out outages of any length. A
bounded startup retry gives the same behaviour for a short glitch at startup.
A wrong URL or a dead camera still fails within about `open_deadline +
timeout_ms` (at most about 14 s with the defaults), and the error says why. It
does not hang forever. Use `open_deadline=0` to fail fast.

## Which timeout setting actually works

The first comparison came from a throwaway probe: raw `cv2.VideoCapture` with
3000 ms settings, `SIGSTOP` on the publisher or the server. The table records
what that probe showed. The numbers that count are the repo-tool runs below.

| Setting | Publisher stalled: max single `cap.read()` | Server stalled: max `cap.read()` / open |
|---|---|---|
| none | 10.05 s (mediamtx's own 10 s publisher timeout ended it) | 30.12 s / 30.08 s (OpenCV default) |
| open params, 3000 ms | 3.10 s | 3.03 s / 3.04 s |
| env `timeout;3000000` | 3.71 s | 3.15 s / 3.13 s |

Both layers work on this build. The params are the main control: they are
precise and they survive a user-set env var. `cap.get(CAP_PROP_*_TIMEOUT_MSEC)`
returns -1 on this build even when the params took effect, so don't use it to
check the setting.

A publisher kill (`SIGKILL`) needs no timeout. mediamtx tears down the reader
session and `cap.read()` returns `ok=False` within 0.01 to 0.03 s, after about
11 buffered frames.

## Measurements (repo tool)

```
PYTHONPATH=src python -m worldcam.cli.rtsp_soak --clip clip.mp4 --port 8554 [--down S] [--events ...] [--no-timeouts] [--late-publisher S]
```

The tool starts its own mediamtx (temp config, `127.0.0.1:<port>`, every
other listener off) and an ffmpeg publisher. It runs `StreamSource` inside
`LatestFrameGrabber` and applies these events:

- `pubkill`: SIGKILL the publisher, then restart it after `--down` s.
- `pubstall`: SIGSTOP the publisher; after `--down` s, kill and restart it.
- `srvstall`: SIGSTOP mediamtx, then SIGCONT after `--down` s.
- `srvkill`: SIGKILL mediamtx and the publisher, then restart both after `--srv-down` s.

Output fields:

- `max_read`: the longest single `read()` that overlapped the event window, including one still blocked.
- `ttff`: time from recovery (restart or SIGCONT) to the first new frame.
- `frames_while_down`: frames delivered between 0.5 s after the event and recovery. These are stale frames.
- `connect attempts`: when each reconnect attempt started, in seconds after the event; `x` marks a failed attempt.

### Default settings (timeout_ms=4000), 10 s publisher outage, 20 s server outage

```
opencv=5.1.0 url=rtsp://127.0.0.1:8554/soak timeout_ms=4000 capture_opts=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|timeout;4000000
[warmup] 29.9 fps max_read=1.19s
[pubkill] max_read=0.83s thread_alive=True ttff=6.96s reconnects=1 connected=True frames_while_down=0
[pubkill]   connect attempts, s after event (x=failed), recovery at 10.0: 0.5x 1.6x 3.6x 7.6x 15.6
[pubstall] max_read=4.09s thread_alive=True ttff=2.87s reconnects=3 connected=True frames_while_down=0
[pubstall]   connect attempts, s after event (x=failed), recovery at 10.0: 4.6 11.0
[srvstall] max_read=4.10s thread_alive=True ttff=0.05s reconnects=4 connected=True frames_while_down=0
[srvstall]   connect attempts, s after event (x=failed), recovery at 10.0: 4.6x 9.7
[srvkill] max_read=1.26s thread_alive=True ttff=5.66s reconnects=5 connected=True frames_while_down=0
[srvkill]   connect attempts, s after event (x=failed), recovery at 20.0: 0.5x 1.5x 3.6x 7.6x 15.6x 23.6
[final] reconnects=5 frames=886 read_failures=683 max_read_overall=4.10s thread_alive=True
```

### Stalls held for 15 s, with timeouts vs `--no-timeouts`

The 15 s hold is longer than mediamtx's 10 s publisher timeout.

```
opencv=5.1.0 url=rtsp://127.0.0.1:8554/soak timeout_ms=4000 capture_opts=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|timeout;4000000
[warmup] 30.0 fps max_read=1.03s
[pubstall] max_read=4.09s thread_alive=True ttff=3.88s reconnects=2 connected=True frames_while_down=0
[pubstall]   connect attempts, s after event (x=failed), recovery at 15.1: 4.6 11.0x 13.0x 17.0
[srvstall] max_read=4.05s thread_alive=True ttff=1.98s reconnects=3 connected=True frames_while_down=0
[srvstall]   connect attempts, s after event (x=failed), recovery at 15.0: 4.5x 9.6x 15.6
[final] reconnects=3 frames=453 read_failures=196 max_read_overall=4.09s thread_alive=True
```

```
opencv=5.1.0 url=rtsp://127.0.0.1:8554/soak timeout_ms=0 capture_opts=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay
[warmup] 30.0 fps max_read=1.03s
[pubstall] max_read=10.01s thread_alive=True ttff=3.90s reconnects=1 connected=True frames_while_down=10
[pubstall]   connect attempts, s after event (x=failed), recovery at 15.0: 10.5x 11.5x 13.6x 17.6
[srvstall] max_read=15.03s thread_alive=True ttff=0.01s reconnects=1 connected=True frames_while_down=0
[srvstall]   connect attempts, s after event (x=failed), recovery at 15.0: -
[final] reconnects=1 frames=619 read_failures=132 max_read_overall=15.03s thread_alive=True
```

Without timeouts, a stalled server blocks `read()` for the whole stall: 15 s
here, and up to OpenCV's 30 s default. A stalled publisher blocks for 10 s and
then delivers 10 stale frames.

### Camera unreachable at startup (publisher starts 4 s after `open()`)

```
opencv=5.1.0 url=rtsp://127.0.0.1:8554/soak timeout_ms=4000 capture_opts=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|timeout;4000000
[startup] publisher started 4s after open(); open() returned after 8.34s, attempts=5
[warmup] 30.0 fps max_read=0.54s
[final] reconnects=0 frames=151 read_failures=0 max_read_overall=0.54s thread_alive=True
```

## Reading the numbers

- **Longest single `read()`**: 4.10 s with the defaults, against 10 to 15 s (up to 30 s) without timeouts. This meets the "~5 s" requirement.
- **Grabber thread**: alive after every event.
- **Time to first frame after the publisher restarts**: 6.96 s. Backoff dominates: attempts ran at 7.6 s and 15.6 s after the kill, and the publisher came back at 10.0 s. After a stall the time is 2.9 to 3.9 s. After a server restart it is 5.66 s (attempt at 23.6 s, server back at 20.0 s). The worst case is about `max_backoff` plus about 1 s for the first keyframe.
- **Backoff**: whole-server outage attempts ran at 0.5, 1.5, 3.6, 7.6, 15.6, 23.6 s, which is doubling capped at 8 s, as designed.
- **Reconnects**: 1 for each kill. A stalled publisher can add an extra reconnect (`pubstall` went from 1 to 3): mediamtx still advertises the path, so the re-open at 4.6 s succeeds and then stalls again.
- **Stale frames**: none delivered with the stall check (`frames_while_down=0`), against 10 without it.

## Limitations

- Everything was measured on loopback against mediamtx. Real cameras and Wi-Fi phones were not tested, and their open time and keyframe interval add to `ttff`.
- `frames_while_down` only counts frames that arrived 0.5 s or more after the event, so it misses frames drained in the first half second.
- Faster recovery means lowering `max_backoff`, at the cost of more connection attempts against a dead camera.
- The `timeout` capture option is in microseconds for FFmpeg 5 and later. Older FFmpeg builds used `stimeout` for RTSP, and only this OpenCV 5.1.0 build was tested.
