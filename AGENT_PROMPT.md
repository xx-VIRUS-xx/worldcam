# Agent prompt: worldcam, Week 1 (capture layer)

Paste everything below the line into Claude Code, run from the repo root.

---

You are working in `worldcam`, a Python project whose end goal is a local-first perception pipeline:

```
FrameSource -> LatestFrameGrabber -> Detector -> Tracker -> IdentityModel
    -> EventGate -> VisionLLM -> WorldState -> Memory
```

Sources: laptop webcam first, then a phone camera, then CCTV (RTSP). Stage contracts live in `src/worldcam/pipeline.py`. **Only the capture layer is in scope this week.**

The dev machine is a **Mac (macOS, Apple Silicon)**. The only local-camera backend is AVFoundation. Windows/Linux notes below are for later portability, not for this machine.

## Why this exists
A plain `cv2.VideoCapture` + `imshow` loop was laggy on the owner's laptop. The scaffold fixes the structure (threaded latest-frame grabber, `FrameSource` abstraction, MJPG, explicit backend). Your job is to make it reliable on real hardware and prove it with numbers.

Read `docs/ROADMAP.md` first: it explains the patterns being adopted, the conventions (normalized coordinates, no frame queues, local-only) and the later weeks, so week 1 decisions don't block them.

## Step 0: setup
- The repo is under git; commit in small, reviewable steps.
- Use the project venv: `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`. Run tools as `.venv/bin/worldcam-bench`, `.venv/bin/pytest`.
- External binaries allowed for testing only (not as Python deps): `ffmpeg` (installed via Homebrew), `mediamtx` (`brew install mediamtx` if needed).

## Rules
1. **Plan first.** Before editing, read the repo and post a short plan. Then wait for approval before coding.
2. Downstream code never touches `cv2.VideoCapture`; only `FrameSource` implementations do. Adding CCTV must mean adding or configuring one source class, nothing else.
3. Never keep a queue of frames. Consumers get the newest frame; stale frames are dropped. No unbounded buffers anywhere.
4. Capture at native resolution; detection and display run on a downscaled copy. Crops for faces or VLM come from the full-res frame.
5. Every performance claim must come from the repo's own CLI tools or scripts (`worldcam-bench`, `worldcam-preview`, or a measurement script you add under `src/worldcam/cli/`), with the output pasted into your report. No "should be faster".
6. Do not add detectors, trackers, face recognition, VLMs, databases or web UIs. Interfaces in `pipeline.py` and helpers in `geometry.py` stay as they are unless a change is required, in which case explain why. All boxes/polygons stay normalized (0..1). Changes inside `sources/`, `capture.py` and `cli/` are expected.
7. Keep everything local. No network calls, no telemetry, no uploading frames. Never log RTSP credentials (see `StreamSource._redacted`).
8. Keep dependencies to `opencv-python` and `numpy` unless a step below proves they are insufficient. Use stdlib (`resource`, `tracemalloc`) for memory measurement, not `psutil`.
9. Steps that need a human (phone setup, changing lighting, plugging devices) are not guessed: stop and ask, or list them under "could not verify".

## Tasks
1. **Bench tooling.** Add `--backend` (e.g. `avfoundation|dshow|msmf|v4l2|any`) and `--fourcc` (e.g. `MJPG|YUYV|none`) flags to `worldcam-bench`, plumbed through `make_source`, so task 3 can vary one thing at a time.
2. **Baseline on real hardware.** Run `worldcam-bench --source 0 --seconds 30 --work-ms 80` at 640x480, 1280x720 and 1920x1080 (30 fps requested). Record, for each: negotiated resolution/fps/fourcc, phase 1 fps, frame-gap p95, phase 2 age p95, dropped. Tabulate in `docs/baseline.md`, along with OS, laptop and webcam model. Note that AVFoundation largely ignores requested FOURCC and buffer size; record what was actually negotiated.
3. **Diagnose.** If phase 1 is below ~25 fps at 720p, work through, in order, changing one variable at a time and recording each result:
   - macOS: lighting / auto-exposure first (AVFoundation drops to ~15 fps in low light), then resolution, then `--backend any` vs `avfoundation`.
   - Windows/Linux (portability notes only): backend (DSHOW/MSMF/V4L2), MJPG vs YUYV, USB port.
   If OpenCV capture is still unstable, implement `FfmpegPipeSource(FrameSource)` (ffmpeg subprocess, raw BGR over stdout, bounded read; input `-f avfoundation` on macOS, `-f dshow`/`-f v4l2 -input_format mjpeg` elsewhere) and benchmark it against `WebcamSource`.
4. **Phone source.** Support a phone as (a) an IP-camera-app stream via `StreamSource` and (b) a camera index via macOS Continuity Camera (iPhone), DroidCam or Camo. Benchmark what is physically available and document setup steps for all of them in `docs/phone.md`.
5. **RTSP robustness.** Test `StreamSource` against a local test stream (mediamtx + ffmpeg publishing a looping file), including killing and restarting the publisher mid-run. Requirements:
   - It reconnects without crashing the grabber thread, and `read()` keeps returning `None` (not raise) while down.
   - No read may block for more than ~5 s when the publisher dies: set FFmpeg open/read timeouts (`CAP_PROP_OPEN_TIMEOUT_MSEC` / `CAP_PROP_READ_TIMEOUT_MSEC` and/or a `timeout`/`stimeout` capture option).
   - Expose a reconnect counter on the source. Report time-to-first-frame after the publisher restarts.
6. **Dual-stream check.** `DualGrabber` (substream for detection pacing, mainstream for crops) exists and is unit-tested with synthetic sources. To measure it, `crop()` (or a sibling method) must expose which main frame was used (e.g. return the `Frame` or its `ts`). Verify on real hardware: on macOS a second open of the same built-in camera often fails, so prefer webcam + phone, or the same RTSP stream opened twice (one downscaled). Report the crop-vs-sub-frame time offset (`Frame.ts` of the sub frame used vs the main frame `peek()`ed at crop time) at p50/p95. If the offset exceeds ~100 ms p95, say so and propose a fix; do not paper over it.
7. **Tests.** Extend `tests/` for anything you change. Tests must run without a camera or network (use `SyntheticSource`/`FileSource`; mock or skip RTSP).
8. **Soak test.** Add to `worldcam-preview --no-display` periodic stats: rolling fps, max frame gap, RSS (via `resource.getrusage`; note macOS reports bytes, Linux KiB), and source reconnects. Run 30 minutes on the real source in the background so it doesn't block the session, and report FPS stability, max frame gap, RSS start/end, and reconnects.

## Acceptance criteria (all must hold)
- 720p webcam: sustained >= 25 fps raw read, frame-gap p95 < 60 ms (in adequate lighting; report low-light numbers separately).
- With `--work-ms 80`, frame age at use p95 < 100 ms (i.e. no backlog), and dropped > 0 (i.e. frames are being skipped, not queued).
- RTSP: publisher kill/restart causes no crash, no read blocks > 5 s, and frames resume after the publisher is back.
- 30-minute soak: no crash, RSS growth < 50 MB.
- Swapping `--source 0` for `--source rtsp://...` requires zero code changes outside `sources/`.
- `pytest` passes without any camera attached.

## Report format
Finish with a single message containing: the baseline table, what the root cause of the original lag was (one sentence, with the evidence), what you changed, anything you could not verify and why, and a recommended resolution/FPS/backend default for this machine. Be direct about failures and limitations. Do not pad.
