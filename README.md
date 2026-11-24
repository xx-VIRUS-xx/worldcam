# worldcam

Camera -> detector -> tracker -> identity -> VLM -> world state -> memory.
This repo currently contains only the **capture layer** and stage contracts.

## Layout
```
src/worldcam/
  types.py        Frame(seq, ts, image)
  capture.py      LatestFrameGrabber: capture thread, newest-frame-only
  sources/        FrameSource implementations: webcam, rtsp/http stream, file, synthetic
  sources/dual.py DualGrabber: substream paces detection, mainstream gives full-res crops
  geometry.py     normalized boxes/polygons, point-in-polygon, IoU, resolution-independent crop
  pipeline.py     Typed contracts for later stages (track lifecycle, zones, world state, memory); no implementations yet
docs/ROADMAP.md  adopted NVR patterns, conventions, week-by-week plan, risks
  cli/bench.py    two-phase benchmark (raw read vs threaded grabber)
  cli/preview.py  live preview with fps / frame-age overlay
tests/            run without any camera
AGENT_PROMPT.md   prompt for a coding agent to finish week 1 on real hardware
```

## Run
```bash
pip install -e ".[dev]"
pytest

worldcam-bench   --source 0 --width 1280 --height 720 --fps 30
worldcam-preview --source 0
worldcam-preview --source http://<phone-ip>:8080/video      # IP-camera app
worldcam-preview --source rtsp://user:pass@<cctv-ip>/stream1 # CCTV later
worldcam-preview --source synthetic --no-display --seconds 3 # no hardware
```

## Reading the benchmark
- **phase 1 slow** -> driver/backend/pixel format/low light. Try another backend, keep MJPG, brighten the room.
- **phase 1 fine, phase 2 age high** -> your loop is building a backlog (should not happen with the grabber).
- **`age`** is measured from when `read()` returned, so it excludes sensor and USB latency. For true glass-to-screen latency, film a running clock next to the preview.

## Design rules
1. Everything downstream consumes `FrameSource`; CCTV is a new URL, not new code.
2. Newest frame wins; no frame queues.
3. Capture full-res, process downscaled, crop from full-res.
4. Local-only. Never log stream credentials.
