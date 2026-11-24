# worldcam

Camera -> detector -> tracker -> identity -> VLM -> world state -> memory.
This repo currently contains only the **capture layer** and stage contracts.

**How to run it and where everything lives: [docs/RUNNING.md](docs/RUNNING.md).**

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/worldcam-preview --source 0                    # built-in camera, q to quit
.venv/bin/worldcam-bench   --source 0 --seconds 30 --work-ms 80
```

Docs: [RUNNING](docs/RUNNING.md) · [ROADMAP](docs/ROADMAP.md) ·
[baseline](docs/baseline.md) · [phone](docs/phone.md) · [rtsp](docs/rtsp.md) · [dual](docs/dual.md)

## Reading the benchmark
- **phase 1 slow** -> driver/backend/pixel format/low light. Try another backend, keep MJPG, brighten the room.
- **phase 1 fine, phase 2 age high** -> your loop is building a backlog (should not happen with the grabber).
- **`age`** is measured from when `read()` returned, so it excludes sensor and USB latency. For true glass-to-screen latency, film a running clock next to the preview.

## Design rules
1. Everything downstream consumes `FrameSource`; CCTV is a new URL, not new code.
2. Newest frame wins; no frame queues.
3. Capture full-res, process downscaled, crop from full-res.
4. Local-only. Never log stream credentials.
