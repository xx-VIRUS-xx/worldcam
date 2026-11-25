# Running worldcam

This is a hands-on guide: set up, run each tool, and find where things live.
Every command runs from the repo root (`~/PY/worldcam`) on this Mac.

Current state: capture, detection + tracking, face identity, the Neo4j graph,
and the ambiguity engine (a `claude -p` worker that asks you on Telegram who
unknown people are) all exist. The VLM/activity stage is still only an interface.
Design: `docs/AMBIGUITY_ENGINE.md`.

---

## 1. One-time setup

```bash
cd ~/PY/worldcam
python3 -m venv .venv                 # already done; harmless to re-run
.venv/bin/pip install -e '.[dev]'     # installs worldcam + pytest; re-run after pyproject changes
source .venv/bin/activate             # optional: lets you drop the .venv/bin/ prefix
```

External tools, needed only for the RTSP and phone tests: `ffmpeg` and
`mediamtx`, both from Homebrew and both already installed.

**macOS camera permission:** the first camera run may show a permission prompt
for the app that launched Python (Terminal, iTerm or VS Code). If you get
`cannot open camera index 0`, enable that app under System Settings →
Privacy & Security → Camera, then restart it.

The examples below assume the venv is active. If it isn't, prefix each command
with `.venv/bin/`.

## 2. Check that everything works (no camera needed)

```bash
pytest -q                                                    # all pass; Neo4j tests skip if it is down
worldcam-preview --source synthetic --no-display --seconds 5 # generated frames, no hardware
```

## 3. The tools

| Command | What it's for |
|---|---|
| `worldcam-preview` | Live window with fps / frame-age overlay. With `--no-display`, a headless run that prints stats, used for soak tests. |
| `worldcam-bench` | Two-phase benchmark: raw read speed, then threaded grabber with fake work. The source of all performance numbers. |
| `worldcam-dualcheck` | Measures main-vs-sub stream time offset for the dual-stream crop path. |
| `worldcam-rtsp-soak` | Starts a local RTSP server, then kills and restarts it, to prove reconnects work. |

Every command takes `--help`.

### Sources (`--source ...`): same flag for every tool

| Spec | Meaning |
|---|---|
| `0` | FaceTime HD Camera (built-in) |
| `1` | iPhone via Continuity Camera. Only present while the phone is near, locked, on its side and still. The index shifts if devices change. |
| `rtsp://user:pass@ip/path` | CCTV or a phone RTSP app (credentials are never logged) |
| `http://ip:8080/video` | Phone IP-camera app (MJPEG) |
| `synthetic` | Generated frames, no hardware (capped at 640x360) |
| `path/to/file.mp4` | Video file, loops |

To list cameras: `ffmpeg -f avfoundation -list_devices true -i ""`
(on this Mac, ffmpeg's order matched OpenCV's).

### Live preview

```bash
worldcam-preview --source 0                        # 1280x720@30 by default; press q or ESC to quit
worldcam-preview --source 1 --width 1920 --height 1080
worldcam-preview --source http://<phone-ip>:8080/video
```

The overlay shows resolution, fps, frame age and dropped frames. A rising
**dropped** count is expected and good: it means stale frames are being skipped
instead of queued.

### Benchmark

```bash
worldcam-bench --source 0 --width 1280 --height 720 --fps 30 --seconds 30 --work-ms 80
```

How to read the output:

- `negotiated:` shows what the camera actually gave you. On macOS, FOURCC shows `n/a`, because AVFoundation doesn't report it.
- `[phase1 raw read]` is camera speed with no processing. **Slow here** means camera, driver or lighting, not our code.
- `[phase2 grabber]` is the threaded grabber plus `--work-ms` of fake detector work. `age at use p95` should stay below about 100 ms, and `dropped` should be > 0.

Diagnostic flags: `--backend {any,avfoundation,...}`, `--fourcc {MJPG,YUYV,none}`,
`--warmup SEC` (seconds of camera warm-up excluded from the stats, default 1).

**Pass bar (720p):** phase 1 ≥ 25 fps, gap p95 < 60 ms, phase 2 age p95 < 100 ms.

### Soak test (30 min)

```bash
mkdir -p captures
worldcam-preview --source 0 --no-display --seconds 1800 --stats-every 10 \
  --stats-csv captures/soak.csv | tee captures/soak.log
```

A `stats` line prints every 10 s, and a `summary` line prints at the end (also on Ctrl-C).
Check these fields:

- `rss_growth_mb`: must be < 50. It compares the mean memory use of the first minute
  (after warm-up) with the last minute. Single samples on macOS swing by tens
  of MB, so don't read anything into `rss_mb` going up and down.
- `gap_max_all_ms`, `fps_min`/`fps_mean`, `reconnects`. `reconnects` is always 0 for a webcam.

`captures/` is git-ignored.

### Dual-stream check (main + sub)

```bash
# Real hardware: iPhone as full-res main, FaceTime as sub (needs the iPhone present)
worldcam-dualcheck --main 1 --sub 0 --main-size 1920x1080 --sub-size 640x480 \
  --seconds 30 --work-ms 80 --ring 1     # then repeat with --ring 5
```

Compare `|offset| p95` with and without `--ring 5`. The local-RTSP version
(publish main and sub with ffmpeg) is in `docs/dual.md`.

### RTSP outage test

```bash
mkdir -p captures
ffmpeg -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=20" \
  -c:v libx264 -preset veryfast -g 30 -bf 0 -pix_fmt yuv420p captures/clip.mp4
worldcam-rtsp-soak --clip captures/clip.mp4 --port 8554
```

The script starts and stops its own mediamtx and ffmpeg. Each `[event]` line
should show `thread_alive=True`, `frames_while_down=0`, and `max_read` under
about 4.1 s.

### v2: the autonomous agent (`worldcam-run`)

`worldcam-run` now starts the whole agent: eyes (people, animals, objects), the brain (graph in
Neo4j + timeline/routines in `captures/brain.db`), the autonomous loop, doubts, and two-way chat.
Vision: `docs/VISION.md`. Architecture and file map: `docs/ARCHITECTURE_V2.md`.

```bash
worldcam-run --source 0 --channel log                       # dev: messages are logged, nothing sent
worldcam-run --source 0 --profiles cctv,companion           # which application intents are active
worldcam-run --source 0 --profiles cctv,companion,medical --scene   # + gated vision scene descriptions
worldcam-run --source 0 --no-engine                         # eyes + brain only (never asks, never chats)
```

First-time setup for real use:
1. `worldcam-telegram-setup` (bot token + your chat id into `.env`). You become the owner automatically.
2. `worldcam-users invite --role caregiver --profiles medical` -> the person sends `/join CODE` to the bot.
3. Models: `models.json` (copy `models.example.json`) maps each task to `claude:*` / `gemini:*`.
   Without it, everything runs on the `claude` CLI (it handles images too). No silent fallbacks.
4. Zones: `zones.example.json`; check with `worldcam-scene-replay --overlay frame.jpg --zones zones.json`.

Tools: `worldcam-brain` (timeline, routines, summaries, retention), `worldcam-users` (whitelist, roles),
`worldcam-graph` (people, doubts, forget). Fetch the animal/object re-id model once:
`python -m worldcam.perception.reid --fetch` (needs `pip install onnx`).

### Full pipeline: `worldcam-run`

One-time setup:

```bash
pip install -e '.[dev,all]'                   # YOLO, InsightFace, neo4j, mcp
docker compose up -d                          # Neo4j (password in .env, browse http://127.0.0.1:7474)
worldcam-graph init                           # schema + vector indexes
worldcam-telegram-setup                       # after putting TELEGRAM_BOT_TOKEN in .env
```

Run:

```bash
worldcam-run --source 0                       # everything, questions go to Telegram (auto selects Gemini if GEMINI_API_KEY is set)
worldcam-run --source 0 --worker gemini       # force Gemini API worker (reads GEMINI_API_KEY from .env)
worldcam-run --source 0 --worker claude       # force claude -p worker
worldcam-run --source 0 --notifier log        # questions only logged (no Telegram)
worldcam-run --source 0 --no-engine           # detect + identify + store, never ask
```

Smoke test Gemini on a person, animal, or object without a camera or Telegram:
```bash
PYTHONPATH=src .venv/bin/python -m worldcam.ambiguity.smoke --dir /tmp/wc-smoke \
  --worker gemini --entity-kind animal \
  --reply "That's Bruno, our golden retriever, he usually sleeps by the door"
```

What happens: a person is tracked, and their best snapshot's face is embedded
and matched against the graph. A stranger seen 3 times, a weak match or a
conflict opens a doubt. The engine runs `claude -p` (no shell or file tools,
only graph and ask-owner tools), which sends you a question with the crop. Your
reply triggers a second run that writes the name, relation and facts into the
graph. Limits: 6 questions per hour, quiet hours 23:00–08:00, 3 rounds per doubt.

Inspect and manage the graph:

```bash
worldcam-graph status | people | show <name|id> | doubts --status asked
worldcam-graph forget <person_id>             # hard-deletes person, faces, sightings, doubts + snapshot files
```

Other tools: `worldcam-detect --source 0` (live detection overlay),
`python -m worldcam.identity.bench` (face model latency/similarity).

## 4. Where is what

```
worldcam/
├── README.md              short overview + design rules
├── AGENT_PROMPT.md        week-1 brief given to the coding agents
├── pyproject.toml         deps (opencv-python, numpy) + the 4 commands above
├── docs/
│   ├── RUNNING.md         this file
│   ├── ROADMAP.md         architecture, conventions, week-by-week plan
│   ├── baseline.md        FaceTime camera numbers at 480p/720p/1080p + diagnosis
│   ├── phone.md           iPhone / IP-camera app / DroidCam / Camo setup + numbers
│   ├── rtsp.md            RTSP timeouts, reconnect behaviour, outage measurements
│   ├── dual.md            main/sub offset measurements and the ring fix
│   ├── AMBIGUITY_ENGINE.md design + decisions for perception → doubts → Telegram
│   ├── perception.md / identity.md / graph.md / ambiguity.md / telegram.md   per-component notes + numbers
├── src/worldcam/
│   ├── types.py           Frame(seq, ts, image): the one data type everything passes around
│   ├── capture.py         LatestFrameGrabber: background thread, keeps only the newest frame
│   ├── sources/           ← the ONLY place that touches cv2.VideoCapture
│   │   ├── base.py        FrameSource interface: open / read / close / describe
│   │   ├── __init__.py    make_source("0" | "rtsp://..." | "synthetic" | file)
│   │   ├── webcam.py      WebcamSource: local cameras, backend/FOURCC handling
│   │   ├── rtsp.py        StreamSource: RTSP/HTTP with timeouts, stall detection, reconnects
│   │   ├── file.py        FileSource (video file) and SyntheticSource (generated frames)
│   │   └── dual.py        DualGrabber: low-res sub paces detection, full-res main for crops
│   ├── perception/        motion gate, YOLO detector, IoU tracker, PerceptionLoop
│   ├── identity/          FaceAnalyzer (InsightFace) + IdentityResolver (match/ask/conflict)
│   ├── graph/             api.py (GraphStore contract + in-memory fake), store.py (Neo4j)
│   ├── ambiguity/         doubt triggers, engine, claude -p worker, MCP server, prompts/
│   ├── messenger/         Notifier contract + Telegram bot
│   ├── geometry.py        normalized (0..1) boxes/polygons, IoU, crop_norm
│   ├── metrics.py         RateMeter, percentile, RSS helpers, SoakStats
│   ├── pipeline.py        interfaces for the later stages (detector → … → memory); no code yet
│   └── cli/               run.py (whole pipeline), bench, preview, detect, graph_admin, telegram_setup, ...
├── tests/                 pytest suite; runs with no camera or network
└── captures/              (git-ignored) soak logs, CSVs, test clips
```

**Where to make common changes:**

- **New camera type:** add a class in `src/worldcam/sources/` and a branch in
  `make_source`. Nothing else should change.
- **Default resolution/fps:** the `--width/--height/--fps` defaults in `cli/*.py`.
  Recommended for this Mac: **1280x720 @ 30**. Avoid 640x480, which tops out at about 23 fps.
- **Crop with time matching:** `DualGrabber(sub, main, main_ring=5)` and
  `crop_with_frame(box, ref=sub_frame)`. This is opt-in; read the trade-offs in `docs/dual.md`.
- **RTSP timeouts and backoff:** `StreamSource(url, timeout_ms=4000, max_backoff=8.0, open_deadline=10.0)`.

## 5. Open items (need you)

1. Re-run the baseline in known **bright** and **low** light (`docs/baseline.md`).
2. Wake the iPhone, then re-run its 1080p bench and the dual-stream check above.
3. Decide whether to accept the `main_ring` frame buffer as a documented exception to "no frame queues".
4. Week 2: `WebcamSource` has no reconnect and picks cameras by index, which is fragile for Continuity Camera.
