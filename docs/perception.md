# Perception: motion gate, detector, tracker

Implements the first three stages of `docs/AMBIGUITY_ENGINE.md` (data flow):
`MotionGate -> Detector -> Tracker`, plus a stage-only `PerceptionLoop` and a
CLI. No identity or graph code here; the coordinator wires those in on top of
the `TrackUpdate`s the loop yields.

| File | What |
|---|---|
| `perception/motion.py` | `FrameDiffMotionGate`: running-average background on a 160 px gray copy; merged + padded normalized boxes; whole frame on first frame, every `full_every_s` (5 s), and on global changes (> 50 % of pixels) |
| `perception/detector.py` | `YoloDetector`: ultralytics `yolo11n.pt`, `mps` (falls back to `cpu`), class filter, `conf`, `imgsz`, `region_mode` filter/crop/full |
| `perception/tracker.py` | `IouTracker`: IoU + Hungarian matching per label, lifecycle, best snapshot |
| `perception/boxes.py` | normalized-box helpers; `overlaps_any` is the shared "observed" rule |
| `perception/pipeline_runner.py` | `PerceptionLoop(grabber, detector, tracker, gate)` -> `TrackUpdate` per frame |
| `cli/detect.py` | `python -m worldcam.cli.detect`: live overlay, or `--no-display --seconds N` benchmark |

Weights: `models/yolo11n.pt` (git-ignored). Lookup order: `$WORLDCAM_YOLO_WEIGHTS`,
`./models/`, `<repo>/models/`, `./yolo11n.pt`; if missing, ultralytics downloads it into `models/`.

## Data path
```
LatestFrameGrabber (1280x720 full-res)  or  DualGrabber (sub paces, main for crops)
  -> sub = 640-wide copy (single camera only; DualGrabber's frame already is the sub)
  -> gate.regions(sub)       []  => detector skipped, tracker told "nothing observed"
  -> detector.detect(sub, regions)              boxes normalized to the sub frame
  -> tracker.update(sub, dets, observed_regions=regions, crop_fn=...)
        best snapshot crop = crop_fn(box):
          single camera: crop_norm(full-res frame of the SAME grab)   reasons: src=main, main=1280x720
          DualGrabber:   crop_with_frame(box, ref=sub, max_skew=0.25)  reasons: src=main, skew_ms=..
                         fallback to sub                               reasons: src=sub, main_unavailable_or_skewed
          no crop_fn:    sub frame                                     reasons: src=sub, main_unavailable
```

## Design choices
- **Own IoU tracker, not `model.track(persist=True)`.** The gate skips the
  detector on still frames and the detector may only look at motion regions.
  ByteTrack/BoT-SORT inside `model.track` assume every frame is seen in full,
  so skipped frames would age tracks into lost. Our `update()` takes
  `observed_regions`: a track outside every observed region (or on a skipped
  frame) is not counted as a miss. It also keeps Detector and Tracker separate
  per the protocols and lets the tracker be tested with scripted detections and
  no weights. Indoors at 10–30 fps with few objects, IoU + Hungarian
  (scipy; greedy fallback) is enough; revisit (Kalman + low-score pass) if ids
  swap in crowded scenes.
- **Lifecycle.** NEW -> ACTIVE after `n_init`=3 matches. ACTIVE -> STATIONARY
  when the foot point (`box_foot`) stays within `still_eps`=0.03 for
  `still_after_s`=10 s; back to ACTIVE when it moves. LOST after `lost_misses`=2
  observed misses; ENDED `end_after_s`=3 s after becoming LOST (or after 120 s
  without any match, safety net). An ended track is removed from the live set
  in the same update that emits it, so it appears in `TrackUpdate.ended`
  exactly once. NEW tracks that die before confirmation are dropped silently
  (`stats["dropped_tentative"]`).
- **Best snapshot.** score = conf x box area x sharpness, sharpness = Laplacian
  variance of the box on the sub frame resized to 96 px height, squashed to
  0..1 (`v/(v+100)`). Scored on the cheap sub frame every match; the full-res
  crop is cut only when the score beats the current best by 5 %. `reasons`
  holds `conf=, area=, sharp=`, `first` / `largest` / `sharpest` /
  `most_confident` (which component improved), and the `src=` tags above.
- **Stationary people.** The gate returns the whole frame every 5 s, so the
  detector re-checks still people; between checks they are unobserved, not
  missed, so they stay STATIONARY instead of flapping to LOST.
- **Region mode = filter** (run on the whole 640x360 sub frame, drop detections
  whose box has < 10 % inside every region). Measured below: crop is not
  faster on MPS at this size (fixed per-call overhead dominates), and filter
  keeps the full context, so a person only partly inside the motion box is
  still detected whole.

## Measurements (2026-11-24, MacBook Air M2 8 GB, FaceTime HD 1280x720@30 MJPG, detection on a 640x360 copy, yolo11n imgsz 640 on MPS)

**Caveat: these runs happened while 4 other agents were working on the same
Mac (load average 25–27 on 8 cores).** Detector numbers are therefore noisy
and pessimistic; re-run on an idle machine for a clean baseline.

**Nobody was in front of the camera during any run.** The only detection was a
chair (tracked, confirmed, STATIONARY). No person tracks were produced or faked;
person tracking on the real camera still needs a run with someone in the scene.

### Detector on every frame (`--no-motion`), 30 s
```
$ python -m worldcam.cli.detect --source 0 --no-display --seconds 30 --warmup 3 --no-motion
detector: models/yolo11n.pt device=mps imgsz=640 conf=0.35 mode=filter load+warmup=9.2s
negotiated: {'requested': '1280x720@30 MJPG', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '1280x720@30 n/a'}

=== perception bench (30.0s measured, warm-up 3.0s excluded, wall 33.1s) ===
frames=484 e2e_fps=16.1 detector_calls=484 detector_fps=16.1 detector_capacity_fps=20.0
motion_skip_ratio=0.000 (gate=off) grabber_dropped=467
stage ms        p50     p95     max       n
  wait          0.01    2.95    23.1     484
  resize        0.32    0.70    17.0     484
  motion        0.00    0.00     0.0     484
  detect       50.09  131.07   270.9     484
  track         0.95    3.86    94.9     484
  total        51.85  132.44   272.3     484
tracks: created=1 confirmed=1 ended=0 dropped_tentative=0 max_concurrent=1 live_at_end=1
  ended by label: {} | live by label/state: {'chair:stationary': 1}
  best snapshots: src={'src=main': 1} updates=2 sub_fallbacks=0 crop sizes (w,h) sample=[(555, 383)]
```
Detector ≈ 16 fps end-to-end (p50 50 ms, p95 131 ms). An earlier 15 s run in
the same session gave 9.5 fps (p50 84 ms), which shows how much the load swings.
CPU (`--device cpu`, 15 s) was ~1.8 fps (p50 574 ms) under the same load, so
MPS is the right default.

### Scene empty, motion gate on (default), 30 s
```
$ python -m worldcam.cli.detect --source 0 --no-display --seconds 30 --warmup 3
detector: models/yolo11n.pt device=mps imgsz=640 conf=0.35 mode=filter load+warmup=7.5s
negotiated: {'requested': '1280x720@30 MJPG', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '1280x720@30 n/a'}

=== perception bench (30.0s measured, warm-up 3.0s excluded, wall 33.0s) ===
frames=879 e2e_fps=29.3 detector_calls=6 detector_fps=0.2 detector_capacity_fps=7.6
motion_skip_ratio=0.993 (gate=on) grabber_dropped=37
stage ms        p50     p95     max       n
  wait         32.32   45.49    79.6     879
  resize        0.27    0.94    10.6     879
  motion        0.21    1.42     7.8     879
  detect      131.68  322.73   322.7       6
  track         0.04    0.09    58.6     879
  total         0.57    2.54   333.2     879
tracks: created=1 confirmed=1 ended=0 dropped_tentative=0 max_concurrent=1 live_at_end=1
  ended by label: {} | live by label/state: {'chair:stationary': 1}
  best snapshots: src={'src=main': 1} updates=1 sub_fallbacks=0 crop sizes (w,h) sample=[(555, 383)]
```
- Skip ratio 99.3 %: the 6 detector calls are the 5 s periodic full-frame
  re-checks. The loop keeps up with the camera (29.3 fps, `wait` ≈ the 33 ms
  frame interval), costing ~0.6 ms per frame.
- Isolated detector calls are slower (p50 132 ms) than back-to-back ones
  (50 ms): MPS goes cold between calls 5 s apart. That's fine for periodic
  re-checks; when someone moves, calls are back-to-back and fast.

### Region mode: filter vs crop (MPS, one real frame, 640x360, 55 interleaved calls each)
```
device=mps sub=640x360 n=55 per cell (interleaved)
  filter  full            p50=  55.9 ms  p95= 107.0 ms
  filter  big(0.5x0.8)    p50=  63.8 ms  p95= 130.0 ms
  crop    big(0.5x0.8)    p50=  57.5 ms  p95= 131.9 ms
  filter  small(0.2x0.3)  p50=  52.7 ms  p95= 126.7 ms
  crop    small(0.2x0.3)  p50=  53.5 ms  p95= 140.2 ms
```
No measurable win for crop (differences are inside the noise), so `filter`
is the default. `--region-mode crop` stays available for larger sub frames or
CPU-only runs, where a smaller imgsz does pay off.

## Running
```bash
python -m worldcam.cli.detect --source 0                       # live overlay: boxes, #id label state, fps, motion reason
python -m worldcam.cli.detect --source 0 --no-display --seconds 30
python -m worldcam.cli.detect --source 0 --no-display --seconds 30 --no-motion   # detector every frame
python -m worldcam.cli.detect --source <sub> --main <main> --main-size 1920x1080  # DualGrabber, ring 5
```
Overlay colours: grey NEW, green ACTIVE, blue STATIONARY, red LOST; thin grey
rectangles are the motion regions of the current frame.

## Tests (no camera, no weights)
`tests/test_perception_motion.py`, `tests/test_perception_tracker.py`,
`tests/test_perception_loop.py`: moving square on synthetic frames, periodic
and global full-frame returns, lifecycle on scripted detections (NEW/ACTIVE,
STATIONARY and back, LOST and recovery, ENDED exactly once, unobserved ≠
missed), best-snapshot selection and crop source, normalized-box invariants,
loop with fake single and dual grabbers (detector sees only the 640 copy,
crop comes from the 1280 frame), detector filter/crop region logic with a
stubbed model. One YOLO smoke test is skipped when `yolo11n.pt` is missing.
