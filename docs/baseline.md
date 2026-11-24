# Capture baseline (week 1, tasks 2-3)

## Hardware / software
| | |
|---|---|
| OS | macOS 26.6.2 (build 25G83) |
| Laptop | MacBook Air M2 (Mac14,2) |
| Camera | FaceTime HD Camera (built-in), cv2 index 0, AVFoundation |
| OpenCV / NumPy | 5.1.0 / 2.5.5 (as reported by `cv2.__version__`, `numpy.__version__` in the shared venv) |
| Lighting | **Unknown / not controlled.** Indoor room, not changed between runs. A human needs to repeat the runs in known good light and in low light (see "Human steps"). |

### Index check
`ffmpeg -f avfoundation -list_devices true -i ""` lists `[0] FaceTime HD Camera`, `[1] iPhone Camera`,
`[2] iPhone Desk View Camera`, plus screens. To confirm that **cv2** index 0 is the same device,
one frame was grabbed by name with ffmpeg (`-i "FaceTime HD Camera:none"`) and one with
`cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)`. The scenes matched (downscaled grey correlation 0.829, mean luma 152.2 vs 150.1).
The images were deleted after the check. The phone agent independently confirmed index 0 = FaceTime and index 1 = iPhone.

FaceTime modes AVFoundation advertises (`ffmpeg ... -video_size 1x1`): 1920x1080, 1280x720, 640x480,
1760x1328, 1552x1552 and portrait variants, all `[15, 30] fps`.

## Method
`worldcam-bench --source 0 --width W --height H --fps 30 --seconds 30 --work-ms 80` (default backend = AVFoundation, default FOURCC request = MJPG).

- Phase 1: raw `read()` loop. Phase 2: `LatestFrameGrabber` plus 80 ms of simulated work per frame.
- **Warm-up:** the first 1.0 s after open (`--warmup`, default 1.0) is excluded from phase 1 fps/gap
  stats and phase 2 age stats. Without this, the first gap measures camera start-up rather than steady-state delivery.
  Time-to-first-frame is printed separately. Some max gaps of 120-140 ms remain even after the warm-up is excluded (see table).
- AVFoundation ignores the requested FOURCC and does not report one: `CAP_PROP_FOURCC` returns -1.
  The first baseline runs printed this as `ÿÿÿÿ`. `describe()` now prints `n/a` instead. The pixel format actually delivered is unknown via OpenCV.
  `CAP_PROP_BUFFERSIZE` is set but is not honoured by AVFoundation either.

## Results
| # | Run | Negotiated (res@fps, fourcc) | Phase 1 fps | Gap p50 / p95 / max ms | Phase 2 age p50 / p95 ms | Dropped | Phase 2 read fps |
|---|-----|---|---|---|---|---|---|
| 1 | 640x480, avfoundation, MJPG | 640x480@30, n/a | **22.8** | 43.2 / 51.5 / 139.0 | 21.3 / 43.4 | 323 | 22.3 |
| 2 | 1280x720, avfoundation, MJPG | 1280x720@30, n/a | **29.8** | 33.2 / 40.0 / 121.9 | 17.7 / 33.4 | 527 | 29.2 |
| 3 | 1920x1080, avfoundation, MJPG | 1920x1080@30, n/a | **29.6** | 33.3 / 40.1 / 120.1 | 16.6 / 35.7 | 523 | 29.1 |
| 4 | 1280x720, **any**, MJPG | 1280x720@30, n/a (any -> AVFOUNDATION) | 29.9 | 33.9 / 41.6 / 72.2 | 17.5 / 34.2 | 535 | 29.6 |
| 5 | 640x480, avfoundation, MJPG (repeat of #1) | 640x480@30, n/a | 23.1 | 43.9 / 50.0 / 67.6 | 21.3 / 42.2 | 330 | 22.7 |
| 6 | 640x480, **any**, MJPG | 640x480@30, n/a (any -> AVFOUNDATION) | 23.1 | 43.7 / 51.0 / 64.6 | 22.2 / 41.5 | 328 | 22.7 |
| 7 | 640x480, avfoundation, FOURCC **none** | 640x480@30, n/a | 23.1 | 43.9 / 51.3 / 82.2 | 22.3 / 44.1 | 328 | 22.7 |

Cross-check outside the repo tools, using ffmpeg alone (so not a repo performance claim, only evidence about where the limit sits).
The command was `ffmpeg -f avfoundation -framerate 30 -video_size S -pixel_format uyvy422 -i "FaceTime HD Camera:none" -t 12 -f null -`:
- 640x480: 277 frames / 12 s = **23.1 fps**
- 1280x720: 360 frames / 12 s = **30.0 fps**

### Against the acceptance criteria
- 720p phase 1 >= 25 fps: **met** (29.8 fps). Gap p95 < 60 ms: **met** (40.0 ms).
- Phase 2 age p95 < 100 ms with `--work-ms 80`: **met** at all resolutions (33-44 ms). dropped > 0: **met** (323-535), so frames are being skipped rather than queued.
- Lighting was not controlled, so this is not yet a confirmed "adequate lighting" result, and no low-light numbers exist.

## Diagnosis (task 3)
720p phase 1 is 29.8 fps, which is above the 25 fps threshold, so the formal diagnosis sequence was not required.
The one anomaly is **640x480 at ~23 fps** even though the mode is advertised at 30 fps. I varied one thing at a time:
- Backend `any` vs `avfoundation`: no change (23.1 vs 23.1). On macOS, `any` resolves to AVFOUNDATION anyway (runs 4 and 6).
- FOURCC MJPG vs none: no change (23.1 vs 23.1). This is expected because AVFoundation ignores it.
- Resolution: 640x480 gives 23 fps. 1280x720 and 1920x1080 give 30 fps.
- Plain ffmpeg on the same device also gets 23.1 fps at 640x480 and 30.0 fps at 720p. So the limit is in the camera / AVFoundation 640x480 mode,
  not in OpenCV or in our read loop.

That is why **FfmpegPipeSource was not implemented**: OpenCV capture is stable (0 failures in every run) and matches ffmpeg.
Unverified hypothesis: 640x480 is a 4:3 crop that meters differently, so auto-exposure stretches the frame time (~43 ms ≈ 1/23 s) within the
device's 15-30 fps range. Testing this needs controlled lighting (see below).

Recommended default for this machine: **1280x720 @ 30, AVFoundation** (the current default). 1080p reaches the same rate if the extra pixels are needed.
Avoid 640x480 as a capture mode. Downscale 720p for detection instead (which is the roadmap rule anyway).

## Human steps (could not verify)
1. Lighting: repeat runs 1-3 in known bright light and in low light. Record the 720p fps for both, and whether 640x480 reaches 30 fps in bright light.
2. Windows/Linux backend notes (DSHOW/MSMF/V4L2, MJPG vs YUYV, USB port) were not tested because no such machine was available. The flags exist for that.

<details>
<summary>Raw output</summary>

Runs 1-3 were made before the FOURCC display fix, so they show `ÿÿÿÿ` where later runs show `n/a`.

```
### $ worldcam-bench --source 0 --width 640 --height 480 --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '640x480@30 MJPG', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '640x480@30 ÿÿÿÿ'}
[phase1 raw read] 22.8 fps | frame gap ms p50=43.2 p95=51.5 max=139.0 | failures=0 | first frame 140 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.6 fps read=22.3 fps dropped=323 | age at use ms p50=21.3 p95=43.4
### $ worldcam-bench --source 0 --width 1280 --height 720 --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '1280x720@30 MJPG', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '1280x720@30 ÿÿÿÿ'}
[phase1 raw read] 29.8 fps | frame gap ms p50=33.2 p95=40.0 max=121.9 | failures=0 | first frame 26 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.7 fps read=29.2 fps dropped=527 | age at use ms p50=17.7 p95=33.4
### $ worldcam-bench --source 0 --width 1920 --height 1080 --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '1920x1080@30 MJPG', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '1920x1080@30 ÿÿÿÿ'}
[phase1 raw read] 29.6 fps | frame gap ms p50=33.3 p95=40.1 max=120.1 | failures=0 | first frame 2 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.7 fps read=29.1 fps dropped=523 | age at use ms p50=16.6 p95=35.7
### $ worldcam-bench --source 0 --width 1280 --height 720 --backend any --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '1280x720@30 MJPG', 'backend': 'any -> AVFOUNDATION', 'actual': '1280x720@30 n/a'}
[phase1 raw read] 29.9 fps | frame gap ms p50=33.9 p95=41.6 max=72.2 | failures=0 | first frame 220 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.8 fps read=29.6 fps dropped=535 | age at use ms p50=17.5 p95=34.2
### $ worldcam-bench --source 0 --width 640 --height 480 --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '640x480@30 MJPG', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '640x480@30 n/a'}
[phase1 raw read] 23.1 fps | frame gap ms p50=43.9 p95=50.0 max=67.6 | failures=0 | first frame 5 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.8 fps read=22.7 fps dropped=330 | age at use ms p50=21.3 p95=42.2
### $ worldcam-bench --source 0 --width 640 --height 480 --backend any --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '640x480@30 MJPG', 'backend': 'any -> AVFOUNDATION', 'actual': '640x480@30 n/a'}
[phase1 raw read] 23.1 fps | frame gap ms p50=43.7 p95=51.0 max=64.6 | failures=0 | first frame 5 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.8 fps read=22.7 fps dropped=328 | age at use ms p50=22.2 p95=41.5
### $ worldcam-bench --source 0 --width 640 --height 480 --fourcc none --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '640x480@30 none', 'backend': 'avfoundation -> AVFOUNDATION', 'actual': '640x480@30 n/a'}
[phase1 raw read] 23.1 fps | frame gap ms p50=43.9 p95=51.3 max=82.2 | failures=0 | first frame 118 ms, warm-up 1.0s excluded
[phase2 grabber, work=80ms] consumed=11.8 fps read=22.7 fps dropped=328 | age at use ms p50=22.3 p95=44.1

### ffmpeg cross-check (not a repo tool)
### ffmpeg -f avfoundation -framerate 30 -video_size 640x480 -i 'FaceTime HD Camera:none' -t 12 -f null -
frame=  277 fps= 24 q=-0.0 Lsize=N/A time=00:00:12.00 bitrate=N/A speed=1.03x
### ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i 'FaceTime HD Camera:none' -t 12 -f null -
frame=  360 fps= 30 q=-0.0 Lsize=N/A time=00:00:12.00 bitrate=N/A speed=1.01x

### ffmpeg mode listing
[avfoundation] Supported modes:
  1920x1080@[15.000000 30.000000]fps
  1280x720@[15.000000 30.000000]fps
  1080x1920@[15.000000 30.000000]fps
  1760x1328@[15.000000 30.000000]fps
  640x480@[15.000000 30.000000]fps
  1328x1760@[15.000000 30.000000]fps
  1552x1552@[15.000000 30.000000]fps
```
</details>
