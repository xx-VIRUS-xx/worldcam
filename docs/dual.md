# Dual-stream check (Task 6)

`DualGrabber` paces the pipeline on a low-res **sub**stream and cuts crops from a
high-res **main**stream. Sub and main are separate connections, so the main frame
used for a crop is not the same instant as the sub frame the box was detected on.
This page measures by how much.

## API

```python
d = DualGrabber(sub_src, main_src, main_ring=1)   # main_ring=1: old behaviour
f = d.get(after_seq=last)                          # sub frame, run detector on it
r = d.crop_with_frame(box, pad=0.1, ref=f, max_skew=None)
r.image     # crop (np.ndarray) or None
r.frame     # the Frame actually cropped (compare r.frame.ts with f.ts)
r.fallback  # True if it came from the substream (main down / too stale)
d.fallbacks # running count
d.crop(box, pad)  # unchanged: newest main frame, ndarray only
```

- `ref`: the sub frame the box came from. The main frame whose `ts` is closest to
  `ref.ts` is used (with `main_ring=1` that is simply the newest main frame). On
  fallback the crop is cut from `ref` itself, not from whatever sub frame is newest.
- `max_skew` (seconds): if the chosen main frame is further than this from `ref.ts`
  (e.g. mainstream stalled), fall back to `ref`. Off by default.
- `main_ring=N`: remember the last N main frames (bounded `deque(maxlen=N)`,
  references not copies, filled by the main grabber thread, never consumed).

## What is measured

`python -m worldcam.cli.dualcheck` paces on `get()`, sleeps `--work-ms`, crops the
fixed box `(0.25, 0.25, 0.75, 0.75)` and records
`offset_ms = (main_frame.ts - sub_frame.ts) * 1000` (signed and absolute). The
first `--warmup` seconds (default 2) are excluded.

**`Frame.ts` is software arrival time** (`time.monotonic()` after `read()`
returned) on each connection. A ts offset is therefore *arrival* skew, not
*content* skew: two streams with different decode/network latency can arrive in
lockstep and still show different moments. For the RTSP test I also measured
content skew: `--ref-clip` loads the looping source clip, and each received
frame is matched to its nearest clip frame (64x36 grayscale thumbnails, mean abs
diff); `(main_idx - sub_idx) / fps` is the content offset, quantised to one frame
(33.3 ms). The matcher was checked offline: all 600 clip frames, downscaled to
640x360 and JPEG-q50 re-encoded, matched their own index exactly. The clip has a
`%{pts} f%{n}` drawtext overlay, which makes frames unique; it was not OCR'd.

## Setup (local RTSP, Apple M2, macOS)

```sh
# mediamtx on 127.0.0.1:8654 only (rtmp/hls/webrtc/srt/moq/api off)
mediamtx mediamtx.yml
# 20 s 1280x720@30 testsrc2 clip with timestamp/frame overlay
ffmpeg -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=20" \
  -vf "drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial Bold.ttf:text='%{pts\:hms} f%{n}':fontsize=64:fontcolor=white:box=1:boxcolor=black:x=20:y=20" \
  -c:v libx264 -preset veryfast -g 30 -bf 0 -pix_fmt yuv420p main.mp4
# one ffmpeg, two outputs: /main 1280x720 and /sub 640x360 of the same frames
ffmpeg -re -stream_loop -1 -i main.mp4 \
  -filter_complex "[0:v]split=2[m][s0];[s0]scale=640:360[s]" \
  -map "[m]" -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -bf 0 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8654/main \
  -map "[s]" -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -bf 0 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8654/sub
```

## Results

### RTSP, newest main frame (`--ring 1`, the original behaviour)

```
$ python -m worldcam.cli.dualcheck --main rtsp://127.0.0.1:8654/main --sub rtsp://127.0.0.1:8654/sub --seconds 30 --work-ms 0 --ref-clip main.mp4
sub : {'url': 'rtsp://127.0.0.1:8654/sub', 'actual': '640x360'}
main: {'url': 'rtsp://127.0.0.1:8654/main', 'actual': '1280x720'}
[dualcheck work=0ms ring=1 30s] crops=875 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(1280, 720)]
  ts offset ms (signed) n=875 min=-322.6 p5=-43.0 p50=-31.9 p95=-5.5 max=8.0
  ts offset ms |abs|    n=875 p50=31.9 p95=43.0 max=322.6
  content skew ms (sgn) n=875 min=-333.3 p5=-33.3 p50=-33.3 p95=-33.3 max=0.0
  content skew |abs|    n=875 p50=33.3 p95=33.3 max=333.3
  sub age at get ms     n=875 p50=0.1 p95=1.0 max=28.8

$ python -m worldcam.cli.dualcheck --main rtsp://127.0.0.1:8654/main --sub rtsp://127.0.0.1:8654/sub --seconds 30 --work-ms 80 --ref-clip main.mp4
[dualcheck work=80ms ring=1 30s] crops=346 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(1280, 720)]
  ts offset ms (signed) n=346 min=1.7 p5=63.3 p50=72.3 p95=105.5 max=237.9
  ts offset ms |abs|    n=346 p50=72.3 p95=105.5 max=237.9
  content skew ms (sgn) n=346 min=0.0 p5=66.7 p50=66.7 p95=100.0 max=300.0
  content skew |abs|    n=346 p50=66.7 p95=100.0 max=300.0
  sub age at get ms     n=346 p50=18.2 p95=37.3 max=154.1
```

### RTSP, ts-matched from a ring of 5 main frames (`--ring 5`)

```
$ python -m worldcam.cli.dualcheck ... --seconds 30 --work-ms 80 --ring 5 --ref-clip main.mp4
[dualcheck work=80ms ring=5 30s] crops=346 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(1280, 720)]
  ts offset ms (signed) n=346 min=-36.8 p5=0.4 p50=0.9 p95=3.0 max=112.7
  ts offset ms |abs|    n=346 p50=0.9 p95=3.6 max=112.7
  content skew ms (sgn) n=346 min=-100.0 p5=0.0 p50=0.0 p95=0.0 max=200.0
  content skew |abs|    n=346 p50=0.0 p95=0.0 max=200.0
  sub age at get ms     n=346 p50=17.7 p95=36.0 max=84.1

$ python -m worldcam.cli.dualcheck ... --seconds 30 --work-ms 0 --ring 5 --ref-clip main.mp4
[dualcheck work=0ms ring=5 30s] crops=898 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(1280, 720)]
  ts offset ms (signed) n=898 min=-270.7 p5=-42.3 p50=-31.9 p95=-7.7 max=2.0
  ts offset ms |abs|    n=898 p50=31.8 p95=42.3 max=270.7
  content skew ms (sgn) n=898 min=-133.3 p5=-33.3 p50=-33.3 p95=-33.3 max=0.0
  content skew |abs|    n=898 p50=33.3 p95=33.3 max=133.3
  sub age at get ms     n=898 p50=0.1 p95=1.0 max=20.0
```

### Control: SyntheticSource vs SyntheticSource (no decode, no network)

`make_source("synthetic")` caps size at 640x360, so both are 640x360 here.

```
$ python -m worldcam.cli.dualcheck --main synthetic --sub synthetic --seconds 30 --work-ms 0 --ring 1
[dualcheck work=0ms ring=1 30s] crops=898 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(640, 360)]
  ts offset ms (signed) n=898 min=-172.6 p5=-36.0 p50=-4.3 p95=0.0 max=10.0
  ts offset ms |abs|    n=898 p50=4.3 p95=36.0 max=172.6
  sub age at get ms     n=898 p50=0.0 p95=0.4 max=29.7

$ python -m worldcam.cli.dualcheck --main synthetic --sub synthetic --seconds 30 --work-ms 80 --ring 1
[dualcheck work=80ms ring=1 30s] crops=359 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(640, 360)]
  ts offset ms (signed) n=359 min=56.6 p5=63.3 p50=95.2 p95=103.8 max=110.0
  ts offset ms |abs|    n=359 p50=95.2 p95=103.8 max=110.0
  sub age at get ms     n=359 p50=17.5 p95=32.8 max=45.0

$ python -m worldcam.cli.dualcheck --main synthetic --sub synthetic --seconds 30 --work-ms 80 --ring 5
[dualcheck work=80ms ring=5 30s] crops=359 fallbacks=0 | sub read=30.0 fps main read=30.0 fps | crop-source shapes=[(640, 360)]
  ts offset ms (signed) n=359 min=-5.9 p5=-3.5 p50=0.0 p95=3.9 max=5.1
  ts offset ms |abs|    n=359 p50=0.7 p95=4.3 max=5.9
```

## Findings

| run | ring | p50 \|ts offset\| | p95 \|ts offset\| | p50/p95 \|content skew\| |
|---|---|---|---|---|
| RTSP work 0  | 1 | 31.9 | 43.0  | 33.3 / 33.3 |
| RTSP work 80 | 1 | 72.3 | **105.5** | 66.7 / **100.0** |
| RTSP work 80 | 5 | 0.9  | 3.6   | 0.0 / 0.0 |
| RTSP work 0  | 5 | 31.8 | 42.3  | 33.3 / 33.3 |
| synth work 80 | 1 | 95.2 | 103.8 | n/a |
| synth work 80 | 5 | 0.7 | 4.3 | n/a |

1. **With newest-main crops and 80 ms of detector work, p95 |offset| is 105.5 ms
   (RTSP) / 103.8 ms (synthetic): above the 100 ms bar.** The cause is structural,
   not network: `crop()` peeks the newest main frame *after* the detector ran, so the
   main frame is newer than the sub frame by roughly the work time. Content skew
   confirms it (main shows 2-3 frames later than the sub frame the box came from).
   Any real detector cost adds directly to the skew.
2. **Fix, implemented (opt-in, `main_ring=N`)**: keep the last N main frames and
   pick the one closest in ts to the sub frame. At work 80 it drops p95 |ts offset|
   to 3.6 ms and content skew to 0 frames at p95. Ring size rule of thumb:
   `N >= ceil(work_ms * main_fps / 1000) + 2` (5 covers ~100 ms at 30 fps).
   Trade-offs: (a) memory, N full-res frames pinned (1080p BGR = 6.2 MB each, so
   N=5 is ~31 MB; 4K is ~125 MB); (b) the crop is older by the work time, which is
   the point (it matches what the detector saw) but means it is not the freshest
   view; (c) it matches *arrival* times, so a constant latency difference between
   the two connections is not corrected. It does not break the no-queue rule: the
   ring is bounded, never drained, and nothing waits on it. Default stays
   `main_ring=1` so existing behaviour is unchanged; callers opt in.
3. **At work 0 the main frame is one frame behind** (content -33.3 ms, p50 and p95)
   and the ring cannot help, since the matching main frame has not arrived yet
   when the crop is taken (the 1280x720 decode is slower than 640x360). Fixing that
   would need to wait up to one main frame period for a newer main frame; not done,
   33 ms is well under the bar.
4. Tails: single max outliers of 230-330 ms (RTSP) / 173 ms (synthetic at work 0)
   are one-off stalls of one grabber thread (GIL/decoder hiccups); they stay out of
   p95. `max_skew` lets a caller reject such crops (falls back to the sub frame).
5. Fallbacks were 0 after warmup. In a first run without warmup there were 10
   fallbacks at startup, because the main RTSP connection finished opening after
   the sub one.

Not measured: real-camera content skew (needs filming a running clock with both
cameras, a human step), and a stream that dies mid-run (Task 5 owns reconnects;
`max_skew` + `fallback` are the hooks for it).

## Run on real hardware later (FaceTime webcam + iPhone Continuity Camera)

Indices are TBD (check with the bench tool or System Information; Continuity
Camera usually appears as 1 when the iPhone is near and unlocked). The phone
is the higher-res main, the FaceTime camera the sub:

```sh
PYTHONPATH=$PWD/src .venv/bin/python -m worldcam.cli.dualcheck \
  --main 1 --sub 0 --main-size 1920x1080 --sub-size 640x480 \
  --seconds 30 --work-ms 0
PYTHONPATH=$PWD/src .venv/bin/python -m worldcam.cli.dualcheck \
  --main 1 --sub 0 --main-size 1920x1080 --sub-size 640x480 \
  --seconds 30 --work-ms 80 --ring 1
PYTHONPATH=$PWD/src .venv/bin/python -m worldcam.cli.dualcheck \
  --main 1 --sub 0 --main-size 1920x1080 --sub-size 640x480 \
  --seconds 30 --work-ms 80 --ring 5
```

Two different cameras have different sensor and transport latency (Continuity
Camera goes over Wi-Fi/USB), so ts offset there will NOT reflect content skew. To
get content skew, point both at a running millisecond clock and compare saved
crops by eye.
