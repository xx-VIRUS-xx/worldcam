# Phone as a camera source

Two ways to use a phone, both without code changes outside `sources/`:

| Option | `--source` | Source class | Status on this machine |
|--------|-----------|--------------|------------------------|
| macOS Continuity Camera (iPhone) | `1` (camera index) | `WebcamSource` | Benchmarked (see below) |
| IP-camera app (RTSP / HTTP MJPEG) | `rtsp://...` or `http://.../video` | `StreamSource` | HTTP MJPEG path verified with a local ffmpeg server; no phone app tested |
| DroidCam / Camo (virtual webcam) | camera index | `WebcamSource` | Not installed; setup steps only |

Everything stays on the LAN or USB. No vendor cloud relay, no uploads.

Test machine: MacBook Air M2, macOS 26.6.2, OpenCV 5.1.0, iPhone model `iPhone17,1`.

## 1. Continuity Camera (iPhone)

### Camera index: **1**

How it was determined, without opening the FaceTime camera:

```
$ ffmpeg -f avfoundation -list_devices true -i ""
[0] FaceTime HD Camera
[1] <owner>'s iPhone Camera
[2] <owner>'s iPhone Desk View Camera
[3] Capture screen 0
[4] Capture screen 1
```

ffmpeg's order is not guaranteed to match OpenCV's. OpenCV's AVFoundation backend
enumerates with `AVCaptureDeviceDiscoverySession([builtInWideAngleCamera, externalUnknown])`.
A Swift script that only enumerates and opens nothing gave the same order:

```
== opencv-style [builtInWide, externalUnknown]
0 FaceTime HD Camera AVCaptureDeviceTypeBuiltInWideAngleCamera FDF90FEB-...
1 <owner>'s iPhone Camera AVCaptureDeviceTypeExternal 71719267-...
```

The Desk View camera and screens are not in OpenCV's list. A short cv2 open of index 1
then confirmed the device: requesting 1920x1440 returned a 1552x1552 frame, a square
Center Stage format that the FaceTime HD camera does not offer.

Caveats:
- The index depends on position in the list. When the iPhone disconnects, index 1 no
  longer exists. OpenCV prints `out device of bound (0-0): 1` and `WebcamSource.open()`
  raises `cannot open camera index 1`. It does not silently fall back to another camera.
  If another external camera is plugged in, the order can change, so check it again
  before a dual-stream run.
- `WebcamSource` has no reconnect. If Continuity Camera drops mid-run, `read()`
  returns `None` until the process restarts.

### Setup (human steps)
1. Sign in to the iPhone and the Mac with the same Apple ID, with Wi-Fi and Bluetooth on
   for both. Two-factor authentication must be on.
2. On the iPhone: Settings > General > AirPlay & Continuity (older iOS: AirPlay & Handoff),
   then turn on **Continuity Camera**.
3. Lock the iPhone, put it in landscape orientation and keep it still (a mount works
   best), close to the Mac. A USB cable makes the connection more stable.
4. Check that it is listed with `ffmpeg -f avfoundation -list_devices true -i ""`.
   It should appear as "<name>'s iPhone Camera".
5. If it is missing: unlock and relock the phone, or open FaceTime or Photo Booth once and
   choose the iPhone from the camera menu to wake it up.

### Measured (`--seconds 30 --work-ms 80`)

```
$ worldcam-bench --source 1 --width 1280 --height 720 --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '1280x720@30 MJPG', 'actual': '1280x720@30 ÿÿÿÿ'}
[phase1 raw read] 29.1 fps | frame gap ms p50=33.4 p95=41.7 max=149.9 | failures=0
[phase2 grabber, work=80ms] consumed=11.7 fps read=29.3 fps dropped=528 | age at use ms p50=17.4 p95=32.8

$ worldcam-bench --source 1 --width 1920 --height 1080 --fps 30 --seconds 30 --work-ms 80
negotiated: {'requested': '1920x1080@30 MJPG', 'actual': '1920x1080@30 ÿÿÿÿ'}
[phase1 raw read] 24.3 fps | frame gap ms p50=33.2 p95=69.4 max=1395.0 | failures=1
[phase2 grabber, work=80ms] consumed=10.1 fps read=24.4 fps dropped=430 | age at use ms p50=17.3 p95=38.0
```

| Requested | Negotiated | Phase 1 fps | Gap p95 | Gap max | Phase 2 age p95 | Dropped |
|-----------|-----------|-------------|---------|---------|-----------------|---------|
| 1280x720@30 | 1280x720@30 | 29.1 | 41.7 ms | 149.9 ms | 32.8 ms | 528 |
| 1920x1080@30 | 1920x1080@30 | 24.3 | 69.4 ms | 1395.0 ms | 38.0 ms | 430 |

- At 720p the camera meets the week 1 webcam criteria (>=25 fps, gap p95 < 60 ms, age p95 < 100 ms, dropped > 0).
- The 1080p run had one stall of about 1.4 s and one failed read, which lowered its
  averages. Right after that run the iPhone disappeared from the device list. A repeat
  1080p run failed with `cannot open camera index 1`, and the camera had not come back
  when this was written. So the 1080p numbers come from a single run that may have
  been degraded, possibly because the Continuity link was already dropping. Repeat the
  run once the phone is reconnected.
- The FOURCC column shows `ÿÿÿÿ` because AVFoundation returns -1 for
  `CAP_PROP_FOURCC` (see "Known issues").

## 2. IP-camera app via `StreamSource`

The phone runs an app that serves RTSP or HTTP MJPEG on the LAN. Pass its URL as
`--source`. Credentials in the URL are redacted in logs (`StreamSource._redacted`).

### Android: "IP Webcam" (Pavel Khlebovich)
1. Install from Play Store and open it. Under Video preferences, set the resolution
   (for example 1280x720) and quality.
2. Scroll down and tap **Start server**. The app shows `http://<phone-ip>:8080`.
3. Phone and Mac must be on the same Wi-Fi. Turn off "client isolation" or "guest
   network" on the router if they can't reach each other.
4. URLs:
   - MJPEG: `http://<phone-ip>:8080/video` (widely documented for this app)
   - RTSP: `rtsp://<phone-ip>:8080/h264_ulaw.sdp` (**verify**; the path depends on the app version and audio settings)
   - If you set a username and password in the app: `http://user:pass@<phone-ip>:8080/video` (**verify**)
5. `worldcam-bench --source http://<phone-ip>:8080/video --seconds 30 --work-ms 80`

### iOS: RTSP or MJPEG camera apps
App Store apps change often. Search "RTSP camera" or "IP camera". Most show their URL
on screen once streaming starts. Common patterns, all **verify**:
- `rtsp://<phone-ip>:8554/live` or `rtsp://<phone-ip>:554/...`
- `http://<phone-ip>:<port>/video` or `/mjpeg` for MJPEG mode
- iOS suspends background apps, so keep the app in the foreground with the screen on.
  Use Guided Access or turn off Auto-Lock.

Prefer RTSP (H.264) over MJPEG on Wi-Fi: MJPEG uses much more bandwidth for the same
resolution. `StreamSource` forces RTSP over TCP.

### Verified locally: HTTP MJPEG through `StreamSource`
No phone app was available, so a localhost-only stdlib HTTP server stood in for one.
It ran ffmpeg per client (`testsrc2=size=1280x720:rate=30`, `-c:v mjpeg -f mpjpeg`) and
served it as `multipart/x-mixed-replace;boundary=ffmpeg` at `http://127.0.0.1:8090/video`,
the same shape as IP Webcam's `/video`.

```
$ worldcam-bench --source http://127.0.0.1:8090/video --width 1280 --height 720 --fps 30 --seconds 30 --work-ms 80
negotiated: {'url': 'http://127.0.0.1:8090/video', 'actual': '1280x720'}
[phase1 raw read] 30.3 fps | frame gap ms p50=31.8 p95=69.6 max=356.7 | failures=0
[phase2 grabber, work=80ms] consumed=11.8 fps read=30.1 fps dropped=550 | age at use ms p50=21.6 p95=59.9
```

`StreamSource` (CAP_FFMPEG) accepts an HTTP MJPEG URL. `--width/--height/--fps` are
ignored for streams; the resolution is whatever the app sends. These numbers are
localhost only and say nothing about Wi-Fi latency or jitter from a real phone.

The first attempt used ffmpeg's own server (`-f mpjpeg -listen 1 http://127.0.0.1:8090/video`).
That server takes one client and then exits. Phase 1 worked (30.3 fps), but phase 2
reconnects immediately, before the server had restarted, and failed in `StreamSource.open()`:
`RuntimeError: cannot open stream http://127.0.0.1:8090/video`. Some phone apps also
allow only one viewer at a time, so do not run the preview and the bench against the
same phone stream simultaneously.

## 3. DroidCam / Camo (not installed, setup only)

Both install a macOS virtual camera and the phone shows up as a camera index, used
through `WebcamSource` exactly like Continuity Camera.

- **DroidCam** (Dev47Apps): install the phone app (Android or iOS) and the macOS client.
  Connect over Wi-Fi (enter the phone IP shown in the app) or USB. Find the index with
  `ffmpeg -f avfoundation -list_devices true -i ""`, and check the OpenCV index as in §1.
  The free tier is limited to lower resolutions (**verify** the current limits).
  DroidCam also serves an MJPEG stream at `http://<phone-ip>:4747/video` (**verify**),
  which can go to `StreamSource` without the Mac client.
- **Camo** (Reincubate): install Camo on the phone and Camo Studio on the Mac, then
  connect by USB or Wi-Fi. The virtual camera appears as "Reincubate Camo" (**verify**
  the name) and needs Camo Studio running. Some resolutions need the paid tier.
- On recent macOS, virtual cameras are Camera Extensions. Approve them in
  System Settings > General > Login Items & Extensions (or Privacy & Security), and
  check whether OpenCV's discovery session (`externalUnknown`) lists them.
  **Not verified**: a virtual camera may not appear in OpenCV even when ffmpeg lists it.

## Known issues found (not fixed here; source files are out of scope for this task)
1. `WebcamSource.describe()` prints FOURCC as `ÿÿÿÿ` on AVFoundation:
   `CAP_PROP_FOURCC` returns -1 and `_fourcc_to_str(-1)` decodes it to four 0xFF bytes.
   It should print `n/a` when the value is <= 0.
2. `StreamSource.open()` gives up on the first failed connect, while `read()` retries
   with backoff. A phone app that is briefly unreachable at startup crashes the grabber
   start (see the phase 2 failure above).
3. `WebcamSource` has no reconnect, and the camera index depends on device order. A
   dropped Continuity Camera needs a restart, and the index can change when devices are
   added or removed. Selecting the camera by AVFoundation unique ID or name would be
   more robust.

## Could not verify (needs a human)
- **iPhone 1080p repeatability and recovery**: the iPhone disconnected after the first
  1080p run and had not come back. Wake it (unlock and relock, or pick it once in
  FaceTime or Photo Booth), check that `ffmpeg -list_devices` shows it again, then
  re-run the 1080p bench.
- A real IP-camera app on a phone over Wi-Fi (RTSP and MJPEG). Install the app, then
  give the phone IP and URL.
- DroidCam and Camo (not installed).
