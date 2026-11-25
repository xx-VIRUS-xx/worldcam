from __future__ import annotations

import os

from typing import Optional

from .base import FrameSource
from .file import FileSource, SyntheticSource
from .rtsp import StreamSource
from .webcam import WebcamSource, backend_from_name


# The camera every tool uses unless --source says otherwise: this laptop's camera.
# Override once for a shell/session with:  export WORLDCAM_SOURCE=0
DEFAULT_SOURCE = os.environ.get("WORLDCAM_SOURCE", "1")


def make_source(
    spec: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    *,
    backend: Optional[str] = None,
    fourcc: Optional[str] = None,
) -> FrameSource:
    """Build a source from a CLI-style spec string.

    "0", "1"               local camera index (laptop webcam / DroidCam)
    "rtsp://...", "http://..."  IP camera app or CCTV
    "synthetic"            generated frames, no hardware
    anything else          treated as a video file path

    `backend` ("avfoundation", "dshow", "msmf", "v4l2", "any"; None = platform
    default) and `fourcc` ("MJPG", "YUYV", "none"; None = WebcamSource default
    MJPG) only apply to camera indices; other specs ignore them.
    """
    if spec == "synthetic":
        return SyntheticSource(width=min(width, 640), height=min(height, 360), fps=fps)
    if spec.isdigit():
        kw: dict = {"backend": backend_from_name(backend)}
        if fourcc is not None:
            kw["fourcc"] = fourcc
        return WebcamSource(int(spec), width, height, fps, **kw)
    if spec.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return StreamSource(spec)
    return FileSource(spec)


__all__ = [
    "FrameSource", "WebcamSource", "StreamSource", "FileSource",
    "SyntheticSource", "make_source", "DEFAULT_SOURCE",
]
