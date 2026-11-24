from __future__ import annotations

from .base import FrameSource
from .file import FileSource, SyntheticSource
from .rtsp import StreamSource
from .webcam import WebcamSource


def make_source(spec: str, width: int = 1280, height: int = 720, fps: int = 30) -> FrameSource:
    """Build a source from a CLI-style spec string.

    "0", "1"               local camera index (laptop webcam / DroidCam)
    "rtsp://...", "http://..."  IP camera app or CCTV
    "synthetic"            generated frames, no hardware
    anything else          treated as a video file path
    """
    if spec == "synthetic":
        return SyntheticSource(width=min(width, 640), height=min(height, 360), fps=fps)
    if spec.isdigit():
        return WebcamSource(int(spec), width, height, fps)
    if spec.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return StreamSource(spec)
    return FileSource(spec)


__all__ = [
    "FrameSource", "WebcamSource", "StreamSource", "FileSource",
    "SyntheticSource", "make_source",
]
