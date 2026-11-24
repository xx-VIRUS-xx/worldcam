from __future__ import annotations

import sys
import time
from typing import Optional

import cv2

from ..types import Frame
from .base import FrameSource


def _default_backend() -> int:
    # The default backend is the most common cause of laggy webcams on Windows.
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


# CLI backend names -> OpenCV capture API ids. "any" lets OpenCV pick.
BACKENDS: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
}

# FOURCC values the bench accepts. "none" means: do not set CAP_PROP_FOURCC.
FOURCCS = ("MJPG", "YUYV", "none")


def backend_from_name(name: Optional[str]) -> Optional[int]:
    """Map a CLI backend name to a cv2.CAP_* id. None -> platform default."""
    if name is None:
        return None
    key = name.strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[key]


def normalize_fourcc(fourcc: Optional[str]) -> Optional[str]:
    """'none'/''/None -> None (leave the driver default); else a 4-char code."""
    if fourcc is None or fourcc.strip().lower() in ("", "none"):
        return None
    code = fourcc.strip().upper()
    if len(code) != 4:
        raise ValueError(f"fourcc must be 4 characters or 'none', got {fourcc!r}")
    return code


def _backend_to_str(backend: int) -> str:
    for name, val in BACKENDS.items():
        if val == backend:
            return name
    return str(backend)


def _fourcc_to_str(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


class WebcamSource(FrameSource):
    """Local camera (laptop webcam or a phone exposed as a webcam by
    DroidCam/Camo). Requests MJPG first: YUYV caps FPS at high resolutions
    because of USB bandwidth.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fourcc: Optional[str] = "MJPG",
        backend: Optional[int] = None,
    ):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.fourcc = normalize_fourcc(fourcc)
        self.backend = _default_backend() if backend is None else backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"cannot open camera index {self.index} "
                f"(backend={_backend_to_str(self.backend)}); on macOS check camera permission"
            )
        # Order matters: fourcc first, then size, then fps.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # not honoured by every backend
        self._cap = cap

    def read(self) -> Optional[Frame]:
        assert self._cap is not None, "open() first"
        ok, img = self._cap.read()
        ts = time.monotonic()
        if not ok or img is None:
            return None
        self._seq += 1
        return Frame(self._seq, ts, img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict:
        if self._cap is None:
            return {}
        c = self._cap
        return {
            "requested": f"{self.width}x{self.height}@{self.fps} {self.fourcc or 'none'}",
            "backend": f"{_backend_to_str(self.backend)} -> {c.getBackendName()}",
            "actual": (
                f"{int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                f"@{c.get(cv2.CAP_PROP_FPS):.0f} "
                f"{_fourcc_to_str(c.get(cv2.CAP_PROP_FOURCC))}"
            ),
        }
