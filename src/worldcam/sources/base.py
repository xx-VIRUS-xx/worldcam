from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import Frame


class FrameSource(ABC):
    """The only thing the rest of the pipeline knows about cameras.

    Laptop webcam, phone IP-cam app, and CCTV RTSP are all implementations of
    this. Downstream code must never import cv2.VideoCapture directly.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the device/stream. Raise RuntimeError on failure."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Block for the next frame. Return None on a transient failure
        (caller may retry) and raise EOFError when the source is finished."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Must be safe to call more than once."""

    def describe(self) -> dict:
        """Negotiated properties (actual resolution/fps/fourcc), for logs."""
        return {}

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
