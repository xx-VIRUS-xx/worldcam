"""Perception stages: motion gate, YOLO detector, IoU tracker, PerceptionLoop.

YoloDetector needs the optional `perception` extra (ultralytics) and is
imported lazily from .detector; everything else needs only opencv + numpy.
"""
from .motion import FrameDiffMotionGate
from .pipeline_runner import PerceptionLoop
from .tracker import IouTracker

__all__ = ["FrameDiffMotionGate", "IouTracker", "PerceptionLoop"]
