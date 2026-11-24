"""Camera-free tests for WebcamSource helpers."""
import cv2

from worldcam.sources.webcam import _fourcc_to_str


def test_fourcc_to_str_decodes_real_codes():
    assert _fourcc_to_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
    assert _fourcc_to_str(cv2.VideoWriter_fourcc(*"YUYV")) == "YUYV"


def test_fourcc_to_str_flags_unreported_values():
    # AVFoundation returns -1 for CAP_PROP_FOURCC (would print "ÿÿÿÿ"); 0 = not set.
    assert _fourcc_to_str(-1.0) == "n/a"
    assert _fourcc_to_str(0.0) == "n/a"


def test_fourcc_to_str_flags_nonprintable_codes():
    assert _fourcc_to_str(0x01020304) == "n/a"
