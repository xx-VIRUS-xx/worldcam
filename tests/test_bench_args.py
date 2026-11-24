"""Camera-free tests for bench flags and make_source backend/fourcc plumbing."""
import cv2
import pytest

from worldcam.cli import bench
from worldcam.sources import FileSource, SyntheticSource, make_source
from worldcam.sources.rtsp import StreamSource
from worldcam.sources.webcam import (
    WebcamSource, _default_backend, backend_from_name, normalize_fourcc,
)


def test_parser_defaults_leave_backend_and_fourcc_unset():
    a = bench.build_parser().parse_args([])
    assert a.backend is None and a.fourcc is None
    assert a.warmup == 1.0


@pytest.mark.parametrize("val,expected", [
    ("avfoundation", "avfoundation"), ("ANY", "any"), ("dshow", "dshow"),
    ("msmf", "msmf"), ("V4L2", "v4l2"),
])
def test_parser_backend_choices(val, expected):
    assert bench.build_parser().parse_args(["--backend", val]).backend == expected


@pytest.mark.parametrize("val,expected", [
    ("mjpg", "MJPG"), ("YUYV", "YUYV"), ("none", "none"), ("NONE", "none"),
])
def test_parser_fourcc_choices(val, expected):
    assert bench.build_parser().parse_args(["--fourcc", val]).fourcc == expected


@pytest.mark.parametrize("flag,val", [("--backend", "gstreamer"), ("--fourcc", "H264")])
def test_parser_rejects_unknown(flag, val):
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args([flag, val])


def test_backend_name_mapping():
    assert backend_from_name(None) is None
    assert backend_from_name("avfoundation") == cv2.CAP_AVFOUNDATION
    assert backend_from_name("any") == cv2.CAP_ANY
    assert backend_from_name("dshow") == cv2.CAP_DSHOW
    assert backend_from_name("msmf") == cv2.CAP_MSMF
    assert backend_from_name("v4l2") == cv2.CAP_V4L2
    with pytest.raises(ValueError):
        backend_from_name("nope")


def test_normalize_fourcc():
    assert normalize_fourcc(None) is None
    assert normalize_fourcc("none") is None
    assert normalize_fourcc("") is None
    assert normalize_fourcc("mjpg") == "MJPG"
    with pytest.raises(ValueError):
        normalize_fourcc("MJPEG")


def test_make_source_positional_signature_unchanged():
    src = make_source("3", 640, 480, 15)  # never opened: no camera touched
    assert isinstance(src, WebcamSource)
    assert (src.index, src.width, src.height, src.fps) == (3, 640, 480, 15)
    assert src.fourcc == "MJPG"
    assert src.backend == _default_backend()


def test_make_source_passes_backend_and_fourcc_to_webcam():
    src = make_source("0", 1280, 720, 30, backend="any", fourcc="none")
    assert isinstance(src, WebcamSource)
    assert src.backend == cv2.CAP_ANY
    assert src.fourcc is None
    src = make_source("1", backend="avfoundation", fourcc="YUYV")
    assert src.backend == cv2.CAP_AVFOUNDATION and src.fourcc == "YUYV"


def test_make_source_non_digit_specs_ignore_camera_kwargs():
    kw = dict(backend="dshow", fourcc="YUYV")
    assert isinstance(make_source("synthetic", **kw), SyntheticSource)
    s = make_source("rtsp://127.0.0.1:8554/x", **kw)
    assert isinstance(s, StreamSource) and s.url == "rtsp://127.0.0.1:8554/x"
    assert isinstance(make_source("clip.mp4", **kw), FileSource)


def test_bench_runs_on_synthetic_with_warmup(capsys):
    bench.main(["--source", "synthetic", "--seconds", "1.5", "--warmup", "0.5",
                "--work-ms", "20", "--backend", "any", "--fourcc", "none"])
    out = capsys.readouterr().out
    assert "[phase1 raw read]" in out and "warm-up 0.5s excluded" in out
    assert "[phase2 grabber" in out


def test_bench_rejects_warmup_longer_than_run():
    with pytest.raises(SystemExit):
        bench.main(["--source", "synthetic", "--seconds", "1", "--warmup", "2"])
