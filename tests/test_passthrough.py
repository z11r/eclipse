import numpy as np
import pytest

from eclipse import main as app
from eclipse.capture.pattern import TestPattern
from eclipse.capture.screen import to_bgr
from eclipse.compositor.compositor import Compositor
from eclipse.output.virtual_cam import OUTPUT_HEIGHT, OUTPUT_WIDTH, fit


class FakeSink:
    def __init__(self, *args, **kwargs):
        self.images = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        self.images.append(image)


def test_to_bgr_drops_alpha_and_keeps_channel_order():
    raw = np.zeros((4, 6, 4), dtype=np.uint8)
    raw[:, :] = (10, 20, 30, 255)
    image = to_bgr(raw, max_width=1920)
    assert image.shape == (4, 6, 3)
    assert tuple(image[0, 0]) == (10, 20, 30)


def test_to_bgr_shrinks_to_max_width_with_even_size():
    raw = np.zeros((1964, 3024, 4), dtype=np.uint8)
    image = to_bgr(raw, max_width=1511)
    height, width = image.shape[:2]
    assert width <= 1511
    assert width % 2 == 0 and height % 2 == 0


def test_to_bgr_rounds_odd_size_down_even_without_shrinking():
    raw = np.zeros((101, 201, 4), dtype=np.uint8)
    assert to_bgr(raw, max_width=1920).shape[:2] == (100, 200)


@pytest.mark.parametrize("size", [(982, 1512), (720, 1280), (1964, 3024), (1000, 400), (1, 1)])
def test_fit_always_gives_the_exact_output_size(size):
    image = np.full((*size, 3), 200, dtype=np.uint8)
    assert fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT).shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)


def test_fit_adds_black_bars_instead_of_stretching():
    image = np.full((982, 1512, 3), 200, dtype=np.uint8)
    output = fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    assert tuple(output[540, 5]) == (0, 0, 0)
    assert tuple(output[540, 960]) == (200, 200, 200)


def test_pattern_frames_follow_the_contract():
    source = TestPattern(width=320, height=240)
    first, second = source.read(), source.read()
    assert first.image.shape == (240, 320, 3)
    assert first.image.dtype == np.uint8
    assert (first.frame_id, second.frame_id) == (0, 1)
    assert second.ts >= first.ts


def test_run_stops_at_max_frames():
    sink = FakeSink()
    latencies = app.run(TestPattern(320, 240), sink, lambda f: [], Compositor(), max_frames=5)
    assert len(sink.images) == 5
    assert len(latencies) == 5


def test_real_screen_is_refused_by_default(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("capture must not start")

    monkeypatch.setattr(app, "ScreenCapture", fail)
    assert app.main([]) == 2


def test_blank_screen_capture_stops_instead_of_streaming(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("capture must not start")

    monkeypatch.setattr(app, "screen_capture_allowed", lambda: False)
    monkeypatch.setattr(app, "ScreenCapture", fail)
    assert app.main(["--unsafe-passthrough"]) == 1


def test_test_pattern_needs_no_screen_permission(monkeypatch):
    monkeypatch.setattr(app, "screen_capture_allowed", lambda: False)
    monkeypatch.setattr(app, "VirtualCam", FakeSink)
    assert app.main(["--test-pattern", "--max-frames", "3"]) == 0
