import numpy as np
import pytest

from eclipse import main as app
from eclipse.capture.pattern import TestPattern
from eclipse.capture.screen import to_bgr
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


def test_fit_stays_cheap_when_opencv_has_only_one_thread():
    # A resize that needs OpenCV's thread pool misses output ticks whenever another
    # thread is using it, so its cost on one thread is the one that counts.
    import time

    import cv2

    image = np.full((1964, 3024, 3), 90, dtype=np.uint8)
    image[::3] = 220
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        times = []
        for _ in range(5):
            started = time.perf_counter()
            fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            times.append(time.perf_counter() - started)
    finally:
        cv2.setNumThreads(threads)
    assert min(times) < 0.016


def test_pattern_frames_follow_the_contract():
    source = TestPattern(width=320, height=240)
    first, second = source.read(), source.read()
    assert first.image.shape == (240, 320, 3)
    assert first.image.dtype == np.uint8
    assert (first.frame_id, second.frame_id) == (0, 1)
    assert second.ts >= first.ts


def test_blank_screen_capture_stops_instead_of_streaming(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("capture must not start")

    monkeypatch.setattr(app, "screen_capture_allowed", lambda: False)
    monkeypatch.setattr(app, "ScreenCapture", fail)
    assert app.main([]) == 1


def test_real_screen_always_goes_through_the_text_layer(monkeypatch):
    seen = []

    def detector():
        def detect(frame):
            seen.append(frame.frame_id)
            return []

        return detect

    monkeypatch.setattr(app, "screen_capture_allowed", lambda: True)
    monkeypatch.setattr(app, "screen_capture_kit_available", lambda: False)
    monkeypatch.setattr(
        app, "ScreenCapture", lambda max_width, retina, first_id=0: TestPattern(320, 240)
    )
    monkeypatch.setattr(app, "TextDetector", detector)
    monkeypatch.setattr(app, "VirtualCam", FakeSink)
    assert app.main(["--max-frames", "3", "--delay", "0.05"]) == 0
    assert seen == [0, 1, 2]


def test_there_is_no_way_to_skip_scanning():
    with pytest.raises(SystemExit):
        app.main(["--unsafe-passthrough"])


def test_test_pattern_needs_no_screen_permission(monkeypatch):
    monkeypatch.setattr(app, "screen_capture_allowed", lambda: False)
    monkeypatch.setattr(app, "VirtualCam", FakeSink)
    assert app.main(["--test-pattern", "--max-frames", "3", "--delay", "0.05"]) == 0


def test_stats_log_has_numbers_and_no_screen_text(monkeypatch, tmp_path):
    log = tmp_path / "run.log"
    monkeypatch.setattr(app, "VirtualCam", FakeSink)
    monkeypatch.setattr(app.StatsLog, "__init__", _quick_stats_log)
    assert app.main(["--test-pattern", "--max-frames", "40", "--delay", "0.05", "--fps", "60", "--log", str(log)]) == 0
    text = log.read_text()
    assert "capture" in text and "cores" in text
    assert "sk-FAKE" not in text


_real_stats_init = app.StatsLog.__init__


def _quick_stats_log(self, path, pipeline, parts):
    _real_stats_init(self, path, pipeline, parts, every=0.1)


def test_repeated_frame_is_not_resized_again(monkeypatch):
    calls = []

    class Camera:
        device = "fake"

        def __init__(self, **kwargs):
            pass

        def send(self, image):
            pass

        def sleep_until_next_frame(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("eclipse.output.virtual_cam.pyvirtualcam.Camera", Camera)
    monkeypatch.setattr(
        "eclipse.output.virtual_cam.fit", lambda image, w, h: calls.append(1) or image
    )
    from eclipse.output.virtual_cam import VirtualCam

    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = first.copy()
    with VirtualCam() as cam:
        cam.send(first)
        cam.send(first)
        cam.send(first)
        cam.send(second)
    assert len(calls) == 2
    assert cam.stats["sends"] == 4


def test_apple_mode_scans_retina_frames_with_doubled_distances(monkeypatch):
    captured = {}
    detectors = {}

    class RecordingCompositor(app.Compositor):
        def __init__(self, padding):
            captured["padding"] = padding
            super().__init__(padding)

    def screen(max_width, retina, first_id=0):
        captured.update(max_width=max_width, retina=retina)
        return TestPattern(320, 240)

    def text_detector(reader=None, pixel_scale=1):
        detectors.update(reader=reader, pixel_scale=pixel_scale)
        return lambda frame: []

    monkeypatch.setattr(app, "screen_capture_allowed", lambda: True)
    monkeypatch.setattr(app, "screen_capture_kit_available", lambda: False)
    monkeypatch.setattr(app, "ScreenCapture", screen)
    monkeypatch.setattr(app, "TextDetector", text_detector)
    monkeypatch.setattr(app, "AppleVisionReader", lambda: "apple reader")
    monkeypatch.setattr(app, "Compositor", RecordingCompositor)
    monkeypatch.setattr(app, "VirtualCam", FakeSink)
    assert (
        app.main(["--ocr", "apple", "--ocr-workers", "0", "--max-frames", "2", "--delay", "0.05"])
        == 0
    )
    assert captured["retina"] is True
    assert captured["max_width"] > 3024
    assert captured["padding"] == app.PADDING * 2
    assert detectors == {"reader": "apple reader", "pixel_scale": 2}


def test_rapid_is_still_the_default(monkeypatch):
    used = {}

    def text_detector(reader=None, pixel_scale=1):
        used.update(reader=reader, pixel_scale=pixel_scale)
        return lambda frame: []

    monkeypatch.setattr(app, "screen_capture_allowed", lambda: True)
    monkeypatch.setattr(app, "screen_capture_kit_available", lambda: False)
    monkeypatch.setattr(
        app, "ScreenCapture", lambda max_width, retina, first_id=0: TestPattern(320, 240)
    )
    monkeypatch.setattr(app, "TextDetector", text_detector)
    monkeypatch.setattr(app, "VirtualCam", FakeSink)
    assert app.main(["--max-frames", "2", "--delay", "0.05"]) == 0
    assert used == {"reader": None, "pixel_scale": 1}


def test_to_bgr_keeps_odd_retina_sizes_even_without_a_resize():
    raw = np.zeros((101, 201, 4), dtype=np.uint8)
    raw[:, :] = (1, 2, 3, 255)
    image = to_bgr(raw, max_width=6016)
    assert image.shape == (100, 200, 3)
    assert tuple(image[0, 0]) == (1, 2, 3)


def test_windows_mode_uses_native_reader_without_changing_rapid_default(monkeypatch, tmp_path):
    reader = object()
    seen = []
    monkeypatch.setattr(app, 'WindowsOCRReader', lambda: reader)
    monkeypatch.setattr(app, 'screen_capture_allowed', lambda: True)
    monkeypatch.setattr(app, 'screen_capture_kit_available', lambda: False)
    monkeypatch.setattr(app, 'ScreenCapture', lambda **kwargs: TestPattern(320, 240))
    monkeypatch.setattr(app, 'VirtualCam', FakeSink)

    def detector(**kwargs):
        seen.append(kwargs)
        return lambda frame: []

    monkeypatch.setattr(app, 'TextDetector', detector)
    for index, extra in enumerate([['--ocr', 'windows', '--ocr-workers', '0'], []]):
        assert app.main(extra + ['--max-frames', '3', '--delay', '0.05',
                                 '--log', str(tmp_path / f'run-{index}.log')]) == 0
    assert seen == [{'reader': reader}, {}]
