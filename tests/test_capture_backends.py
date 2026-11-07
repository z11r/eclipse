import sys
import threading
import time

import numpy as np
import pytest

from eclipse import main as app
from eclipse.capture.preferred import PreferredCapture
from eclipse.capture.pattern import TestPattern
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.types import Frame

needs_mac = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")


class Backend:
    def __init__(self, fail_on_enter=False, fail_on_read=False, paced=True, first_id=0):
        self.fail_on_enter = fail_on_enter
        self.fail_on_read = fail_on_read
        self.paced = paced
        self.first_id = first_id
        self.next_id = 7
        self.closed = False
        self.stats = {"grab_seconds": 0.0}

    def __enter__(self):
        if self.fail_on_enter:
            raise RuntimeError("no permission")
        return self

    def __exit__(self, *exc):
        self.closed = True

    def read(self):
        if self.fail_on_read:
            raise RuntimeError("stream died")
        return Frame(np.zeros((4, 4, 3), np.uint8), time.monotonic(), "screen", self.first_id)


def test_uses_the_preferred_backend_when_it_works():
    preferred = Backend()
    with PreferredCapture(preferred, lambda first_id: pytest.fail("no fallback needed")) as source:
        assert source.read() is not None
        assert source.paced
        assert not source.fell_back


def test_falls_back_when_the_stream_cannot_start(capsys):
    with PreferredCapture(Backend(fail_on_enter=True), lambda i: Backend(paced=False)) as source:
        assert source.fell_back
        assert not source.paced
        assert source.read() is not None
    assert "no permission" not in capsys.readouterr().out  # error text could hold screen text


def test_falls_back_mid_run_and_keeps_counting_frames():
    made = []

    def fallback(first_id):
        made.append(Backend(first_id=first_id))
        return made[-1]

    preferred = Backend(fail_on_read=True)
    with PreferredCapture(preferred, fallback) as source:
        frame = source.read()
    assert preferred.closed
    assert frame.frame_id == 7  # continues, so a gap can't look like a repeat
    assert made[0].closed


def test_a_failing_fallback_stops_capture_instead_of_hiding_it():
    with PreferredCapture(Backend(fail_on_read=True), lambda i: Backend(fail_on_read=True)) as source:
        with pytest.raises(RuntimeError):
            source.read()


def test_mss_flag_never_touches_the_streaming_backend(monkeypatch):
    monkeypatch.setattr(app, "screen_capture_kit_available", lambda: pytest.fail("not needed"))
    assert isinstance(app.screen_source("mss", 1920, False, 60), app.ScreenCapture)


def test_no_binding_means_polling(monkeypatch):
    monkeypatch.setattr(app, "screen_capture_kit_available", lambda: False)
    assert isinstance(app.screen_source("auto", 1920, False, 60), app.ScreenCapture)


class Sink:
    def __init__(self):
        self.images = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        self.images.append(image)
        time.sleep(0.002)


def test_pipeline_rides_out_a_still_screen_without_a_paced_source_blocking_shutdown():
    class Quiet:
        paced = True

        def __init__(self):
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            self.reads += 1
            time.sleep(0.005)
            if self.reads == 1:
                return Frame(np.zeros((8, 8, 3), np.uint8), time.monotonic(), "screen", 0)
            return None  # nothing new

    source, sink = Quiet(), Sink()
    pipeline = DelayedPipeline(source, sink, lambda frame: [], Compositor(padding=0), 0.01, 60)
    result = {}
    thread = threading.Thread(target=lambda: result.update(stats=pipeline.run()))
    thread.start()
    time.sleep(0.15)
    pipeline._stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result["stats"].captured == 1  # a still screen isn't new frames
    assert sink.images  # the last safe frame is still being sent


def test_a_paced_source_is_not_slowed_by_the_capture_clock():
    class Fast(TestPattern):
        __test__ = False
        paced = True

    pipeline = DelayedPipeline(
        Fast(32, 32), Sink(), lambda frame: [], Compositor(padding=0), 0.01, 1, 30
    )
    started = time.monotonic()
    pipeline.run()
    assert time.monotonic() - started < 2  # at the fallback clock of 1 fps this would take 30 s


@needs_mac
class TestScreenCaptureKitLogic:
    """What can be checked without a live stream, which needs Screen Recording."""

    @pytest.fixture(autouse=True)
    def backend(self):
        from eclipse import capture  # noqa: F401
        from eclipse.capture import sck

        self.sck = sck

    def test_even_size_shrinks_and_rounds_down(self):
        assert self.sck.even_size(3024, 1964, 6016) == (3024, 1964)
        assert self.sck.even_size(3024, 1964, 1920) == (1920, 1246)
        assert self.sck.even_size(1513, 983, 6016) == (1512, 982)

    def test_dirty_rects_become_whole_pixels_that_still_cover_the_area(self):
        assert self.sck.dirty_rects([(10.5, 20.25, 3.0, 4.0)], 100, 100) == ((10, 20, 4, 5),)

    def test_dirty_rects_are_clipped_to_the_frame(self):
        assert self.sck.dirty_rects([(90, 90, 50, 50)], 100, 100) == ((90, 90, 10, 10),)

    @pytest.mark.parametrize(
        "rects",
        [
            None,
            [],
            [(0, 0, 0, 10)],
            [(0, 0, -5, 10)],
            [(500, 500, 10, 10)],
            [(0, 0, float("nan"), 10)],
            [(0, 0, 10, 10), (0, 0, 0, 0)],
        ],
    )
    def test_any_doubtful_list_means_unknown_not_nothing_changed(self, rects):
        assert self.sck.dirty_rects(rects, 100, 100) is None

    def test_frame_ids_count_pictures_that_were_never_read(self):
        source = self.sck.ScreenCaptureKitCapture()
        image = np.zeros((4, 4, 3), np.uint8)
        for _ in range(self.sck._INBOX_LIMIT + 2):
            source.accept(image.copy(), time.monotonic(), None)
        assert source.stats["dropped"] == 2
        assert source.read().frame_id == 2  # the gap is visible to whoever reads next

    def test_copied_frames_are_read_only(self):
        source = self.sck.ScreenCaptureKitCapture()
        source.accept(np.zeros((4, 4, 3), np.uint8), time.monotonic(), None)
        assert not source.read().image.flags.writeable

    def test_a_still_screen_reads_as_nothing_new_rather_than_blocking(self):
        source = self.sck.ScreenCaptureKitCapture()
        started = time.monotonic()
        assert source.read() is None
        assert time.monotonic() - started < 1

    def test_a_stopped_stream_raises_once_queued_frames_are_read(self):
        source = self.sck.ScreenCaptureKitCapture()
        source.accept(np.zeros((4, 4, 3), np.uint8), time.monotonic(), None)
        source.on_failure(RuntimeError("stopped"))
        assert source.read() is not None
        with pytest.raises(self.sck.CaptureStopped):
            source.read()

    def test_stream_output_class_keeps_nothing_after_the_callback(self):
        # The delegate holds the owner only, so no surface can stay behind on it.
        delegate = self.sck._StreamDelegate.alloc().initWithOwner_(object())
        assert [name for name in vars(delegate) if name != "owner"] == []
