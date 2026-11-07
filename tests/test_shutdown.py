"""However a run ends, the last thing sent is black, and nothing gets out afterwards."""

import sys
import threading
import time

import numpy as np
import pytest

from eclipse import main as app
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.output import virtual_cam
from eclipse.types import Detection, Frame


class Source:
    def __init__(self):
        self.next_id = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def read(self):
        time.sleep(0.001)
        image = np.full((40, 60, 3), 120, np.uint8)
        frame = Frame(image, time.monotonic(), "screen", self.next_id)
        self.next_id += 1
        return frame


class Sink:
    """Records pictures, and refuses any that arrive after it has closed."""

    def __init__(self):
        self.images = []
        self.closed = False
        self.after_close = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def send(self, image):
        if self.closed:
            self.after_close += 1
        self.images.append(image)
        time.sleep(0.001)


def pipeline_with(detect=lambda frame: [], source=None, sink=None, **kw):
    options = dict(delay=0.02, fps=200, max_frames=20, join_seconds=0.3)
    options.update(kw)
    return DelayedPipeline(source or Source(), sink or Sink(), detect, Compositor(), **options)


def stop_soon(pipeline, after=0.05):
    timer = threading.Timer(after, pipeline._stop.set)
    timer.daemon = True
    timer.start()


def assert_ended_black(pipeline):
    sink = pipeline._sink
    assert sink.images, "nothing was sent"
    assert not sink.images[-1].any()
    assert pipeline.stats.final_black_sent is True
    assert sink.after_close == 0


def test_a_normal_exit_ends_on_black():
    pipeline = pipeline_with()
    pipeline.run()
    assert_ended_black(pipeline)


def test_ctrl_c_ends_on_black():
    class Interrupts(Sink):
        def send(self, image):
            super().send(image)
            if len(self.images) == 4:
                raise KeyboardInterrupt

    pipeline = pipeline_with(sink=Interrupts(), max_frames=None)
    pipeline.run()
    assert_ended_black(pipeline)
    assert pipeline._source.closed


def test_a_capture_failure_ends_on_black_and_is_reported():
    class Fails(Source):
        def read(self):
            if self.next_id == 10:
                raise OSError("capture died")
            return super().read()

    pipeline = pipeline_with(source=Fails(), max_frames=None, delay=0.005)
    with pytest.raises(OSError):
        pipeline.run()
    assert_ended_black(pipeline)


def test_a_detector_failure_stays_safe_and_still_ends_on_black():
    def fails_later(frame):
        if frame.frame_id > 5:
            raise RuntimeError("ocr died")
        return [Detection(5, 5, 20, 20, "api_key", 1.0, 0, frame.frame_id)]

    pipeline = pipeline_with(fails_later)
    pipeline.run()
    assert_ended_black(pipeline)
    assert pipeline.stats.rendered_held > 0


def test_a_sink_failure_is_reported_and_workers_still_stop():
    class Breaks(Sink):
        def send(self, image):
            if len(self.images) == 3:
                raise OSError("camera went away")
            super().send(image)

    pipeline = pipeline_with(sink=Breaks(), max_frames=None)
    with pytest.raises(OSError):
        pipeline.run()
    assert pipeline.output_error == "OSError"
    assert pipeline.stats.final_black_sent is False  # it couldn't, and it says so
    assert pipeline.stats.stalled_workers == ()
    assert pipeline._sink.closed


def test_a_sink_that_recovers_gets_the_final_black():
    class Hiccups(Sink):
        def __init__(self):
            super().__init__()
            self.failed = False

        def send(self, image):
            if len(self.images) == 3 and not self.failed:
                self.failed = True
                raise OSError("one bad frame")
            super().send(image)

    pipeline = pipeline_with(sink=Hiccups(), max_frames=None)
    with pytest.raises(OSError):
        pipeline.run()
    assert_ended_black(pipeline)


def test_a_stalled_worker_is_reported_and_cannot_reach_the_output():
    release = threading.Event()

    class Stalls:
        """Like a native OCR call: cancel is asked for, and can't take effect."""

        def __init__(self):
            self.calls = 0
            self.cancelled = threading.Event()

        def cancel(self):
            self.cancelled.set()

        def __call__(self, frame):
            self.calls += 1
            if self.calls == 5:
                release.wait(10)
            return []

    detect = Stalls()
    sink = Sink()
    pipeline = pipeline_with(detect, sink=sink, max_frames=None)
    stop_soon(pipeline, 0.1)
    started = time.monotonic()
    pipeline.run()
    assert time.monotonic() - started < 2
    assert detect.cancelled.is_set()
    assert pipeline.stats.stalled_workers == ("eclipse-scan",)
    assert_ended_black(pipeline)
    sent = len(sink.images)
    release.set()  # the scan comes back late, and finds the door shut
    pipeline._threads[1].join(timeout=2)
    assert not pipeline._threads[1].is_alive()
    assert len(sink.images) == sent


def test_a_stalled_worker_makes_the_exit_code_a_failure(monkeypatch, capsys):
    release = threading.Event()

    class FakeCam:
        def __init__(self, *args, **kwargs):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def send(self, image):
            self.count += 1
            time.sleep(0.01)
            if self.count == 30:
                raise KeyboardInterrupt

    def stalling(frame):
        if frame.frame_id == 1:
            release.wait(5)
        return []

    class Pattern:
        def __init__(self):
            self.next_id = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            time.sleep(0.001)
            frame = Frame(np.zeros((8, 8, 3), np.uint8), time.monotonic(), "screen", self.next_id)
            self.next_id += 1
            return frame

        detect = staticmethod(stalling)

    def quick_pipeline(*args):
        return DelayedPipeline(*args[:7], join_seconds=0.1)

    monkeypatch.setattr(app, "TestPattern", Pattern)
    monkeypatch.setattr(app, "VirtualCam", FakeCam)
    monkeypatch.setattr(app, "DelayedPipeline", quick_pipeline)
    code = app.main(["--test-pattern", "--no-panel", "--no-hotkey", "--delay", "0.01"])
    release.set()
    assert code == 1
    assert "eclipse-scan" in capsys.readouterr().out


def test_the_scanner_stops_at_once_instead_of_draining_its_queue():
    scans = []

    def slow(frame):
        scans.append(frame.frame_id)
        time.sleep(0.02)
        return []

    pipeline = pipeline_with(slow, max_frames=None, delay=5.0, fps=500)
    stop_soon(pipeline, 0.1)
    pipeline.run()
    assert len(scans) <= 8  # about five fit in 0.1 s, though dozens were waiting
    assert pipeline.stats.backlog > 5 or pipeline.stats.max_backlog > 5
    assert pipeline.stats.in_flight_frames == 0
    assert pipeline.stats.in_flight_bytes == 0


def test_once_stopped_the_scanner_takes_no_more_work_even_if_some_is_queued():
    pipeline = pipeline_with()
    frame = Frame(np.zeros((40, 60, 3), np.uint8), time.monotonic(), "screen", 0)
    pipeline._waiting.append((frame, True))
    assert pipeline._next_to_scan() is not None
    pipeline._waiting.append((frame, True))
    pipeline._stop.set()
    assert pipeline._next_to_scan() is None


def test_capture_is_closed_and_threads_are_gone_after_a_clean_stop():
    before = threading.active_count()
    pipeline = pipeline_with()
    pipeline.run()
    assert pipeline.stats.stalled_workers == ()
    assert pipeline._source.closed
    assert threading.active_count() == before


def test_a_detector_with_no_cancel_is_fine():
    pipeline = pipeline_with()
    pipeline.run()
    assert_ended_black(pipeline)


def test_a_cancel_that_fails_does_not_stop_the_shutdown():
    class BadCancel:
        def cancel(self):
            raise RuntimeError("couldn't")

        def __call__(self, frame):
            return []

    pipeline = pipeline_with(BadCancel())
    pipeline.run()
    assert_ended_black(pipeline)


def test_a_run_that_never_showed_anything_does_not_open_the_camera_just_to_close_it():
    class Fails(Source):
        def read(self):
            raise OSError("capture died")

    pipeline = pipeline_with(source=Fails())
    with pytest.raises(OSError):
        pipeline.run()
    assert pipeline._sink.images == []
    assert pipeline.stats.final_black_sent is False


def test_output_stays_closed_after_shutdown():
    pipeline = pipeline_with()
    pipeline.run()
    assert pipeline.gate.closed
    from eclipse.types import Rendered

    fresh = Rendered(np.ones((4, 4, 3), np.uint8), "scanned", "screen", 9, time.monotonic())
    assert pipeline.gate.allows(fresh) is False


def test_the_real_camera_gets_the_final_black_before_it_closes(monkeypatch):
    log = []

    class Camera:
        device = "fake"

        def __init__(self, **kwargs):
            pass

        def send(self, image):
            log.append(("black" if not image.any() else "picture", image.shape))

        def close(self):
            log.append(("closed", None))

    monkeypatch.setattr(virtual_cam.pyvirtualcam, "Camera", Camera)
    cam = virtual_cam.VirtualCam(fps=500)
    pipeline = DelayedPipeline(
        Source(), cam, lambda frame: [], Compositor(), 0.01, 200, 15, join_seconds=0.3
    )
    pipeline.run()
    assert log[-2] == ("black", (virtual_cam.OUTPUT_HEIGHT, virtual_cam.OUTPUT_WIDTH, 3))
    assert log[-1] == ("closed", None)
    assert any(kind == "picture" for kind, _ in log)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="on Windows os.kill(SIGTERM) ends the process at once, taking pytest with it",
)
def test_sigterm_is_treated_like_ctrl_c(monkeypatch):
    import os
    import signal

    class Waits(Source):
        pass

    class Cam(Sink):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def send(self, image):
            super().send(image)
            if len(self.images) == 5:
                os.kill(os.getpid(), signal.SIGTERM)

    cams = []

    def make_cam(fps):
        cams.append(Cam())
        return cams[0]

    monkeypatch.setattr(app, "VirtualCam", make_cam)
    before = signal.getsignal(signal.SIGTERM)
    code = app.main(["--test-pattern", "--no-panel", "--no-hotkey", "--delay", "0.01"])
    assert code == 0
    assert not cams[0].images[-1].any()
    assert signal.getsignal(signal.SIGTERM) == before
