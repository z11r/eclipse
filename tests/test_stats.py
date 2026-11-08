import time

import numpy as np
import pytest

from eclipse import main as app
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.output import virtual_cam
from eclipse.types import Frame


class Source:
    """Yields the given pictures in order, with the given frame ids."""

    def __init__(self, pictures, ids=None):
        self.pictures = pictures
        self.ids = ids or list(range(len(pictures)))
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        picture = self.pictures[min(self.index, len(self.pictures) - 1)]
        frame_id = self.ids[min(self.index, len(self.ids) - 1)]
        self.index += 1
        return Frame(picture, time.monotonic(), "screen", frame_id)


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


def picture(level):
    return np.full((64, 64, 3), level, dtype=np.uint8)


def run(pictures, detect=lambda frame: [], ids=None, fps=200):
    pipeline = DelayedPipeline(
        Source(pictures, ids), Sink(), detect, Compositor(padding=0), 0.02, fps, len(pictures)
    )
    return pipeline, pipeline.run()


def test_repeated_pictures_are_captured_but_not_fresh():
    a, b = picture(10), picture(20)
    _, stats = run([a, a, a, b, b, a])
    assert stats.captured == 6
    assert stats.fresh == 3  # the first, the change to b, the change back to a
    assert stats.fresh_shown == 3


def test_a_change_in_one_sampled_row_counts_as_fresh():
    a, b = picture(10), picture(10)
    b[32, 5] = 200
    _, stats = run([a, b])
    assert stats.fresh == 2


def test_missing_source_ids_are_counted_at_the_output():
    pictures = [picture(i) for i in range(4)]
    _, stats = run(pictures, ids=[10, 11, 14, 15])
    assert stats.id_gaps == 2
    assert stats.last_sent_id == 15


def test_detect_time_is_kept_apart_from_compositing():
    def slow_detect(frame):
        time.sleep(0.01)
        return []

    _, stats = run([picture(1), picture(2), picture(3)], detect=slow_detect)
    assert stats.detect_seconds >= 0.03
    assert stats.composite_seconds < stats.detect_seconds
    assert abs(stats.detect_seconds + stats.composite_seconds - stats.scan_seconds) < 1e-6


def test_queue_age_slack_and_presentation_are_sampled():
    _, stats = run([picture(i) for i in range(5)])
    assert len(stats.scan_ages) == 5
    assert all(age >= 0 for age in stats.scan_ages)
    assert len(stats.slack) == 5
    assert len(stats.present_intervals) == stats.shown - 1


def test_a_stage_cannot_overwrite_another_stages_counter():
    class Detect:
        stats = {"scan_seconds": 0.25, "diff_seconds": 0.5}

        def __call__(self, frame):
            return []

    class Capture:
        stats = {"grab_seconds": 1.5}

    pipeline, stats = run([picture(1)], detect=Detect())
    stats.scan_seconds = 9.0
    stats.grab_seconds = 4.0
    log = app.StatsLog("unused", pipeline, {"capture": Capture(), "detect": Detect()})
    numbers = log._snapshot()
    assert numbers["pipeline.scan_seconds"] == 9.0
    assert numbers["detect.scan_seconds"] == 0.25
    assert numbers["pipeline.grab_seconds"] == 4.0
    assert numbers["capture.grab_seconds"] == 1.5


def test_log_line_reports_each_stage_on_its_own(tmp_path):
    pipeline, stats = run([picture(i) for i in range(4)])
    log = app.StatsLog(str(tmp_path / "run.log"), pipeline, {})
    before = {name: 0 for name in log._snapshot()}
    line = log._line(before, log._snapshot())
    for word in (
        "fresh-sent",
        "repeats",
        "shed on arrival",
        "id-gaps",
        "present-ms",
        "queue-ms",
        "slack-min",
        "capture-ms",
        "convert",
        "detect-ms",
        "diff",
        "ocr",
        "rules",
        "composite-ms",
        "output-ms",
        "fit",
        "send",
    ):
        assert word in line


def test_percentile_handles_no_samples():
    assert app._percentile([], 0.95) == 0.0
    assert app._percentile([1.0, 2.0, 3.0], 1) == 3.0


class FakeCamera:
    """Records when each frame left and which tick it was stamped with."""

    device = "fake"

    def __init__(self, *args):
        self.times = []
        self.dues = []

    def prepare(self, image):
        return image

    def send(self, image, due):
        self.times.append(virtual_cam.time.monotonic())
        self.dues.append(due)

    def close(self):
        pass


class FakeClock:
    """Time that only moves when told, so the test can't depend on how busy the machine is."""

    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_new_and_repeated_frames_leave_on_the_same_ticks(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(virtual_cam, "time", clock)
    monkeypatch.setattr(virtual_cam, "open_camera", FakeCamera)

    def slow_fit(image, width, height):
        clock.now += 0.008
        return image

    monkeypatch.setattr(virtual_cam, "fit", slow_fit)
    cam = virtual_cam.VirtualCam(fps=60)
    with cam:
        for index in range(30):
            cam.send(picture(index % 200))  # a new object each time means a fit each time
            cam.send(cam._last_image)  # a repeat costs nothing
        gaps = np.diff(cam._camera.times)
    assert gaps == pytest.approx(1 / 60)
    assert cam.stats["late_ticks"] == 0


def test_output_running_late_is_counted_and_does_not_burst_to_catch_up(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(virtual_cam, "time", clock)
    monkeypatch.setattr(virtual_cam, "open_camera", FakeCamera)
    fits = iter([0.0, 0.1, 0.0, 0.0])

    def uneven_fit(image, width, height):
        clock.now += next(fits)
        return image

    monkeypatch.setattr(virtual_cam, "fit", uneven_fit)
    cam = virtual_cam.VirtualCam(fps=60)
    with cam:
        for index in range(4):
            cam.send(picture(index))
        gaps = np.diff(cam._camera.times)
    assert cam.stats["late_ticks"] == 1
    assert min(gaps) >= 1 / 60 - 1e-9  # never two frames at once after a stall


def test_frames_are_stamped_with_their_tick_not_with_when_they_left(monkeypatch):
    """OBS paces by the stamp, so a late wake-up must not move it."""
    clock = FakeClock()
    monkeypatch.setattr(virtual_cam, "time", clock)
    monkeypatch.setattr(virtual_cam, "open_camera", FakeCamera)
    wake_ups = iter([0.0, 0.004, 0.0, 0.012, 0.001, 0.0])

    def oversleep(seconds):
        clock.now += seconds + next(wake_ups)

    clock.sleep = oversleep
    cam = virtual_cam.VirtualCam(fps=60)
    with cam:
        for index in range(6):
            cam.send(picture(index))
    assert np.diff(cam._camera.dues) == pytest.approx(1 / 60)
    assert max(np.array(cam._camera.times) - cam._camera.dues) == pytest.approx(0.012)


def test_a_stall_keeps_the_stamps_on_one_grid(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(virtual_cam, "time", clock)
    monkeypatch.setattr(virtual_cam, "open_camera", FakeCamera)
    fits = iter([0.0, 0.0, 0.07, 0.0, 0.0])

    def uneven_fit(image, width, height):
        clock.now += next(fits)
        return image

    monkeypatch.setattr(virtual_cam, "fit", uneven_fit)
    cam = virtual_cam.VirtualCam(fps=60)
    with cam:
        for index in range(5):
            cam.send(picture(index))
    ticks = (np.array(cam._camera.dues) - cam._camera.dues[0]) * 60
    assert ticks == pytest.approx(np.round(ticks), abs=1e-6)  # whole ticks apart
    assert list(np.round(np.diff(ticks))) == [1, 5, 1, 1]  # the stall skips ticks, never shifts them
    assert all(sent >= due for sent, due in zip(cam._camera.times, cam._camera.dues))


def test_a_lagging_scan_sheds_new_frames_instead_of_scanning_stale_ones():
    """Frames are dropped on arrival, so none waits past its deadline to be dropped late."""

    class Source:
        paced = False

        def __init__(self):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            self.count += 1
            return Frame(picture(self.count % 250), time.monotonic(), "screen", self.count)

    scanned = []

    def slow_detect(frame):
        scanned.append(frame.frame_id)
        time.sleep(0.02)  # can only keep up with 50 frames a second
        return []

    pipeline = DelayedPipeline(
        Source(), Sink(), slow_detect, Compositor(padding=0), 0.2, 100, 150
    )
    stats = pipeline.run()
    assert stats.shed > 0
    assert stats.dropped - stats.shed <= 2  # almost nothing waited until it was too late
    assert stats.scanned + stats.dropped == stats.captured  # every frame accounted for
    assert scanned == sorted(scanned)
