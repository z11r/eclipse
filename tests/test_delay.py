import threading
import time

import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.types import Detection, Frame

SECRET_COLOUR = 123


class StampedSource:
    """Frames whose top-left pixel holds their id, with a fake secret drawn on them."""

    def __init__(self):
        self.next_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        image[0, 0] = (self.next_id // 256, self.next_id % 256, 0)
        image[20:30, 20:60] = SECRET_COLOUR
        frame = Frame(image, time.monotonic(), "screen", self.next_id)
        self.next_id += 1
        return frame


def frame_id(image):
    return int(image[0, 0, 0]) * 256 + int(image[0, 0, 1])


class Sink:
    def __init__(self, tick=0.002):
        self.images = []
        self.times = []
        self.tick = tick

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        self.images.append(image)
        self.times.append(time.monotonic())
        time.sleep(self.tick)


def finds_secret(frame):
    return [Detection(20, 20, 40, 10, "api_key", 1.0, 100, frame.frame_id)]


def run(detect=finds_secret, delay=0.05, fps=200, max_frames=40, sink=None):
    sink = sink or Sink()
    pipeline = DelayedPipeline(
        StampedSource(), sink, detect, Compositor(padding=0), delay, fps, max_frames
    )
    return pipeline.run(), sink


def no_secret_in(images):
    return all(not (image == SECRET_COLOUR).any() for image in images)


def test_every_frame_is_scanned_and_shown_when_scanning_keeps_up():
    stats, sink = run()
    assert stats.captured == 40
    assert stats.scanned == 40
    assert stats.dropped == 0
    assert stats.shown + stats.skipped == 40
    assert no_secret_in(sink.images)


def test_frames_stay_in_order():
    _, sink = run()
    ids = [frame_id(image) for image in sink.images[:-1]]  # the last one is the final black
    assert ids == sorted(ids)
    assert not sink.images[-1].any()


def test_output_waits_for_the_delay():
    started = time.monotonic()
    _, sink = run(delay=0.1, fps=100, max_frames=5)
    assert sink.times[0] - started >= 0.09


def test_last_frames_wait_for_the_delay_too():
    stats, _ = run(delay=0.1, fps=100, max_frames=5)
    assert min(stats.latencies) >= 90


def test_one_slow_scan_does_not_drop_the_quick_frames_after_it():
    def spike(frame):
        if frame.frame_id % 20 == 5:
            time.sleep(0.08)
        return finds_secret(frame)

    stats, sink = run(spike, delay=0.2, fps=100, max_frames=60)
    assert stats.dropped == 0
    assert stats.shown + stats.skipped == 60
    assert no_secret_in(sink.images)


def test_only_scanned_frames_reach_the_output():
    scanned = set()

    def detect(frame):
        scanned.add(frame.frame_id)
        return finds_secret(frame)

    _, sink = run(detect)
    assert {frame_id(image) for image in sink.images} <= scanned


def test_scanner_slower_than_capture_never_shows_a_raw_frame():
    def slow(frame):
        time.sleep(0.03)
        return finds_secret(frame)

    stats, sink = run(slow, delay=0.05, max_frames=30)
    assert stats.dropped > 0
    assert stats.scanned + stats.dropped == stats.captured
    assert no_secret_in(sink.images)


def test_a_slow_scan_is_absorbed_by_the_delay():
    def spike(frame):
        if frame.frame_id == 5:
            time.sleep(0.05)
        return finds_secret(frame)

    stats, sink = run(spike, delay=0.2, fps=100, max_frames=30)
    assert stats.dropped == 0
    assert stats.shown + stats.skipped == 30
    assert no_secret_in(sink.images)


def test_detector_crash_holds_the_last_safe_frame():
    def crashes_later(frame):
        if frame.frame_id >= 10:
            raise RuntimeError("ocr died")
        return finds_secret(frame)

    pipeline = DelayedPipeline(
        StampedSource(), Sink(), crashes_later, Compositor(padding=0), 0.05, 200, 30
    )
    stats = pipeline.run()
    assert stats.shown + stats.skipped == 30
    assert no_secret_in(pipeline._sink.images)
    assert max(frame_id(image) for image in pipeline._sink.images) == 9


def test_detector_crashing_from_the_start_sends_only_black():
    def crashes(frame):
        raise RuntimeError

    _, sink = run(crashes)
    assert sink.images
    assert all(not image.any() for image in sink.images)


def test_capture_failure_stops_and_reports_it():
    class Broken(StampedSource):
        def read(self):
            raise OSError("capture died")

    pipeline = DelayedPipeline(Broken(), Sink(), finds_secret, Compositor(), 0.01, 100, 5)
    with pytest.raises(OSError):
        pipeline.run()


def test_no_threads_left_running():
    before = threading.active_count()
    run(max_frames=5)
    time.sleep(0.05)
    assert threading.active_count() == before


def test_stops_cleanly_on_interrupt():
    class Interrupting(Sink):
        def send(self, image):
            super().send(image)
            if len(self.images) == 3:
                raise KeyboardInterrupt

    stats, sink = run(max_frames=None, sink=Interrupting())
    assert len(sink.images) == 4  # the three before the interrupt, then the final black
    assert not sink.images[-1].any()
    assert stats.final_black_sent
    assert stats.captured >= 3


def test_shown_counts_only_frames_really_sent():
    class SlowSink(Sink):
        def send(self, image):
            super().send(image)
            time.sleep(0.03)

    sink = SlowSink()
    stats = DelayedPipeline(
        StampedSource(), sink, finds_secret, Compositor(padding=0), 0.05, 200, 40
    ).run()
    distinct = len({frame_id(image) for image in sink.images})
    assert stats.shown == distinct
    assert stats.shown + stats.skipped == stats.scanned
    assert stats.skipped > 0


def test_buffer_never_holds_more_than_the_limit_allows():
    frame_bytes = 60 * 80 * 3

    def slow(frame):
        time.sleep(0.02)
        return finds_secret(frame)

    # A full buffer now waits instead of racing through frames, so sixty of them take
    # as long as the delay lets them. A shorter delay keeps this test quick.
    pipeline = DelayedPipeline(
        StampedSource(), Sink(), slow, Compositor(padding=0), 0.1, 500, 60,
        max_buffer_bytes=frame_bytes * 5,
    )
    stats = pipeline.run()
    assert stats.over_limit > 0
    assert stats.max_backlog <= 5


def test_buffer_is_sized_for_the_delay_at_60_fps():
    from eclipse.compositor.delay import _BufferLimit

    limit = _BufferLimit(1.0, 60, 6 * 1024**3)
    assert limit.frames >= 90
    limit.note_frame(3024 * 1964 * 3)
    assert limit.frames >= 90


def test_scan_of_a_dropped_frame_is_never_applied_to_another():
    seen = []

    def detect(frame):
        seen.append(frame.frame_id)
        return finds_secret(frame)

    stats, sink = run(detect, delay=0.02, fps=300, max_frames=40)
    assert sorted(seen) == seen
    assert no_secret_in(sink.images)


def test_jitter_does_not_waste_frames_when_output_keeps_pace():
    class Slightly(Sink):
        def send(self, image):
            super().send(image)
            time.sleep(0.004)

    stats, sink = run(delay=0.05, fps=100, max_frames=60, sink=Slightly())
    assert stats.shown == 60
    assert stats.skipped == 0


def test_output_that_truly_falls_behind_catches_up():
    class Slow(Sink):
        def send(self, image):
            super().send(image)
            time.sleep(0.03)

    stats, sink = run(delay=0.02, fps=200, max_frames=60, sink=Slow())
    assert stats.skipped > 0
    assert stats.shown + stats.skipped + stats.dropped == 60  # every frame is accounted for


INCOMING_COLOUR = 77  # rows a detector would hold black as they scroll in


class IncomingSource(StampedSource):
    def read(self):
        frame = super().read()
        frame.image[40:50, 10:70] = INCOMING_COLOUR
        return frame


class LookAheadDetector:
    """Reports the incoming rows as unread at first. Later, settled() reports them read,
    with the secret still covered, as a detector that looked ahead would."""

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.settled_ids = []

    def __call__(self, frame):
        if frame.frame_id == self.fail_on:
            raise RuntimeError("scan failed")
        return [*finds_secret(frame), Detection(10, 40, 60, 10, "unread", 1.0, 0, frame.frame_id)]

    def settled(self, frame):
        self.settled_ids.append(frame.frame_id)
        return finds_secret(frame)


def run_ahead(detect, look_ahead=0.03, delay=0.1, fps=100, max_frames=20):
    sink = Sink()
    pipeline = DelayedPipeline(
        IncomingSource(), sink, detect, Compositor(padding=0), delay, fps, max_frames,
        look_ahead=look_ahead,
    )
    return pipeline.run(), sink


def shows_incoming(image):
    return bool((image == INCOMING_COLOUR).any())


def test_looking_ahead_sends_what_later_frames_settled():
    detect = LookAheadDetector()
    stats, sink = run_ahead(detect)
    sent = sink.images[:-1]  # the last is the final black
    assert stats.scanned == 20
    assert all(shows_incoming(image) for image in sent)  # no black strip
    assert no_secret_in(sink.images)
    ids = [frame_id(image) for image in sent]
    assert ids == sorted(ids)
    assert detect.settled_ids == sorted(detect.settled_ids)


def test_looking_ahead_waits_within_the_delay():
    stats, sink = run_ahead(LookAheadDetector(), look_ahead=0.03, delay=0.1)
    assert min(stats.latencies) >= 90
    assert min(stats.slack) > 0  # every frame was drawn before it was due


def test_a_failed_scan_sends_the_frames_waiting_with_what_they_first_reported():
    stats, sink = run_ahead(LookAheadDetector(fail_on=10), look_ahead=0.05)
    waited = [image for image in sink.images[:-1] if frame_id(image) < 10]
    assert waited and no_secret_in(sink.images)
    # Frames still waiting when the scan failed go out with their black strip.
    assert any(not shows_incoming(image) for image in waited)
    assert stats.held >= 1


def test_without_settled_the_detector_is_used_as_before():
    stats, sink = run_ahead(lambda frame: LookAheadDetector()(frame))
    assert not any(shows_incoming(image) for image in sink.images)


def test_looking_ahead_must_leave_part_of_the_delay():
    with pytest.raises(ValueError):
        DelayedPipeline(
            StampedSource(), Sink(), finds_secret, Compositor(), 0.1, 30, 1, look_ahead=0.1
        )
