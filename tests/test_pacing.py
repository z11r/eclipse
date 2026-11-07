"""A full buffer must slow capture down, never speed it up."""

import threading
import time

import numpy as np

from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.types import Frame

WIDTH, HEIGHT = 80, 60
FRAME_BYTES = WIDTH * HEIGHT * 3


class Source:
    """Remembers when it was read. Like the screen, each read costs something."""

    paced = False

    def __init__(self):
        self.next_id = 0
        self.times = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        now = time.monotonic()
        self.times.append(now)
        frame = Frame(np.full((HEIGHT, WIDTH, 3), 150, np.uint8), now, "screen", self.next_id)
        self.next_id += 1
        return frame


class PacedSource(Source):
    """Keeps its own time, like streaming capture: it can't be asked to skip a frame."""

    paced = True

    def __init__(self, interval):
        super().__init__()
        self.interval = interval

    def read(self):
        time.sleep(self.interval)
        return super().read()


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


def small_buffer(source, detect=lambda frame: [], fps=50, max_frames=30, frames_of_memory=2):
    return DelayedPipeline(
        source, Sink(), detect, Compositor(), delay=0.08, fps=fps, max_frames=max_frames,
        max_buffer_bytes=FRAME_BYTES * frames_of_memory,
    )


def test_a_full_buffer_does_not_turn_capture_into_a_busy_loop():
    source = Source()
    stats = small_buffer(source).run()
    assert stats.over_limit > 0
    # Thirty grabs at 50 Hz can't take less than 29 intervals, and with a two-frame
    # budget they take longer. It used to be about 0.04 s.
    assert source.times[-1] - source.times[0] >= 0.5


def test_grabs_stay_on_the_capture_clock_when_full():
    source = Source()
    small_buffer(source).run()
    gaps = np.diff(source.times)
    # The clock catches up after a late tick, so one gap can run short of 20 ms. A busy
    # loop's gaps are under a millisecond.
    assert gaps.min() >= 0.008
    assert np.median(gaps) >= 0.018


def test_a_tick_with_no_room_still_waits_for_the_next_tick():
    # Spinning without reading is still a busy loop, and would burn a core.
    source = Source()
    started = time.monotonic()
    stats = small_buffer(source).run()
    ticks_that_fit = (time.monotonic() - started) * 50
    assert stats.capture_skips > 0
    assert stats.capture_attempts <= ticks_that_fit * 1.2 + 2


def test_a_tick_with_no_room_does_not_read_the_source_at_all():
    source = Source()
    stats = small_buffer(source).run()
    assert stats.capture_skips > 0
    assert len(source.times) == stats.captured
    assert stats.discarded == 0


def test_capture_counts_add_up():
    source = Source()
    stats = small_buffer(source).run()
    assert stats.capture_attempts == stats.captured + stats.capture_skips
    assert stats.captured == stats.admitted + stats.discarded + stats.shed
    assert stats.over_limit == stats.capture_skips + stats.discarded


def test_memory_in_flight_never_passes_the_budget():
    stats = small_buffer(Source()).run()
    assert 0 < stats.peak_in_flight_bytes <= 2 * FRAME_BYTES
    assert stats.in_flight_bytes == 0
    assert stats.in_flight_frames == 0


def test_a_slow_scanner_makes_capture_wait_not_spin():
    source = Source()

    def slow(frame):
        time.sleep(0.05)
        return []

    pipeline = small_buffer(source, slow, fps=100, max_frames=12)
    stats = pipeline.run()
    assert stats.capture_skips > 0
    assert np.diff(source.times).min() >= 0.008
    assert stats.peak_in_flight_bytes <= 2 * FRAME_BYTES
    assert len(pipeline._sink.images) > 0


def test_overload_keeps_showing_what_was_scanned_in_order():
    source = Source()

    def slow(frame):
        time.sleep(0.03)
        return []

    pipeline = small_buffer(source, slow, fps=100, max_frames=10)
    stats = pipeline.run()
    assert stats.shown + stats.skipped + stats.dropped == stats.captured


def test_a_source_with_its_own_timing_is_still_read_when_full():
    source = PacedSource(0.01)
    stats = small_buffer(source, fps=100, max_frames=20).run()
    assert stats.capture_skips == 0
    assert stats.discarded > 0
    assert stats.captured == stats.admitted + stats.discarded + stats.shed
    assert stats.peak_in_flight_bytes <= 2 * FRAME_BYTES


def test_a_read_that_finds_nothing_new_is_paced_too():
    class Idle(Source):
        def read(self):
            super().read()

    source = Idle()
    pipeline = small_buffer(source, fps=100, max_frames=None)
    stopper = threading.Timer(0.2, pipeline._stop.set)
    stopper.start()
    pipeline.run()
    assert 5 <= len(source.times) <= 30  # about 20 reads in 0.2 s, not thousands


def test_stopping_does_not_wait_out_a_long_capture_interval():
    source = Source()
    pipeline = small_buffer(source, fps=1, max_frames=None)
    threading.Timer(0.05, pipeline._stop.set).start()
    started = time.monotonic()
    pipeline.run()
    assert time.monotonic() - started < 0.5
