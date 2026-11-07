"""The numbers must mean what they say, and stay the same size however long a run is."""

import sys
import time
from collections import deque

import numpy as np
import pytest

from eclipse import main as app
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import _SAMPLES_KEPT, DelayedPipeline, PipelineStats
from eclipse.telemetry import Distribution
from eclipse.types import Detection, Frame


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Stage:
    def __init__(self, **stats):
        self.stats = stats


class FakePipeline:
    def __init__(self):
        self.stats = PipelineStats()


def log_with(clock, parts=None, pipeline=None):
    return app.StatsLog("unused", pipeline or FakePipeline(), parts or {}, every=2.0, clock=clock)


def field(line, name):
    """The word after a name in a log line."""
    words = line.split()
    return words[words.index(name) + 1]


def test_detector_timing_cannot_replace_the_pipeline_total():
    pipeline = FakePipeline()
    pipeline.stats.scan_seconds = 2.0
    log = log_with(Clock(), {"detect": Stage(scan_seconds=0.5)}, pipeline)
    numbers = log._snapshot()
    assert numbers["pipeline.scan_seconds"] == 2.0
    assert numbers["detect.scan_seconds"] == 0.5


@pytest.mark.parametrize("interval", [0.5, 2.0, 5.0, 17.0])
def test_rates_use_the_time_that_really_passed(interval):
    clock = Clock()
    pipeline = FakePipeline()
    log = log_with(clock, pipeline=pipeline)
    before = log._snapshot()
    clock.now += interval
    pipeline.stats.captured = 300
    line = log._line(before, log._snapshot())
    assert field(line, "capture") == f"{300 / interval:.1f}fps"


def test_busy_shares_use_the_time_that_really_passed():
    clock = Clock()
    pipeline = FakePipeline()
    log = log_with(clock, pipeline=pipeline)
    before = log._snapshot()
    clock.now += 4.0
    pipeline.stats.scan_seconds = 1.0
    assert field(log._line(before, log._snapshot()), "scan") == "25%"


def test_each_stage_keeps_its_own_timing():
    clock = Clock()
    pipeline = FakePipeline()
    detect = Stage(
        copy_seconds=0.0, diff_seconds=0.0, scan_seconds=0.0, find_seconds=0.0, track_seconds=0.0
    )
    capture = Stage(grab_seconds=0.0, convert_seconds=0.0)
    output = Stage(fit_seconds=0.0, send_seconds=0.0, fits=0, sends=0)
    log = log_with(clock, {"detect": detect, "capture": capture, "output": output}, pipeline)
    before = log._snapshot()
    clock.now += 1
    pipeline.stats.scanned = 10
    pipeline.stats.captured = 10
    pipeline.stats.detect_seconds = 1.0
    pipeline.stats.composite_seconds = 0.08
    detect.stats.update(
        copy_seconds=0.01, diff_seconds=0.02, scan_seconds=0.3, find_seconds=0.04,
        track_seconds=0.05,
    )
    capture.stats.update(grab_seconds=0.06, convert_seconds=0.07)
    output.stats.update(fit_seconds=0.1, send_seconds=0.2, fits=10, sends=20)
    line = log._line(before, log._snapshot())
    assert field(line, "copy") == "1.0"
    assert field(line, "diff") == "2.0"
    assert field(line, "ocr") == "30.0"
    assert field(line, "rules") == "4.0"
    assert field(line, "track") == "5.0"
    assert field(line, "other") == "58.0"
    assert field(line, "composite-ms") == "8.0"
    assert field(line, "grab") == "6.0"
    assert field(line, "convert") == "7.0"
    assert field(line, "fit") == "10.0"
    assert field(line, "send") == "10.0"


def test_what_the_pipeline_did_is_told_apart():
    clock = Clock()
    pipeline = FakePipeline()
    log = log_with(clock, {"detect": Stage(full_scans=0, partial_scans=0)}, pipeline)
    before = log._snapshot()
    clock.now += 1
    stats = pipeline.stats
    stats.rendered_scanned, stats.rendered_cached = 3, 5
    stats.rendered_held, stats.rendered_blackout = 7, 11
    stats.sends, stats.black_sends, stats.distinct_presented = 40, 2, 8
    line = log._line(before, log._snapshot())
    assert "results read 3 cached 5 held 7 blackout 11" in line
    assert "camera-sends 40 black 2 new-pictures 8" in line


def test_a_long_hold_shows_up_as_the_age_of_the_last_new_picture():
    clock = Clock()
    pipeline = FakePipeline()
    log = log_with(clock, pipeline=pipeline)
    pipeline.stats.last_presented_at = clock.now
    before = log._snapshot()
    clock.now += 30
    assert field(log._line(before, log._snapshot()), "protected-age-ms") == "30000"
    clock.now += 30
    assert field(log._line(before, log._snapshot()), "protected-age-ms") == "60000"


def test_age_is_not_a_made_up_number_before_anything_has_gone_out():
    clock = Clock()
    log = log_with(clock)
    before = log._snapshot()
    clock.now += 1
    assert field(log._line(before, log._snapshot()), "protected-age-ms") == "-1"


class Source:
    def __init__(self):
        self.next_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        image = np.full((40, 40, 3), self.next_id % 200, np.uint8)
        frame = Frame(image, time.monotonic(), "screen", self.next_id)
        self.next_id += 1
        return frame


class Sink:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        time.sleep(0.001)


def run(detect=lambda frame: [], max_frames=30):
    pipeline = DelayedPipeline(
        Source(), Sink(), detect, Compositor(), delay=0.02, fps=500, max_frames=max_frames
    )
    return pipeline.run()


def test_a_held_image_is_not_a_new_picture():
    def fails_after_the_first(frame):
        if frame.frame_id:
            raise RuntimeError("ocr died")
        return []

    stats = run(fails_after_the_first)
    assert stats.distinct_presented == 1
    assert stats.rendered_held == 29
    assert stats.shown == 30  # every result went out, but viewers saw one picture
    assert stats.max_frozen_seconds > 0


def test_a_detector_that_reuses_its_reads_is_counted_as_cached():
    class Reuses:
        reused_read = True

        def __call__(self, frame):
            return []

    stats = run(Reuses())
    assert stats.rendered_cached == 30
    assert stats.rendered_scanned == 0


def test_a_scan_of_new_pixels_is_counted_as_scanned():
    stats = run()
    assert stats.rendered_scanned == 30
    assert stats.distinct_presented == 30


def test_a_startup_blackout_is_counted_and_is_not_a_picture():
    def fails(frame):
        raise RuntimeError("ocr died")

    stats = run(fails)
    assert stats.rendered_blackout == 30
    assert stats.distinct_presented == 0
    assert stats.latency.count == 0


def test_samples_stay_bounded_however_long_a_run_is():
    stats = PipelineStats()
    for name in ("latencies", "scan_ages", "slack", "present_intervals"):
        samples = getattr(stats, name)
        assert isinstance(samples, deque)
        for value in range(_SAMPLES_KEPT * 3):
            samples.append(value)
        assert len(samples) == _SAMPLES_KEPT


def test_the_distribution_does_not_grow_with_what_it_has_seen():
    few, many = Distribution(), Distribution()
    for value in range(10):
        few.add(value)
    for value in range(200_000):
        many.add(value % 5000)
    assert sys.getsizeof(few._bins) == sys.getsizeof(many._bins)


@pytest.mark.parametrize("share", [0.5, 0.95, 0.99])
def test_the_distribution_tracks_real_percentiles_closely(share):
    rng = np.random.default_rng(5)
    values = rng.lognormal(3, 1, 50_000)
    distribution = Distribution()
    for value in values:
        distribution.add(float(value))
    exact = float(np.quantile(values, share))
    assert abs(distribution.percentile(share) - exact) / exact < 0.05


def test_the_distribution_copes_with_nothing_and_with_extremes():
    empty = Distribution()
    assert empty.percentile(0.95) == 0.0
    extremes = Distribution()
    extremes.add(0.0)
    extremes.add(10_000_000.0)
    assert extremes.percentile(1.0) == 10_000_000.0


def test_nothing_about_a_detection_reaches_the_log(tmp_path):
    def finds_a_secret(frame):
        return [Detection(1, 1, 5, 5, "api_key", 1.0, 100, frame.frame_id)]

    pipeline = DelayedPipeline(Source(), Sink(), finds_a_secret, Compositor(), 0.02, 500, 10)
    with app.StatsLog(str(tmp_path / "run.log"), pipeline, {}, every=0.01):
        pipeline.run()
    text = (tmp_path / "run.log").read_text()
    assert "capture" in text
    assert "api_key" not in text
