import time

import numpy as np
import pytest

from eclipse.capture.pattern import TestPattern
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.types import Detection, Frame


def noise_frame(ts=0.0, frame_id=0):
    image = np.random.default_rng(frame_id).integers(0, 255, (100, 200, 3), dtype=np.uint8)
    return Frame(image, ts, "screen", frame_id)


def box(x=50, y=30, w=40, h=20, ttl_ms=500, frame_id=0):
    return Detection(x, y, w, h, "api_key", 0.9, ttl_ms, frame_id)


def is_hidden(image, x=50, y=30, w=40, h=20):
    region = image[y : y + h, x : x + w].reshape(-1, 3)
    return (region == region[0]).all()


def nothing(frame):
    return []


def crash(frame):
    raise RuntimeError("ocr died")


def test_box_is_hidden_and_the_rest_is_untouched():
    frame = noise_frame()
    output = Compositor(padding=0).process(frame, lambda f: [box()])
    assert is_hidden(output)
    untouched = np.ones(output.shape[:2], dtype=bool)
    untouched[30:50, 50:90] = False
    assert (output[untouched] == frame.image[untouched]).all()


def test_default_padding_covers_at_least_10_px_around_the_box():
    output = Compositor().process(noise_frame(), lambda f: [box()])
    assert is_hidden(output, 40, 20, 60, 40)


def test_fill_uses_none_of_the_pixels_under_it():
    dark = Frame(np.full((100, 200, 3), 10, dtype=np.uint8), 0.0, "screen", 0)
    bright = Frame(np.full((100, 200, 3), 240, dtype=np.uint8), 0.0, "screen", 0)
    dark_output = Compositor(padding=0).process(dark, lambda f: [box()])
    bright_output = Compositor(padding=0).process(bright, lambda f: [box()])
    assert (dark_output[30:50, 50:90] == bright_output[30:50, 50:90]).all()


def test_overlapping_boxes_share_one_fill_in_any_order():
    first, second = box(x=50), box(x=60)
    one_way = Compositor(padding=0).process(noise_frame(), lambda f: [first, second])
    other_way = Compositor(padding=0).process(noise_frame(), lambda f: [second, first])
    assert is_hidden(one_way, 50, 30, 50, 20)
    assert (one_way == other_way).all()


def test_the_original_frame_is_not_changed():
    frame = noise_frame()
    before = frame.image.copy()
    Compositor().process(frame, lambda f: [box()])
    assert (frame.image == before).all()


def test_box_over_the_frame_edge_is_clipped_and_hidden():
    output = Compositor(padding=0).process(noise_frame(), lambda f: [box(x=180, w=50)])
    assert is_hidden(output, 180, 30, 20, 20)


def test_crash_on_the_first_frame_sends_black_not_the_raw_frame():
    frame = noise_frame()
    compositor = Compositor()
    output = compositor.process(frame, crash)
    assert not output.any()
    assert output.shape == frame.image.shape
    assert compositor.held_frames == 1


def test_crash_later_holds_the_last_safe_frame():
    compositor = Compositor()
    safe = compositor.process(noise_frame(frame_id=0), nothing)
    raw = noise_frame(ts=0.1, frame_id=1)
    output = compositor.process(raw, crash)
    assert (output == safe).all()
    assert not (output == raw.image).all()


@pytest.mark.parametrize(
    "result",
    [
        None,
        [box(x=500)],
        [box(y=500)],
        [box(x=-100, w=50)],
        [box(w=0)],
        [box(h=-5)],
        [box(ttl_ms=-1000)],
        [box(ttl_ms=float("nan"))],
        ["not a detection"],
    ],
)
def test_bad_results_are_treated_as_failed_scans(result):
    raw = noise_frame()
    compositor = Compositor()
    output = compositor.process(raw, lambda f: result)
    assert compositor.held_frames == 1
    assert not output.any()


def test_a_good_box_next_to_a_bad_one_does_not_let_the_frame_out():
    raw = noise_frame()
    output = Compositor().process(raw, lambda f: [box(), box(x=500)])
    assert not output.any()


def test_only_this_frames_boxes_are_painted():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(), lambda f: [box(ttl_ms=500)])
    later = noise_frame(ts=0.01, frame_id=1)
    output = compositor.process(later, nothing)
    assert (output == later.image).all()
    assert compositor.covered_areas(100, 200) == []


def test_recovering_from_a_failed_scan_paints_only_the_new_frames_boxes():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(ts=0.0), lambda f: [box()])
    compositor.process(noise_frame(ts=0.1, frame_id=1), crash)
    recovered = noise_frame(ts=0.2, frame_id=2)
    output = compositor.process(recovered, lambda f: [box(x=100, frame_id=2)])
    assert is_hidden(output, 100)
    assert not is_hidden(output)


def test_a_box_from_another_frame_is_a_failed_scan():
    compositor = Compositor()
    compositor.process(noise_frame(), nothing)
    stale = noise_frame(ts=0.1, frame_id=1)
    compositor.process(stale, lambda f: [box(frame_id=0)])
    assert compositor.held_frames == 1


def test_moving_box_leaves_no_trail():
    """30 positions at 60 Hz, 6 px apart. This used to paint a 214 px band."""
    compositor = Compositor()
    width, height = 400, 720
    for index in range(30):
        y = 100 + index * 6
        frame = Frame(np.full((height, width, 3), 200, dtype=np.uint8), index / 60, "screen", index)
        detection = Detection(50, y, 100, 20, "api_key", 1.0, 500, index)
        output = compositor.process(frame, lambda f: [detection])
        covered = (output == 0).all(axis=2)
        rows = np.flatnonzero(covered[:, 100])
        assert rows.size == 40
        assert (rows[0], rows[-1]) == (y - 10, y + 29)
        assert covered.sum() == 120 * 40  # the padded cover, and nothing else


def test_jittering_boxes_never_add_up():
    compositor = Compositor()
    for index in range(20):
        frame = noise_frame(ts=index / 60, frame_id=index)
        wobble = index % 3
        detection = box(x=50 + wobble, y=30 - wobble, w=40 + wobble, frame_id=index)
        compositor.process(frame, lambda f: [detection])
        assert len(compositor.covered_areas(100, 200)) == 1


def test_error_text_is_not_printed(capsys):
    def leaky(frame):
        raise RuntimeError("sk-FAKE-KEY-1234567890")

    Compositor().process(noise_frame(), leaky)
    assert "sk-FAKE" not in capsys.readouterr().out


class RecordingSink:
    def __init__(self):
        self.images = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        self.images.append(image)
        time.sleep(0.001)


def test_pipeline_never_sends_the_fake_secret():
    source = TestPattern(640, 360)
    sink = RecordingSink()
    stats = DelayedPipeline(
        source, sink, source.detect, Compositor(), delay=0.05, fps=200, max_frames=120
    ).run()
    assert stats.shown + stats.skipped == 120
    white = np.array([255, 255, 255])
    for image in sink.images:
        # The secret is the only white text below the middle of the frame.
        assert not (image[180:] == white).all(axis=2).any()


def held(x=20, y=10, w=60, h=30, frame_id=0):
    return Detection(x, y, w, h, "held", 1.0, 0, frame_id)


def test_a_held_area_shows_the_last_safe_picture_there_not_black():
    compositor = Compositor(padding=0)
    first = noise_frame(0.0, 0)
    shown = compositor.process(first, nothing)
    second = noise_frame(0.1, 1)
    output = compositor.process(second, lambda frame: [held(frame_id=1)])
    assert (output[10:40, 20:80] == shown[10:40, 20:80]).all()  # what already went out
    assert (output[50:, :] == second.image[50:, :]).all()  # the rest is live
    assert compositor.held_areas == 1


def test_a_held_area_keeps_the_covers_the_last_safe_picture_had():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(0.0, 0), lambda frame: [box(x=30, y=15, w=20, h=10)])
    output = compositor.process(noise_frame(0.1, 1), lambda frame: [held(frame_id=1)])
    assert is_hidden(output, x=30, y=15, w=20, h=10)


def test_a_held_area_with_no_safe_picture_yet_is_black():
    compositor = Compositor(padding=0)
    output = compositor.process(noise_frame(0.0, 0), lambda frame: [held()])
    assert not output[10:40, 20:80].any()
