import numpy as np
import pytest

from eclipse import main as app
from eclipse.capture.pattern import TestPattern
from eclipse.compositor.compositor import Compositor
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


def test_padding_covers_the_edges_of_the_box():
    output = Compositor(padding=6).process(noise_frame(), lambda f: [box()])
    assert is_hidden(output, 44, 24, 52, 32)


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


def test_held_frame_is_never_the_failed_frames_pixels_after_recovery():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(ts=0.0), lambda f: [box()])
    compositor.process(noise_frame(ts=0.1, frame_id=1), crash)
    recovered = compositor.process(noise_frame(ts=0.2, frame_id=2), nothing)
    assert is_hidden(recovered)


def test_blur_holds_for_ttl_after_the_clue_is_gone():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(ts=0.0), lambda f: [box(ttl_ms=500)])
    held = compositor.process(noise_frame(ts=0.4, frame_id=1), nothing)
    assert is_hidden(held)


def test_blur_ends_after_ttl():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(ts=0.0), lambda f: [box(ttl_ms=500)])
    expired = compositor.process(noise_frame(ts=0.6, frame_id=1), nothing)
    assert not is_hidden(expired)


def test_seeing_a_clue_again_restarts_its_ttl():
    compositor = Compositor(padding=0)
    compositor.process(noise_frame(ts=0.0), lambda f: [box(ttl_ms=500)])
    compositor.process(noise_frame(ts=0.4, frame_id=1), lambda f: [box(ttl_ms=500)])
    still_held = compositor.process(noise_frame(ts=0.8, frame_id=2), nothing)
    assert is_hidden(still_held)


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


def test_pipeline_never_sends_the_fake_secret():
    source = TestPattern(640, 360)
    compositor = Compositor()
    raw_frames = []

    def spy(frame):
        raw_frames.append(frame)
        return source.detect(frame)

    sink = RecordingSink()
    app.run(source, sink, spy, compositor, max_frames=120)
    assert len(sink.images) == 120
    for frame, output in zip(raw_frames, sink.images):
        x, y, w, h = source._secret_box(frame.frame_id)
        raw_secret = frame.image[y : y + h, x : x + w]
        hidden = output[y : y + h, x : x + w].reshape(-1, 3)
        assert raw_secret.std() > 0
        assert (hidden == hidden[0]).all()


def test_pipeline_keeps_running_through_a_detector_crash():
    source = TestPattern(320, 240)
    sink = RecordingSink()

    def crash_on_frame_two(frame):
        if frame.frame_id == 2:
            raise RuntimeError
        return []

    compositor = Compositor()
    app.run(source, sink, crash_on_frame_two, compositor, max_frames=4)
    assert len(sink.images) == 4
    assert (sink.images[2] == sink.images[1]).all()
    assert compositor.held_frames == 1


def test_repeated_box_does_not_pile_up():
    compositor = Compositor()
    for i in range(50):
        compositor.process(noise_frame(ts=i * 0.03, frame_id=i), lambda f: [box()])
    assert len(compositor._active) == 1
