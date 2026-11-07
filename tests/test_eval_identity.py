"""The eval must score what viewers would see, not the frame that just came in."""

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.types import Detection, Frame
from eval import run_eval

HEIGHT, WIDTH = 100, 240
BOX = [0, 20, 220, 50]


def blank():
    return np.full((HEIGHT, WIDTH, 3), 30, np.uint8)


def with_secret():
    image = blank()
    cv2.putText(image, "FAKE SECRET", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return image


def frames(images, fps=30, ids=None):
    ids = ids if ids is not None else range(len(images))
    return [Frame(image, i / fps, "screen", i) for i, image in zip(ids, images)]


def secret(first=0, last=0, box=BOX, move=(0, 0)):
    return {
        "id": "fake", "text": "FAKE SECRET", "category": "keyword",
        "first_frame": first, "last_frame": last, "box": box, "move": list(move),
    }


def labels(*secrets):
    return {"secrets": list(secrets) or [secret()]}


def misses_then_fails(frame):
    if frame.frame_id:
        raise RuntimeError("injected failure")
    return []


def fails_from(failing):
    def detect(frame):
        if frame.frame_id >= failing:
            raise RuntimeError("injected failure")
        return []

    return detect


def hides(frame):
    return [Detection(*BOX, "keyword", 1.0, 0, frame.frame_id)]


def test_a_held_leak_counts_every_time_it_airs_even_after_its_episode_ended():
    images = [with_secret(), blank(), blank()]
    seen = []
    report = run_eval.evaluate_frames(
        frames(images), labels(), misses_then_fails,
        on_frame=lambda raw, out: seen.append(bool(out.max() > 100)),
    )
    assert sum(seen) == 3
    assert report["leaked_frames"] == 3
    assert report["leaks_by_secret"] == {"fake": 3}


def test_a_held_safe_frame_is_not_a_leak_when_the_input_has_a_secret():
    images = [blank(), with_secret(), with_secret(), with_secret()]
    report = run_eval.evaluate_frames(
        frames(images), labels(secret(1, 3)), fails_from(1)
    )
    assert report["leaked_frames"] == 0
    assert report["held_frames"] == 3


def test_a_startup_blackout_shows_nothing_and_counts_as_black():
    images = [with_secret(), with_secret()]
    report = run_eval.evaluate_frames(frames(images), labels(secret(0, 1)), fails_from(0))
    assert report["leaked_frames"] == 0
    assert report["blackout_frames"] == 2
    assert report["fresh_frames"] == 0
    assert report["capture_to_present_p95_ms"] == 0.0


def test_repeated_content_is_told_from_fresh_content():
    images = [blank(), blank(), blank(), blank(), blank()]

    def fails_in_the_middle(frame):
        if frame.frame_id in (2, 3):
            raise RuntimeError("injected failure")
        return []

    report = run_eval.evaluate_frames(frames(images), labels(secret(9, 9)), fails_in_the_middle)
    assert report["fresh_frames"] == 3
    assert report["repeated_frames"] == 2


def test_frozen_time_is_in_clip_time_however_fast_the_eval_runs():
    images = [blank() for _ in range(7)]
    report = run_eval.evaluate_frames(frames(images, fps=10), labels(secret(99, 99)), fails_from(1))
    assert report["frozen_longest_ms"] == pytest.approx(600)
    assert report["frozen_total_ms"] == pytest.approx(600)


def test_presentation_time_is_kept_apart_from_compute_time():
    images = [blank() for _ in range(5)]
    report = run_eval.evaluate_frames(
        frames(images, fps=10), labels(secret(99, 99)), lambda frame: [], delay=0.5
    )
    assert report["compute_p95_ms"] < 100
    assert report["capture_to_present_p95_ms"] == pytest.approx(500, abs=1)


def test_the_age_of_held_content_adds_to_presentation_time():
    images = [blank() for _ in range(21)]
    report = run_eval.evaluate_frames(
        frames(images, fps=10), labels(secret(99, 99)), fails_from(1), delay=0.5
    )
    assert report["capture_to_present_p95_ms"] > 2000


def test_a_scan_longer_than_the_delay_sets_presentation_time():
    import time

    def slow(frame):
        time.sleep(0.03)
        return []

    report = run_eval.evaluate_frames(
        frames([blank(), blank()]), labels(secret(99, 99)), slow, delay=0.0
    )
    assert report["capture_to_present_p95_ms"] >= 25


def test_clip_time_and_our_own_timing_stay_separate():
    late = [Frame(blank(), 1000.0 + i, "screen", i) for i in range(3)]
    report = run_eval.evaluate_frames(late, labels(secret(99, 99)), lambda frame: [], delay=0.1)
    assert report["compute_p95_ms"] < 500  # not the clip's timestamps
    assert report["capture_to_present_p95_ms"] == pytest.approx(100, abs=1)


def test_input_frames_that_never_arrived_are_counted():
    images = [blank(), blank(), blank()]
    report = run_eval.evaluate_frames(
        frames(images, ids=[0, 1, 5]), labels(secret(99, 99)), lambda frame: []
    )
    assert report["dropped_inputs"] == 3


def test_a_moving_secret_is_scored_where_the_held_frame_shows_it():
    images = [with_secret(), blank(), blank()]
    moving = secret(0, 2, move=(50, 0))  # by the last input it would be off the mark
    report = run_eval.evaluate_frames(frames(images), labels(moving), misses_then_fails)
    assert report["leaked_frames"] == 3


def test_coverage_says_whether_detection_or_fallback_did_the_hiding():
    images = [with_secret() for _ in range(4)]

    def hides_then_fails(frame):
        if frame.frame_id >= 2:
            raise RuntimeError("injected failure")
        return hides(frame)

    report = run_eval.evaluate_frames(frames(images), labels(secret(0, 3)), hides_then_fails)
    assert report["leaked_frames"] == 0
    assert report["detected_coverage"] == 0.5
    assert report["fallback_coverage"] == 0.5


def test_a_leak_is_not_counted_as_detected_coverage():
    images = [with_secret() for _ in range(2)]
    report = run_eval.evaluate_frames(frames(images), labels(secret(0, 1)), lambda frame: [])
    assert report["detected_coverage"] == 0.0
    assert report["fallback_coverage"] == 0.0


def test_output_held_across_a_size_change_is_scored_at_its_own_size():
    big = cv2.resize(with_secret(), (WIDTH * 2, HEIGHT * 2))
    report = run_eval.evaluate_frames(
        frames([with_secret(), big, big]), labels(), misses_then_fails
    )
    assert report["leaked_frames"] == 3


@pytest.mark.parametrize("size", [(1920, 1080), (640, 640), (160, 90)], ids=str)
def test_a_leak_survives_the_final_resize_in_the_score(size):
    images = [with_secret(), with_secret()]
    leak = labels(secret(0, 1))
    report = run_eval.evaluate_frames(frames(images), leak, lambda f: [], output_size=size)
    assert report["leaked_frames"] == 2


@pytest.mark.parametrize("size", [(1920, 1080), (640, 640), (160, 90)], ids=str)
def test_a_cover_survives_the_final_resize_in_the_score(size):
    images = [with_secret(), with_secret()]
    report = run_eval.evaluate_frames(
        frames(images), labels(secret(0, 1)), hides, output_size=size
    )
    assert report["leaked_frames"] == 0


def test_fitted_box_lands_where_fit_puts_the_pixels():
    from eclipse.output.virtual_cam import fit

    image = np.zeros((982, 1512, 3), np.uint8)
    x, y, w, h = 300, 200, 400, 60
    image[y : y + h, x : x + w] = 255
    fitted = fit(image, 1920, 1080)
    left, top, width, height = run_eval.fitted_box((x, y, w, h), 982, 1512, (1920, 1080))
    rows, columns = np.nonzero(fitted[:, :, 0] > 128)
    assert left <= columns.min() and columns.max() < left + width
    assert top <= rows.min() and rows.max() < top + height


def test_every_rendered_image_keeps_its_identity():
    compositor = Compositor()
    first = Frame(blank(), 1.5, "screen", 7)
    safe = compositor.render(first, lambda frame: [], generation=3)
    assert (safe.state, safe.source_frame_id, safe.ts, safe.generation, safe.source) == (
        "scanned", 7, 1.5, 3, "screen",
    )
    later = Frame(with_secret(), 2.5, "screen", 8)
    held = compositor.render(later, fails_from(0), generation=4)
    assert (held.state, held.source_frame_id, held.ts, held.generation) == ("held", 7, 1.5, 3)
    assert held.image is safe.image


def test_a_blackout_claims_no_content():
    rendered = Compositor().render(Frame(blank(), 1.0, "screen", 4), fails_from(0), generation=2)
    assert rendered.state == "blackout"
    assert rendered.source is None
    assert rendered.source_frame_id == -1
    assert rendered.generation == 2
    assert not rendered.image.any()


def test_a_detector_that_needed_no_read_is_labelled_cached():
    class Reuses:
        reused_read = True

        def __call__(self, frame):
            return []

    rendered = Compositor().render(Frame(blank(), 0.0, "screen", 0), Reuses())
    assert rendered.state == "cached"


def test_forgetting_the_last_safe_image_means_black_next_time():
    compositor = Compositor()
    compositor.render(Frame(blank(), 0.0, "screen", 0), lambda frame: [])
    compositor.forget()
    held = compositor.render(Frame(blank(), 0.1, "screen", 1), fails_from(0))
    assert held.state == "blackout"


def test_no_frames_is_an_error():
    with pytest.raises(ValueError):
        run_eval.evaluate_frames([], labels(), lambda frame: [])


def test_report_labels_compute_time_as_compute_time(capsys):
    report = run_eval.evaluate_frames(frames([blank()]), labels(secret(9, 9)), lambda f: [])
    assert "p95_ms" not in report
    run_eval.print_report("test", report)
    printed = capsys.readouterr().out
    assert "offline: no capture, delay buffer or output" in printed
    assert "modelled from clip time" in printed
