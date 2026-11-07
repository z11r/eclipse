import numpy as np
import pytest

from eclipse.capture.clip import ClipCapture
from eval import fake_detectors as fakes
from eval import make_clip, run_eval
from eval.truth import box_at


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    labels = make_clip.make_labels()
    path = tmp_path_factory.mktemp("clips") / labels["clip"]
    make_clip.write_clip(path, labels)
    return str(path), labels


def evaluate(clip, factory):
    path, labels = clip
    return run_eval.evaluate(path, labels, factory(labels))


def frames_on_screen(labels, secret_id):
    secret = next(s for s in labels["secrets"] if s["id"] == secret_id)
    return secret["last_frame"] - secret["first_frame"] + 1


def test_every_planted_secret_stays_inside_the_frame(clip):
    _, labels = clip
    for secret in labels["secrets"]:
        for frame in (secret["first_frame"], secret["last_frame"]):
            x, y, w, h = box_at(secret, frame)
            assert 0 <= x and x + w <= labels["width"]
            assert 0 <= y and y + h <= labels["height"]


def test_clip_reader_returns_every_frame_then_none(clip):
    path, labels = clip
    with ClipCapture(path) as reader:
        frames = []
        while (frame := reader.read()) is not None:
            frames.append(frame)
    assert len(frames) == labels["frames"]
    assert frames[30].ts == pytest.approx(1.0)
    assert frames[0].image.shape == (labels["height"], labels["width"], 3)


def test_missing_clip_is_an_error():
    with pytest.raises(FileNotFoundError):
        with ClipCapture("no_such_clip.mp4"):
            pass


def test_perfect_detector_leaks_nothing(clip):
    report = evaluate(clip, fakes.perfect)
    assert report["leaked_frames"] == 0
    assert report["recall"] == 1.0
    assert report["held_frames"] == 0


def test_missed_secret_is_reported_for_every_frame_it_shows(clip):
    report = evaluate(clip, fakes.misses_one)
    assert list(report["leaks_by_secret"]) == ["email"]
    assert report["leaks_by_secret"]["email"] == frames_on_screen(clip[1], "email")


def test_still_secret_leaks_as_soon_as_the_detector_stops_reporting_it(clip):
    report = evaluate(clip, fakes.first_sighting_only)
    expected = frames_on_screen(clip[1], "still_api_key") - 1
    assert report["leaks_by_secret"]["still_api_key"] == expected


def test_half_covered_secret_is_a_leak(clip):
    report = evaluate(clip, fakes.half_box)
    assert set(report["leaks_by_secret"]) == {s["id"] for s in clip[1]["secrets"]}


def test_crashing_detector_leaks_nothing_and_holds_every_frame(clip):
    report = evaluate(clip, fakes.crashes)
    assert report["leaked_frames"] == 0
    assert report["held_frames"] == clip[1]["frames"]


def test_false_blur_is_measured(clip):
    clean = evaluate(clip, fakes.perfect)
    noisy = evaluate(clip, fakes.over_blurs)
    assert noisy["false_blur"] - clean["false_blur"] > 0.03
    assert noisy["leaked_frames"] == 0


def test_visible_share_counts_only_uncovered_text():
    raw = np.zeros((20, 100, 3), dtype=np.uint8)
    raw[5:15, 10:90] = 255
    covered = raw.copy()
    covered[:, 10:50] = 0
    box = (0, 0, 100, 20)
    assert run_eval.visible_share(raw, raw, box) == 1.0
    assert run_eval.visible_share(raw, covered, box) == pytest.approx(0.5)
    assert run_eval.visible_share(raw, np.zeros_like(raw), box) == 0.0


def test_solid_fill_is_not_a_leak_even_if_text_edges_match_its_colour():
    raw = np.full((20, 100, 3), 30, dtype=np.uint8)
    raw[5:15, 10:50] = 255
    raw[5:15, 50:90] = 85  # soft text edge, close to the fill's colour
    fill = np.full_like(raw, 70)
    assert run_eval.visible_share(raw, fill, (0, 0, 100, 20)) == 0.0


def test_empty_label_box_is_an_error_not_a_pass():
    blank = np.zeros((20, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        run_eval.visible_share(blank, blank, (0, 0, 100, 20))


def test_split_screen_video_has_every_frame_side_by_side(clip, tmp_path):
    path, labels = clip
    video = tmp_path / "split.mp4"
    write, writer = run_eval.split_screen_writer(video, labels)
    run_eval.evaluate(path, labels, fakes.perfect(labels), on_frame=write)
    writer.release()
    with ClipCapture(str(video)) as reader:
        first = reader.read()
        count = 1
        while reader.read() is not None:
            count += 1
    assert count == labels["frames"]
    assert first.image.shape == (labels["height"], labels["width"] * 2, 3)
