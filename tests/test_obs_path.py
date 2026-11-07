import cv2
import numpy as np
import pytest

from eval.obs_path import analyse, newest_recording, pattern, read_number, verify


def test_a_folder_means_its_newest_recording(tmp_path):
    import os

    older, newer = tmp_path / "a.mov", tmp_path / "b.mp4"
    older.write_bytes(b"x")
    newer.write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not a video")
    os.utime(older, (1, 1))
    assert newest_recording(str(tmp_path)) == str(newer)


def test_a_folder_with_no_recording_says_so_instead_of_failing_obscurely(tmp_path, capsys):
    assert verify(str(tmp_path)) == 2
    assert "Start Recording" in capsys.readouterr().out


@pytest.mark.parametrize("number", [0, 1, 255, 256, 4097, 65535])
def test_number_survives_a_full_size_round_trip(number):
    assert read_number(pattern(number)) == number


@pytest.mark.parametrize("number", [7, 1234, 60000])
def test_number_survives_obs_scaling_to_720p_and_lossy_encoding(number):
    small = cv2.resize(pattern(number), (1280, 720), interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    assert ok
    assert read_number(cv2.imdecode(encoded, cv2.IMREAD_COLOR)) == number


def test_a_blended_frame_is_rejected_rather_than_misread():
    blended = cv2.addWeighted(pattern(0x1234), 0.5, pattern(0x4321), 0.5, 0)
    assert read_number(blended) is None


def test_a_pattern_is_exactly_1080p_and_the_bar_moves():
    first, second = pattern(100), pattern(101)
    assert first.shape == (1080, 1920, 3)
    assert not np.array_equal(first[300:], second[300:])


def test_perfect_60_fps_recording():
    report = analyse(list(range(600)), 60)
    assert report.repeats == 0
    assert report.skipped == 0
    assert report.fresh_fps == pytest.approx(60, abs=0.1)
    assert report.fresh_per_second and set(report.fresh_per_second) == {60}


def test_a_30_fps_source_in_a_60_fps_recording_reports_30_not_60():
    numbers = [index // 2 for index in range(600)]
    report = analyse(numbers, 60)
    assert report.fresh_fps == pytest.approx(30, abs=0.5)
    assert report.repeats == pytest.approx(240, abs=2)
    assert report.longest_hold == 2


def test_lost_frames_are_counted():
    numbers = [index for index in range(600) if index % 10]
    report = analyse(numbers, 60)
    assert report.skipped > 40


def test_unreadable_frames_are_counted():
    numbers = list(range(300))
    numbers[150] = None
    assert analyse(numbers, 60).undecoded == 1


def test_a_recording_with_nothing_readable_reports_nothing():
    report = analyse([None] * 10, 60)
    assert report.distinct == 0
    assert report.fresh_fps == 0


def test_segments_of_one_recording_are_reported_apart():
    from eval.obs_path import SEGMENT, split_segments

    numbers = [None] * 5 + list(range(300)) + [None] * 90 + [SEGMENT + i // 2 for i in range(300)]
    parts = split_segments(numbers)
    assert sorted(parts) == [0, 1]
    assert analyse(parts[0], 60).fresh_fps == pytest.approx(60, abs=0.5)
    assert analyse(parts[1], 60).fresh_fps == pytest.approx(30, abs=0.5)
    assert analyse(parts[1], 60).backwards == 0


def test_a_plan_names_known_outputs_only():
    from eval.obs_path import parse_plan

    assert parse_plan("pyvirtualcam, stamped+late") == [("pyvirtualcam", False), ("stamped", True)]
    with pytest.raises(ValueError):
        parse_plan("stamped+early")
    with pytest.raises(ValueError):
        parse_plan(",".join(["stamped"] * 7))
