"""The faster detector steps must give exactly the answers the slower ones gave."""

import cv2
import numpy as np
import pytest

from eclipse.text_detect import detection
from eclipse.text_detect.detection import (
    _match_text,
    _row_means,
    _shift_differences,
    find_matches,
)

rng = np.random.default_rng(7)


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (240, 400), (1964, 3024)])
def test_row_means_are_exactly_numpys(shape):
    gray = rng.integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(_row_means(gray), gray.mean(axis=1))


def slow_differences(now_rows, before_rows, limit):
    height = len(now_rows)
    values = []
    for shift in range(-limit, limit + 1):
        start = max(shift, 0), max(-shift, 0)
        length = height - abs(shift)
        values.append(float(np.abs(
            now_rows[start[0]:start[0] + length] - before_rows[start[1]:start[1] + length]
        ).mean()))
    return np.array(values)


def best_shift(differences, limit):
    return min((s for s in range(-limit, limit + 1) if s), key=lambda s: differences[s + limit])


def profiles():
    for height in (241, 600, 1964):
        limit = min(480, height // 2)
        base = rng.uniform(20, 200, height + 800)
        for true_shift in (7, -33, 120, -230):
            yield base[400 + true_shift : 400 + true_shift + height], base[400 : 400 + height], limit
        lines = np.tile(np.r_[np.full(14, 30.0), np.full(26, 90.0)], height // 40 + 2)
        yield lines[5 : 5 + height], lines[:height], limit
        yield np.full(height, 34.0), np.full(height, 34.0), limit


@pytest.mark.parametrize("now_rows,before_rows,limit", list(profiles()))
def test_shift_differences_match_the_loop_and_pick_the_same_shift(now_rows, before_rows, limit):
    fast = _shift_differences(now_rows, before_rows, limit)
    slow = slow_differences(now_rows, before_rows, limit)
    assert np.allclose(fast, slow, rtol=0, atol=1e-9)
    assert best_shift(fast, limit) == best_shift(slow, limit)


def test_mismatch_from_absdiff_is_the_same_mask():
    a = rng.integers(0, 256, (300, 500), dtype=np.uint8)
    b = rng.integers(0, 256, (300, 500), dtype=np.uint8)
    blurred_a, blurred_b = cv2.blur(a, (3, 3)), cv2.blur(b, (3, 3))
    slow = np.abs(blurred_a.astype(np.int16) - blurred_b.astype(np.int16)) > 40
    fast = cv2.absdiff(blurred_a, blurred_b) > 40
    assert np.array_equal(slow, fast)


@pytest.mark.parametrize(
    "text",
    ["contact alex.fake@example.com", "AKIAIOSFODNN7EXAMPLE", "4242 4242 4242 4242", "hello", ""],
)
def test_remembered_matches_equal_fresh_ones(text):
    assert find_matches(text) == _match_text(text)
    assert find_matches(text) == _match_text(text)  # and again, from memory


def test_keywords_are_part_of_what_is_remembered():
    assert find_matches("project bluebird launch") == []
    assert [match[0] for match in find_matches("project bluebird launch", ["bluebird"])] == [
        "keyword"
    ]
    assert find_matches("project bluebird launch") == []


def test_changing_a_returned_list_cannot_change_later_answers():
    first = find_matches("alex.fake@example.com")
    first.clear()
    assert find_matches("alex.fake@example.com")


def test_keywords_given_as_a_list_work():
    assert find_matches("bluebird", ["bluebird"]) == _match_text("bluebird", ["bluebird"])


def test_classify_text_still_uses_the_rules():
    assert detection.classify_text("alex.fake@example.com") == {"email"}


def slow_shifted_match(image, before, dx, dy, area):
    height, width = image.shape[:2]
    top, bottom, left, right = area
    match = np.zeros((bottom - top, right - left), bool)
    for y in range(top, bottom):
        for x in range(left, right):
            if 0 <= y - dy < height and 0 <= x - dx < width:
                difference = np.abs(image[y, x].astype(int) - before[y - dy, x - dx].astype(int))
                match[y - top, x - left] = difference.max() <= detection._NOISE_FLOOR
    return match


@pytest.mark.parametrize("dx,dy", [(0, 0), (3, -2), (-7, 5), (30, 0), (-60, -45), (90, 90)])
def test_shifted_match_is_the_per_pixel_answer(dx, dy):
    before = rng.integers(0, 256, (50, 80, 3), dtype=np.uint8)
    image = np.roll(before, (dy, dx), axis=(0, 1))
    image = np.clip(image.astype(int) + rng.integers(-12, 13, image.shape), 0, 255).astype(np.uint8)
    area = (4, 46, 10, 75)
    fast = detection._shifted_match(image, before, dx, dy, area)
    assert np.array_equal(fast, slow_shifted_match(image, before, dx, dy, area))


def test_shift_shares_match_the_loop():
    before = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    image = np.roll(before, (4, -6), axis=(0, 1))
    image[::7] = rng.integers(0, 256, image[::7].shape, dtype=np.uint8)
    ys, xs = rng.integers(0, 120, 900), rng.integers(0, 160, 900)
    shifts = [(dx, dy) for dx in range(-9, 3) for dy in range(-2, 9)] + [(500, 0)]
    fast = detection._shift_shares(image, before, ys, xs, image[ys, xs].astype(np.int16), shifts)
    for (dx, dy), share in zip(shifts, fast):
        hits = 0
        for y, x in zip(ys, xs):
            if 0 <= y - dy < 120 and 0 <= x - dx < 160:
                difference = np.abs(image[y, x].astype(int) - before[y - dy, x - dx].astype(int))
                hits += difference.max() <= detection._NOISE_FLOOR
        assert share == hits / ys.size


LINES = [
    "Meeting notes for the Friday review", "Lunch is at noon in the big room",
    "billing: pat.sample@example.org", "Ship the build after the demo",
    "rows move with the scroll offset", "support line 780-555-0142",
]


def drag(dx, dy):
    """Two frames of a text window dragged by dx, dy over a plain desktop. Fake text."""
    part = np.full((330, 600, 3), 240, np.uint8)
    part[:28] = (205, 205, 205)
    for row, line in enumerate(LINES):
        cv2.putText(
            part, line, (14, 64 + row * 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 1,
            cv2.LINE_AA,
        )
    frames = []
    for x, y in ((180, 160), (180 + dx, 160 + dy)):
        image = np.full((700, 1000, 3), (90, 60, 40), np.uint8)
        image[y : y + part.shape[0], x : x + part.shape[1]] = part
        frames.append(image)
    return frames


@pytest.mark.parametrize("dx,dy", [(-3, 22), (40, -1), (-25, -30), (-1, 0)])
def test_screened_search_picks_the_full_searchs_shift(monkeypatch, dx, dy):
    """Only a few candidate shifts get the full vote, and the drag is still found."""
    original = detection.Detector._move_shift
    picks = {}
    for finalists in (detection._MOVE_FINALISTS, 10_000):
        monkeypatch.setattr(detection, "_MOVE_FINALISTS", finalists)
        seen = []
        monkeypatch.setattr(
            detection.Detector, "_move_shift",
            lambda self, *args: seen.append(original(self, *args)) or seen[-1],
        )
        detector = detection.Detector(reader=lambda image: ([], None))
        for image in drag(dx, dy):
            detector.detect(image)
        picks[finalists] = seen
    assert picks[detection._MOVE_FINALISTS] == picks[10_000] == [(dx, dy)]
