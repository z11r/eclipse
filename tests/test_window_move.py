"""A window dragged whole keeps its covers exactly on its text.

All text here is fake.
"""

import cv2
import numpy as np

from eclipse.text_detect.detection import Detector

FIRST = "pat.sample@example.org"
SECOND = "alex.fake@example.com"
FONT = cv2.FONT_HERSHEY_SIMPLEX
DESKTOP = (90, 60, 40)


def window(text: str) -> np.ndarray:
    """A small light window with lines of filler and one fake email."""
    image = np.full((220, 520, 3), 235, np.uint8)
    image[:24] = (200, 200, 200)
    for row, line in enumerate(["Meeting notes for Friday", "Lunch is at noon", text]):
        cv2.putText(image, line, (16, 70 + row * 50), FONT, 0.7, (20, 20, 20), 2)
    return image


def screen(part: np.ndarray, x: int, y: int) -> np.ndarray:
    image = np.zeros((600, 900, 3), np.uint8)
    image[:] = DESKTOP
    image[::40] = (120, 90, 60)  # desktop texture, so blank isn't all that is still
    image[y : y + part.shape[0], x : x + part.shape[1]] = part
    return image


class TemplateReader:
    """OCR that finds a planted line only when handed all of it, like real OCR."""

    def __init__(self, texts):
        self.templates = {}
        for text in texts:
            part = window(text)
            self.templates[text] = part[150 - 26 : 150 + 12, 10:400]

    def __call__(self, image):
        found = []
        for text, template in self.templates.items():
            if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
                continue
            cost = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
            left, top = cv2.minMaxLoc(cost)[2]
            height, width = template.shape[:2]
            if np.array_equal(image[top : top + height, left : left + width], template):
                corners = [
                    [left, top], [left + width, top],
                    [left + width, top + height], [left, top + height],
                ]
                found.append((corners, text, 1.0))
        return found, None


def email_boxes(boxes):
    return sorted(box[:4] for box in boxes if box[4] == "email")


def test_cover_follows_a_diagonal_drag_exactly():
    reader = TemplateReader([FIRST])
    detector = Detector(reader=reader)
    part = window(FIRST)
    before = email_boxes(detector.detect(screen(part, 100, 100)))
    assert len(before) == 1

    for step in range(1, 8):
        after = email_boxes(detector.detect(screen(part, 100 + 30 * step, 100 + 12 * step)))
        x, y, w, h = before[0]
        assert after == [(x + 30 * step, y + 12 * step, w, h)], f"cover drifted on step {step}"

    assert detector.stats["moves"] == 7


def test_text_that_changed_during_a_drag_is_read_again_and_covered():
    reader = TemplateReader([FIRST, SECOND])
    detector = Detector(reader=reader)
    detector.detect(screen(window(FIRST), 100, 100))

    boxes = email_boxes(detector.detect(screen(window(SECOND), 130, 112)))

    # The new email sits where the old one would have moved to, so it must be covered,
    # and only by a cover read from the new pixels.
    assert len(boxes) >= 1
    x, y, w, h = boxes[0]
    assert x <= 130 + 10 and x + w >= 130 + 400 and y <= 112 + 124 and y + h >= 112 + 162


def test_still_text_beside_a_dragged_window_keeps_its_cover():
    reader = TemplateReader([FIRST])
    detector = Detector(reader=reader)
    still = window(FIRST)
    image = screen(still, 360, 360)
    moving = window("nothing secret here")
    image[40 : 40 + 220, 20 : 20 + 520] = moving
    first = email_boxes(detector.detect(image))
    assert len(first) == 1

    image = screen(still, 360, 360)
    image[40 + 10 : 40 + 10 + 220, 20 + 24 : 20 + 24 + 520] = moving
    assert email_boxes(detector.detect(image)) == first


class FlakyReader(TemplateReader):
    """Misses everything on every other read, as real OCR misses lines now and then."""

    def __init__(self, texts):
        super().__init__(texts)
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        if self.calls % 2 == 0:
            return [], None
        return super().__call__(image)


def test_cover_stays_on_dragged_text_when_ocr_misses_it():
    reader = FlakyReader([FIRST])
    detector = Detector(reader=reader)
    part = window(FIRST)
    before = email_boxes(detector.detect(screen(part, 100, 100)))
    assert len(before) == 1
    x, y, w, h = before[0]

    for step in range(1, 9):
        boxes = email_boxes(detector.detect(screen(part, 100 + 30 * step, 100 + 12 * step)))
        # The moved text must sit under a cover of the same size in every frame,
        # whether or not this frame's read found it.
        assert (x + 30 * step, y + 12 * step, w, h) in boxes, f"cover lost on step {step}"
