"""On a camera, each cover follows its own text, checked with a tolerance, not exact pixels."""

import cv2
import numpy as np

from eclipse.text_detect.detection import Detector

FLOOR = 8
EMAIL = "alex.fake@example.com"


def _noisy(image, seed):
    noise = np.random.default_rng(seed).integers(-6, 7, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _sheet():
    sheet = np.full((60, 340, 3), 235, np.uint8)
    cv2.putText(sheet, EMAIL, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    return sheet


def _frame(x, y, seed):
    image = np.full((480, 640, 3), (70, 90, 110), np.uint8)
    image[y : y + 60, x : x + 340] = _sheet()
    return _noisy(image, seed)


class FirstReadOnly:
    """Reads the email once, then never again, like OCR losing text to a moving camera."""

    def __init__(self):
        self.place = None
        self.calls = 0

    def read_area(self, image, top, left):
        self.calls += 1
        if self.calls > 1 or self.place is None:
            return [], None
        x, y = self.place
        box = [[x + 8 - left, y + 12 - top], [x + 320 - left, y + 12 - top],
               [x + 320 - left, y + 48 - top], [x + 8 - left, y + 48 - top]]
        return [(box, EMAIL, 1.0)], None


def _covers(boxes):
    return [b for b in boxes if b[4] == "email"]


def test_cover_follows_its_sheet_without_reads():
    reader = FirstReadOnly()
    detect = Detector(reader=reader, noise_floor=FLOOR)
    x, y = 100, 100
    reader.place = (x, y)
    first = _covers(detect.detect(_frame(x, y, 0)))
    assert len(first) == 1
    for step in range(1, 8):
        x, y = x + 15, y + 6
        boxes = _covers(detect.detect(_frame(x, y, step)))
        assert len(boxes) == 1, step
        left, top, width, height, _ = boxes[0]
        # The cover sits on the text, and kept its size: no unsure margin.
        assert left <= x + 8 and left + width >= x + 320
        assert top <= y + 12 and top + height >= y + 48
        assert (width, height) == first[0][2:4]


def test_cover_does_not_chase_other_motion():
    reader = FirstReadOnly()
    detect = Detector(reader=reader, noise_floor=FLOOR)
    reader.place = (100, 100)
    base = np.full((480, 640, 3), (70, 90, 110), np.uint8)
    base[100:160, 100:440] = _sheet()
    detect.detect(_noisy(base, 0))
    start = _covers(detect.detect(_noisy(base, 1)))
    for step in range(2, 8):
        image = base.copy()
        # A hand with no text in it waves beside the sheet.
        cv2.circle(image, (520, 60 + step * 40), 40, (150, 170, 200), -1)
        boxes = _covers(detect.detect(_noisy(image, step)))
        assert boxes == start


def test_camera_frames_are_not_followed_as_scrolls():
    detect = Detector(reader=FirstReadOnly(), noise_floor=FLOOR)
    page = _noisy(np.tile(_sheet(), (8, 1, 1))[:480], 0)
    image = np.full((480, 640, 3), 90, np.uint8)
    image[:, :340] = page
    detect.detect(image)
    scrolled = image.copy()
    scrolled[:-20, :340] = image[20:, :340]
    detect.detect(scrolled)
    assert detect.stats["scrolls"] == 0 and detect.stats["moves"] == 0
