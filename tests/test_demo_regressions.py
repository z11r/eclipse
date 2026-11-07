"""Fake OCR reproductions of misses found in the visible demo."""

import numpy as np

from eclipse.text_detect.detection import Detector


class ChangingCodeReader:
    def __init__(self):
        self.text = ''

    def read_area(self, image, top, left):
        polygon = [[40 - left, 60 - top], [330 - left, 60 - top],
                   [330 - left, 90 - top], [40 - left, 90 - top]]
        return [(polygon, self.text, 0.95)], None


def test_code_cover_survives_repeated_changes_with_missing_ocr_spaces():
    reader = ChangingCodeReader()
    detector = Detector(reader=reader, enabled=['2fa'])
    for index, text in enumerate([
        'Your code:123456', 'Your code:654321', 'Yourcode:918273', 'Yourcode:482619',
    ]):
        reader.text = text
        frame = np.full((200, 400, 3), 255, dtype=np.uint8)
        frame[60:90, 40:330] = 20 + index * 70
        frame[60:90, 40:330:10] = 0
        boxes = detector.detect(frame)
        assert any(
            category == '2fa' and x <= 300 < x + w and y <= 75 < y + h
            for x, y, w, h, category in boxes
        ), f'The cover disappeared on change {index}'


class BlindAfterFirstReader:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        if self.calls > 1:
            return [], None
        return [([[40, 60], [240, 60], [240, 90], [40, 90]],
                 'demo@example.com', 0.95)], None


def test_known_secret_moves_with_window_even_when_ocr_misses_it():
    reader = BlindAfterFirstReader()
    detector = Detector(reader=reader, enabled=['email'])
    patch = np.random.default_rng(19).integers(0, 180, (31, 201, 3), dtype=np.uint8)
    for index, (left, top) in enumerate([(40, 60), (300, 140), (100, 220), (400, 60), (400, 60)]):
        frame = np.full((400, 700, 3), 240, dtype=np.uint8)
        frame[top:top+31, left:left+201] = patch
        boxes = detector.detect(frame)
        assert any(
            category in {'email', 'unread'} and x <= left + 10 < x + w and y <= top + 10 < y + h
            for x, y, w, h, category in boxes
        ), f'Known secret became visible after move {index}'
        email_boxes = [b for b in boxes if b[4] == 'email']
        assert len(email_boxes) == 1, 'Moving a secret must not leave a trail of old boxes'


def test_window_movement_does_not_match_different_colors_with_the_same_gray():
    import cv2
    from eclipse.text_detect.covers import CoverTracks, Sighting

    tracks = CoverTracks()
    values = np.random.default_rng(31).integers(40, 170, (31, 201), dtype=np.uint8)
    patch = np.repeat(values[:, :, None], 3, axis=2)
    other = patch.copy()
    other[:, :, 0] += 5
    other[:, :, 1] -= 1
    assert np.array_equal(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(other, cv2.COLOR_BGR2GRAY))

    before = np.full((400, 700, 3), 240, dtype=np.uint8)
    before[60:91, 40:241] = patch
    tracks.update(before, [Sighting('email', (40, 60, 240, 90), 1, True)], [(0, 400, 0, 700)])
    now = np.full_like(before, 240)
    now[140:171, 300:501] = other
    tracks.update(now, [], [(0, 400, 0, 700)])
    # The old cover may be held while uncertain, but must not jump onto this decoy.
    assert all(rect[0] < 100 for category, rect in tracks.covers())


def test_dragging_window_does_not_black_out_uniform_exposed_background():
    import cv2

    class WindowReader:
        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            if self.calls > 1:
                return [], None
            return [([[170, 145], [370, 145], [370, 175], [170, 175]],
                     'demo@example.com', 0.95)], None

    detector = Detector(reader=WindowReader(), enabled=['email'])
    patch = np.random.default_rng(23).integers(0, 180, (31, 201, 3), dtype=np.uint8)

    def window(top):
        image = np.full((450, 800, 3), 90, dtype=np.uint8)
        cv2.rectangle(image, (150, top), (740, top + 190), (255, 255, 255), -1)
        cv2.rectangle(image, (150, top), (740, top + 190), (30, 30, 30), 2)
        cv2.putText(image, 'FAKE WINDOW', (170, top + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 30, 30), 1)
        image[top + 55:top + 86, 170:371] = patch
        return image

    detector.detect(window(90))
    boxes = detector.detect(window(140))
    assert any(c == 'email' and x <= 180 < x+w and y <= 205 < y+h
               for x, y, w, h, c in boxes)
    assert not any(c == 'unread' for *_, c in boxes), 'Blank desktop gained a black bar'
