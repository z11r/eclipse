"""Run: python -m unittest discover -s tests -v (no OCR model needed)."""
import unittest
import numpy as np
from eclipse.text_detect.detection import Detector, DetectionError, apply_covers, classify_text, luhn_valid


class FakeReader:
    def __init__(self, results=None):
        self.results = results
        self.calls = 0
        self.fail = False

    def __call__(self, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError('FAKE_SECRET_MUST_NOT_ESCAPE')
        return self.results, None


def region(text, left=20, right=90):
    return ([[left, 20], [right, 20], [right, 40], [left, 40]], text, 0.95)


class DetectionTests(unittest.TestCase):
    def test_categories(self):
        for text, expected in [
            ('demo@example.com', 'email'), ('+1 (780) 555-0123', 'phone'),
            ('Your code: 123456', '2fa'), ('code: 123 456', '2fa'),
            ('sk-' + 'a' * 24, 'api_key'), ('API_KEY=abcdef123456', 'api_key'),
            ('4111 1111 1111 1111', 'card'), ('my street', 'keyword'),
        ]:
            with self.subTest(category=expected):
                self.assertIn(expected, classify_text(text, ['My Street']))
        self.assertEqual(classify_text('hello world'), set())
        self.assertFalse(luhn_valid('4111111111111112'))
        self.assertFalse(luhn_valid('0000000000000000'))

    def test_keyword_spacing(self):
        for text in (
            'Welcome to Demo Maple Avenue',
            'WelcometoDemo MapleAvenue',
            'DEMOmapleAVENUE',
            'Demo   Maple\tAvenue',
        ):
            with self.subTest(text=text):
                self.assertIn('keyword', classify_text(text, ['Demo Maple Avenue']))

        self.assertNotIn('keyword', classify_text('hello world', ['', '   ', '\t']))
        self.assertNotIn('keyword', classify_text('Demo Oak Avenue', ['Demo Maple Avenue']))

    def test_fullwidth_text(self):
        for text, category in (
            ('+1\uff08780\uff09 555-0123', 'phone'),
            ('\uff11\uff12\uff13\uff14\uff15\uff16', '2fa'),
            ('demo\uff20example\uff0ecom', 'email'),
        ):
            with self.subTest(category=category):
                self.assertIn(category, classify_text(text))

        self.assertIn('keyword', classify_text('Demo Maple Avenue', ['Demo\u3000Maple Avenue']))

    def test_fullwidth_phone_with_only_phone_enabled(self):
        detector = Detector(
            enabled=['phone'],
            padding=10,
            reader=FakeReader([region('+1\uff08780\uff09 555-0123')]),
        )
        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        boxes = detector.detect(frame)
        self.assertEqual(boxes, [(10, 10, 90, 40, 'phone')])
        self.assertTrue((apply_covers(frame, boxes)[20:40, 20:90] == 0).all())

    def test_padding_and_opaque_covers(self):
        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        boxes = Detector(padding=10, reader=FakeReader([region('demo@example.com')])).detect(frame)
        self.assertEqual(boxes, [(10, 10, 90, 40, 'email')])
        redacted = apply_covers(frame, boxes)
        self.assertTrue((redacted[10:50, 10:100] == 0).all())
        self.assertTrue((frame == 255).all())

    def test_cache_and_single_pixel_change(self):
        reader = FakeReader([region('demo@example.com')])
        detector = Detector(reader=reader)
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        result = detector.detect(frame)
        result.clear()  # Caller must not be able to remove cached boxes.
        self.assertTrue(detector.detect(frame))
        self.assertEqual(reader.calls, 1)
        frame[0, 0, 0] = 1
        detector.detect(frame)
        self.assertEqual(reader.calls, 2)
        detector.reset()
        detector.detect(frame)
        self.assertEqual(reader.calls, 3)

    def test_config_invalidates_cache(self):
        reader = FakeReader([region('private name')])
        detector = Detector(reader=reader)
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [])
        detector.configure(keywords=['private name'])
        self.assertTrue(detector.detect(frame))
        detector.configure(enabled=[])
        self.assertEqual(detector.detect(frame), [])
        self.assertEqual(reader.calls, 3)

    def test_split_email_and_clipping(self):
        detector = Detector(padding=10, reader=FakeReader([
            region('demo@', left=0, right=40),
            region('example.com', left=45, right=120),
        ]))
        boxes = detector.detect(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertIn((0, 10, 50, 40, 'email'), boxes)
        self.assertIn((35, 10, 85, 40, 'email'), boxes)

    def test_failure_is_sanitized_and_not_cached(self):
        reader = FakeReader()
        detector = Detector(reader=reader)
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        detector.detect(frame)
        reader.fail = True
        frame[0, 0] = 1
        with self.assertRaises(DetectionError) as caught:
            detector.detect(frame)
        self.assertNotIn('FAKE_SECRET', str(caught.exception))
        reader.fail = False
        detector.detect(frame)
        self.assertEqual(reader.calls, 3)

    def test_invalid_frames_and_malformed_ocr(self):
        detector = Detector(reader=FakeReader())
        for frame in [None, np.zeros((8, 8)), np.zeros((8, 8, 4), dtype=np.uint8)]:
            with self.assertRaises(DetectionError):
                detector.detect(frame)
        detector = Detector(reader=FakeReader([([[float('nan'), 0]] * 4, 'secret', 1)]))
        with self.assertRaises(DetectionError):
            detector.detect(np.zeros((80, 120, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
