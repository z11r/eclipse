"""Windows OCR adapter checks. All images and text are fake."""
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from eclipse import platform
from eclipse.platform import WindowsOCRReader
from eclipse.text_detect.detection import Detector, DetectionError


def test_windows_reader_rejects_other_platforms(monkeypatch):
    monkeypatch.setattr(platform, 'IS_WINDOWS', False)
    with pytest.raises(RuntimeError, match='requires Windows'):
        WindowsOCRReader()


def test_windows_word_boxes_use_top_left_coordinates(monkeypatch):
    monkeypatch.setattr(platform, 'IS_WINDOWS', True)
    image = np.zeros((80, 200, 3), np.uint8)

    async def recognize(frame):
        assert frame is image
        word = SimpleNamespace(text='demo@example.com', bounding_rect=SimpleNamespace(
            x=20, y=10, width=150, height=30))
        return SimpleNamespace(lines=[SimpleNamespace(words=[word])])

    monkeypatch.setitem(sys.modules, 'winocr', SimpleNamespace(recognize_cv2=recognize))
    words, timing = WindowsOCRReader()(image)
    assert words == [([[20.0, 10.0], [170.0, 10.0], [170.0, 40.0], [20.0, 40.0]],
                      'demo@example.com', 1.0)]
    assert timing is None


def test_windows_reader_failure_remains_fail_closed(monkeypatch):
    monkeypatch.setattr(platform, 'IS_WINDOWS', True)

    async def recognize(frame):
        raise RuntimeError('FAKE_SECRET_IN_OCR_ERROR')

    monkeypatch.setitem(sys.modules, 'winocr', SimpleNamespace(recognize_cv2=recognize))
    detector = Detector(reader=WindowsOCRReader())
    with pytest.raises(DetectionError) as error:
        detector.detect(np.zeros((80, 200, 3), np.uint8))
    assert 'FAKE_SECRET' not in str(error.value)


@pytest.mark.skipif(not platform.IS_WINDOWS, reason='requires native Windows OCR')
def test_native_windows_ocr_detects_fake_email_and_code_from_scan_thread():
    import cv2
    frame = np.full((400, 900, 3), 255, np.uint8)
    for index, text in enumerate(['demo@example.com', 'Your code: 482619']):
        cv2.putText(frame, text, (40, 90 + 90 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 0), 2, cv2.LINE_AA)
    detector = Detector(reader=WindowsOCRReader())
    with ThreadPoolExecutor(max_workers=1) as worker:
        boxes = worker.submit(detector.detect, frame).result(timeout=10)
    assert {'email', '2fa'} <= {b[4] for b in boxes}
