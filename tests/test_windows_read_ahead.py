"""Windows workers read ahead without mixing frames or releasing failed reads."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from eclipse.text_detect import windows_read_ahead as read_ahead
from eclipse.text_detect.read_ahead import ReadAhead, ReadError
from eclipse.text_detect.windows_read_ahead import WindowsOcrPool
from eclipse.types import Frame


def word(text):
    return ([[0, 0], [50, 0], [50, 15], [0, 15]], text, 1.0)


def test_windows_workers_read_concurrently_and_keep_each_jobs_pixels(monkeypatch):
    meet = threading.Barrier(2)

    class Reader:
        def __call__(self, image):
            if image.shape == (64, 64, 3):
                return [], None
            meet.wait(timeout=2)
            return [word(f"demo{int(image[0, 0, 0])}@example.com")], None

    monkeypatch.setattr(read_ahead, 'WindowsOCRReader', Reader)
    pool = WindowsOcrPool(2)
    try:
        first = pool.submit(np.full((30, 100, 3), 1, np.uint8))
        second = pool.submit(np.full((30, 100, 3), 2, np.uint8))
        assert first.result()[0][1] == 'demo1@example.com'
        assert second.result()[0][1] == 'demo2@example.com'
        assert pool.stats['jobs'] == 2
    finally:
        pool.close()
    assert all(not w._thread.is_alive() for w in pool._workers)


def test_windows_startup_failure_is_sanitized(monkeypatch):
    def broken():
        raise RuntimeError('FAKE_SECRET_IN_MODEL_ERROR')

    monkeypatch.setattr(read_ahead, 'WindowsOCRReader', broken)
    with pytest.raises(ReadError) as error:
        WindowsOcrPool(1)
    assert 'FAKE_SECRET' not in str(error.value)


def test_windows_failed_job_cannot_pass_as_empty_text(monkeypatch):
    class Reader:
        def __call__(self, image):
            if image.shape == (64, 64, 3):
                return [], None
            raise RuntimeError('FAKE_SECRET_IN_READ_ERROR')

    monkeypatch.setattr(read_ahead, 'WindowsOCRReader', Reader)
    pool = WindowsOcrPool(1)
    try:
        with pytest.raises(ReadError) as error:
            pool.submit(np.zeros((30, 100, 3), np.uint8)).result()
        assert 'FAKE_SECRET' not in str(error.value)
        assert pool.stats['failed_reads'] == 1
    finally:
        pool.close()


def test_windows_read_ahead_keeps_frame_coordinates_and_unchanged_cache(monkeypatch):
    class Reader:
        def __call__(self, image):
            return ([word('demo@example.com')] if image.shape != (64, 64, 3) else []), None

    monkeypatch.setattr(read_ahead, 'WindowsOCRReader', Reader)
    reader = ReadAhead(WindowsOcrPool(2))
    try:
        image = np.zeros((100, 200, 3), np.uint8)
        assert reader.prepare(Frame(image, 0, 'screen', 1))
        reader.begin_frame(1)
        found, _ = reader.read_area(image[:25, :80], 0, 0)
        assert found == [word('demo@example.com')]
        assert reader.prepare(Frame(image.copy(), 1, 'screen', 2))
        reader.begin_frame(2)
        assert reader._bands == []
        assert reader.stats['hits'] == 1
        assert reader.stats['bands'] == 1
    finally:
        reader.close()


def test_main_windows_workers_are_used_and_closed(monkeypatch, tmp_path):
    from eclipse import main as app
    from eclipse.capture.pattern import TestPattern
    from test_passthrough import FakeSink

    pool = SimpleNamespace(closed=False)
    pool.close = lambda: setattr(pool, 'closed', True)
    wrapped = SimpleNamespace(close=pool.close)
    counts, readers = [], []
    monkeypatch.setattr(read_ahead, 'WindowsOcrPool', lambda n: counts.append(n) or pool)
    monkeypatch.setattr(app, 'ReadAhead', lambda p: wrapped)
    monkeypatch.setattr(app, 'screen_capture_allowed', lambda: True)
    monkeypatch.setattr(app, 'screen_capture_kit_available', lambda: False)
    monkeypatch.setattr(app, 'ScreenCapture', lambda **kwargs: TestPattern(320, 240))
    monkeypatch.setattr(app, 'VirtualCam', FakeSink)
    monkeypatch.setattr(app, 'TextDetector', lambda **kwargs: readers.append(kwargs['reader']) or (lambda frame: []))
    assert app.main(['--ocr', 'windows', '--ocr-workers', '2', '--max-frames', '3',
                     '--delay', '0.05', '--look-ahead', '0', '--log', str(tmp_path/'run.log')]) == 0
    assert counts == [2] and readers == [wrapped] and pool.closed


def test_apple_mode_keeps_using_the_original_pool(monkeypatch, tmp_path):
    from eclipse import main as app
    from eclipse.capture.pattern import TestPattern
    from test_passthrough import FakeSink

    calls, readers = [], []
    wrapped = SimpleNamespace(close=lambda: calls.append('closed'))
    monkeypatch.setattr(app, 'AppleVisionReader', lambda: object())
    monkeypatch.setattr(app, 'OcrPool', lambda n: calls.append(('apple', n)) or object())
    monkeypatch.setattr(app, 'ReadAhead', lambda pool: wrapped)
    monkeypatch.setattr(read_ahead, 'WindowsOcrPool', lambda n: pytest.fail('Apple used Windows pool'))
    monkeypatch.setattr(app, 'screen_capture_allowed', lambda: True)
    monkeypatch.setattr(app, 'screen_capture_kit_available', lambda: False)
    monkeypatch.setattr(app, 'ScreenCapture', lambda **kwargs: TestPattern(320, 240))
    monkeypatch.setattr(app, 'VirtualCam', FakeSink)
    monkeypatch.setattr(app, 'TextDetector', lambda **kwargs: readers.append(kwargs) or (lambda frame: []))
    assert app.main(['--ocr', 'apple', '--max-frames', '3', '--delay', '0.05',
                     '--look-ahead', '0', '--log', str(tmp_path/'apple.log')]) == 0
    assert calls == [('apple', 4), 'closed']
    assert readers == [{'reader': wrapped, 'pixel_scale': 2}]


def test_windows_pool_never_constructs_an_apple_worker(monkeypatch):
    from eclipse.text_detect import read_ahead as shared

    class Reader:
        def __call__(self, image):
            return [word('demo@example.com')], None

    monkeypatch.setattr(read_ahead, 'WindowsOCRReader', Reader)
    monkeypatch.setattr(shared, '_Worker', lambda *args: pytest.fail('Windows started Apple worker'))
    pool = WindowsOcrPool(1)
    try:
        assert pool.submit(np.zeros((30, 100, 3), np.uint8)).result() == [word('demo@example.com')]
    finally:
        pool.close()


def test_native_windows_read_ahead_detects_fake_values_on_their_frame():
    from eclipse import platform
    if not platform.IS_WINDOWS:
        pytest.skip('native Windows OCR requires Windows')
    import cv2
    from eclipse.text_detect import TextDetector

    reader = ReadAhead(WindowsOcrPool(2))
    detect = TextDetector(reader=reader)
    try:
        for index in range(2):
            image = np.full((400, 900, 3), 255, np.uint8)
            for row, text in enumerate(['demo@example.com', 'Your code: 482619']):
                cv2.putText(image, text, (40 + index*30, 90 + row*90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
            frame = Frame(image, float(index), 'fake-screen', index)
            assert detect.prepare(frame)
            boxes = detect(frame)
            assert {'email', 'two_fa'} <= {b.category for b in boxes}
            assert all(b.frame_id == index for b in boxes)
        assert reader.stats['hits'] > 0
    finally:
        reader.close()
