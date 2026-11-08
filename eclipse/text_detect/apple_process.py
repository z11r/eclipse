"""Synchronous Apple OCR crops in a dedicated process, separate from screen OCR.

One pool is shared by webcam-only and PIP detectors, but every read has its own
job/result. No read-ahead pixel differencing: camera noise filtering remains with
the camera's ordered TextDetector. Worker failure raises and holds output.
"""
from threading import Lock

from eclipse.text_detect.read_ahead import OcrPool, ReadError


class AppleProcessReader:
    def __init__(self, pool=None):
        self._pool = OcrPool(1) if pool is None else pool
        self._reading = Lock()
        self._closed = False
        self.stats = self._pool.stats

    def __call__(self, image):
        with self._reading:  # at most one camera crop submitted at a time
            if self._closed:
                raise ReadError('camera reader is closed')
            try:
                words = self._pool.submit(image, urgent=True).result()
            except Exception:
                raise ReadError('camera read failed') from None
            return words, None

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool.close()
