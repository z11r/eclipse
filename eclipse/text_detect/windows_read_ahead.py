"""Windows-native OCR workers for the shared read-ahead queue.

Imported only by the Windows OCR path. The Apple process pool stays unchanged.
"""
from __future__ import annotations

import threading
from itertools import count
from queue import PriorityQueue
import time

import numpy as np

from eclipse.platform import WindowsOCRReader
from eclipse.text_detect.read_ahead import OcrPool, ReadError, _READY_WAIT


class _WindowsWorker:
    """One native Windows OCR reader, kept on its own background thread."""

    def __init__(self, jobs, stats, lost) -> None:
        self._jobs, self._stats, self._lost = jobs, stats, lost
        self._ready = threading.Event()
        self._started = False
        self._thread = threading.Thread(target=self._serve, name="eclipse-windows-ocr", daemon=True)

    def wait_until_ready(self) -> None:
        self._thread.start()
        if not self._ready.wait(_READY_WAIT) or not self._started:
            raise ReadError("worker didn't start")

    def _serve(self) -> None:
        try:
            reader = WindowsOCRReader()
            reader(np.zeros((64, 64, 3), np.uint8))  # load the language model on this thread
            self._started = True
        except Exception:
            self._lost(self)
            return
        finally:
            self._ready.set()
        while True:
            _, _, job = self._jobs.get()
            if job is None:
                return
            if job.cancelled:
                self._stats["cancelled"] += 1
                job.failed = True
                job.image = None
                job.done.set()
                continue
            started = time.monotonic()
            try:
                job.words, _ = reader(job.image)
            except Exception:
                job.failed = True
                self._stats["failed_reads"] += 1
            finally:
                job.worked = time.monotonic() - started
                self._stats["jobs"] += 1
                self._stats["wait_seconds"] += started - job.queued
                self._stats["work_seconds"] += job.worked
                job.image = None
                job.done.set()

    def close(self) -> None:
        if self._thread.ident is not None:
            self._thread.join(timeout=2)


class WindowsOcrPool(OcrPool):
    """Use native Windows threads with the existing bounded read-ahead interface."""

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("at least one Windows OCR worker is required")
        self._jobs = PriorityQueue()
        self._order = count()
        self.stats = {
            "jobs": 0, "failed_reads": 0, "cancelled": 0,
            "wait_seconds": 0.0, "work_seconds": 0.0,
        }
        self._alive_lock = threading.Lock()
        self._workers = [_WindowsWorker(self._jobs, self.stats, self._lost) for _ in range(workers)]
        self._alive = len(self._workers)
        try:
            for worker in self._workers:
                worker.wait_until_ready()
        except Exception:
            self.close()
            raise
