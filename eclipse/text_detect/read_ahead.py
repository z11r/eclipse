"""Reads text before a frame's turn in the ordered scan, so OCR stops costing frame time.

One read of a changed area takes 5 to 60 ms, and only 16.7 ms pass between frames at 60 fps,
so reading in frame order can't keep up. Threads don't help: Vision allows one read at a
time per process. Processes do, nearly linearly (full frames per second, measured):

    workers     1     2     4     6
    threads    10    16    17    17
    processes  17    36    65    94

Frames wait in the delay buffer anyway, so reading starts as each frame is captured. Each
job is one frame's pixels and nothing else. The Detector stays single-threaded and in frame
order: on its turn it asks for an area of that frame and gets the answer read ahead for it.
Its state never reaches a worker. A result only describes the pixels of one frame id, so it
stays true whatever the detector's settings are, because rules run when it is collected.

If the detector asks for an area nobody read ahead, that area is read on the spot. So this
can only ever change how fast an answer comes, and a failed read raises, which holds the
frame back as any other failed scan does.

When frames arrive faster than they can be read, the queue must not grow. Reads would finish
after their frames expired, spending all the work on nothing: measured 14 fps from readers
able to do 37. So a frame that can't be reached in time is dropped on arrival, before any
work is spent on it, and the last safe frame stays up.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from multiprocessing import Pipe
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from itertools import count
from queue import PriorityQueue

import cv2
import numpy as np

from eclipse.types import Frame

MAX_FRAME_PIXELS = 7680 * 4320  # shared memory per worker; only pages in use cost anything
_READ_MARGIN = 96  # rows read either side of a change, past the detector's own margin
_QUEUE_PER_WORKER = 3  # reads allowed to wait per worker, a few hundred ms at worst
# Past this share of rows changed, read the whole frame. The detector reads whole frames
# once 60% of the screen changed, and a band that skips a still menu bar can't answer that.
_WHOLE_FRAME_SHARE = 0.45
_URGENT, _AHEAD, _STOP = 0, 1, 2  # queue order: a scan waiting now, reading ahead, shutdown
_READY_WAIT = 30.0  # seconds for a worker to load Vision
_RESULT_WAIT = 2.0  # seconds; past this a read has failed, and the frame is held back
_ROOT = Path(__file__).resolve().parents[2]


class ReadError(RuntimeError):
    """A read failed. Never holds the reason, which could contain screen text."""


class _Job:
    __slots__ = ("image", "done", "words", "failed", "cancelled", "queued", "worked")

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.done = threading.Event()
        self.words: list | None = None
        self.failed = False
        self.cancelled = False  # its frame was dropped, so nobody will ask for the answer
        self.queued = time.monotonic()
        self.worked = 0.0

    def result(self) -> list:
        if not self.done.wait(_RESULT_WAIT) or self.failed:
            raise ReadError("read failed")
        return self.words or []


class _Worker:
    """One OCR process and the thread that feeds it jobs."""

    def __init__(self, jobs: PriorityQueue, stats: dict, lost) -> None:
        self._jobs = jobs
        self._stats = stats
        self._lost = lost  # called with this worker if its process dies
        self._size = MAX_FRAME_PIXELS * 4
        self._memory = SharedMemory(create=True, size=self._size)
        self._channel, child = Pipe()
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "eclipse.ocr_worker",
                str(child.fileno()),
                str(self._memory._fd),
                str(self._size),
            ],
            pass_fds=(child.fileno(), self._memory._fd),
            cwd=_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,  # nothing from a worker can reach a log
        )
        child.close()
        self._memory.unlink()  # no name left behind; both sides keep their mapping
        self._thread = threading.Thread(target=self._serve, name="eclipse-ocr", daemon=True)

    def wait_until_ready(self) -> None:
        try:
            ready = self._channel.poll(_READY_WAIT) and self._channel.recv() == "ready"
        except (EOFError, OSError):
            ready = False
        if not ready:
            raise ReadError("worker didn't start")
        self._thread.start()

    def _serve(self) -> None:
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
                height, width = job.image.shape[:2]
                target = np.ndarray((height, width, 4), np.uint8, self._memory.buf)
                cv2.cvtColor(job.image, cv2.COLOR_BGR2BGRA, dst=target)
                del target  # shared memory can't close while a view of it lives
                self._channel.send((height, width))
                succeeded, words = self._channel.recv()
            except (EOFError, OSError, ValueError):
                # The process is gone. This frame is held back; the others carry on.
                job.failed = True
                job.done.set()
                self._stats["failed_reads"] += 1
                self._lost(self)
                return
            job.failed = not succeeded
            job.words = words
            job.worked = time.monotonic() - started
            self._stats["jobs"] += 1
            self._stats["wait_seconds"] += started - job.queued
            self._stats["work_seconds"] += job.worked
            self._stats["failed_reads"] += job.failed
            job.image = None  # let the frame go as soon as its read is done
            job.done.set()

    def close(self) -> None:
        self._channel.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self._memory.close()


class OcrPool:
    """Worker processes reading images in any order. Safe to submit from any thread."""

    def __init__(self, workers: int) -> None:
        self._jobs: PriorityQueue = PriorityQueue()
        self._order = count()  # first come, first served within each priority
        self.stats = {
            "jobs": 0,
            "failed_reads": 0,
            "cancelled": 0,
            "wait_seconds": 0.0,
            "work_seconds": 0.0,
        }
        self._alive_lock = threading.Lock()
        self._workers = [_Worker(self._jobs, self.stats, self._lost) for _ in range(workers)]
        self._alive = len(self._workers)
        try:
            for worker in self._workers:
                worker.wait_until_ready()
        except Exception:
            self.close()
            raise

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def backlog(self) -> int:
        return self._jobs.qsize()

    @property
    def alive(self) -> int:
        with self._alive_lock:
            return self._alive

    def _lost(self, worker: _Worker) -> None:
        with self._alive_lock:
            self._alive -= 1

    def submit(self, image: np.ndarray, urgent: bool = False) -> _Job:
        """Queue an image that nothing will change until the job is done.

        Urgent reads go ahead of reading ahead: the scan is stopped until they're done.
        """
        job = _Job(image)
        if image.shape[0] * image.shape[1] > MAX_FRAME_PIXELS or not self.alive:
            # Nobody can read this, so the frame must not pass as having no text.
            job.failed = True
            job.done.set()
            self.stats["failed_reads"] += 1
            return job
        self._jobs.put((_URGENT if urgent else _AHEAD, next(self._order), job))
        return job

    def close(self) -> None:
        for _ in self._workers:
            self._jobs.put((_STOP, next(self._order), None))
        for worker in self._workers:
            worker.close()


def changed_rows(now: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Rows that differ. About 1.3 ms for a Retina frame, and any change at all counts."""
    height = now.shape[0]
    different = cv2.compare(now.reshape(height, -1), before.reshape(height, -1), cv2.CMP_NE)
    return np.flatnonzero(cv2.reduce(different, 1, cv2.REDUCE_MAX))


def bands_to_read(rows: np.ndarray, height: int, margin: int) -> list[tuple[int, int]]:
    """Whole-width bands holding every changed row, with margin, without overlap.

    Whole width, because the detector grows areas sideways to hold known text, and a
    band read once can then answer any of them.
    """
    bands: list[list[int]] = []
    for row in rows.tolist():
        top, bottom = max(row - margin, 0), min(row + margin + 1, height)
        if bands and top <= bands[-1][1]:
            bands[-1][1] = bottom
        else:
            bands.append([top, bottom])
    return [(top, bottom) for top, bottom in bands]


def words_in_area(
    words: list, band_top: int, top: int, left: int, height: int, width: int
) -> list:
    """Words touching an area, moved from band coordinates to the area's own."""
    kept = []
    for corners, text, confidence in words:
        xs = [point[0] for point in corners]
        ys = [point[1] + band_top for point in corners]
        if max(xs) > left and min(xs) < left + width and max(ys) > top and min(ys) < top + height:
            moved = [[x - left, y - top] for x, y in zip(xs, ys)]
            kept.append((moved, text, confidence))
    return kept


class ReadAhead:
    """A reader for the text layer that starts work when a frame is captured.

    prepare() runs on the capture thread, in capture order. begin_frame() and read_area()
    run on the scan thread, in the order frames are scanned.
    """

    def __init__(self, pool: OcrPool, margin: int = _READ_MARGIN) -> None:
        self._pool = pool
        self._margin = margin
        self._before: np.ndarray | None = None
        self._lock = threading.Lock()
        self._read: dict[int, list[tuple[int, int, _Job]]] = {}
        self._bands: list[tuple[int, int, _Job]] = []
        self.stats = pool.stats
        self.stats.update({"hits": 0, "misses": 0, "prepare_seconds": 0.0, "bands": 0, "shed": 0})
        self.stats.setdefault("cancelled", 0)

    def prepare(self, frame: Frame) -> bool:
        """Start reading what changed since the last frame taken. False means drop this one.

        A dropped frame doesn't count as the last one taken, so the next frame's read covers
        everything that changed during it. Takes no copy: frames don't change.
        """
        if self._pool.backlog >= self._pool.size * _QUEUE_PER_WORKER:
            self.stats["shed"] += 1
            return False
        started = time.monotonic()
        image = frame.image
        height = image.shape[0]
        if self._before is None or self._before.shape != image.shape:
            bands = [(0, height)]
        else:
            bands = bands_to_read(changed_rows(image, self._before), height, self._margin)
            if sum(bottom - top for top, bottom in bands) > _WHOLE_FRAME_SHARE * height:
                bands = [(0, height)]
        self._before = image
        jobs = [(top, bottom, self._pool.submit(image[top:bottom])) for top, bottom in bands]
        with self._lock:
            self._read[frame.frame_id] = jobs
        self.stats["bands"] += len(jobs)
        self.stats["prepare_seconds"] += time.monotonic() - started
        return True

    def begin_frame(self, frame_id: int) -> None:
        """Collect this frame's jobs and forget any older ones, which were dropped."""
        with self._lock:
            for older in [key for key in self._read if key < frame_id]:
                for _, _, job in self._read.pop(older):
                    job.cancelled = True  # free the workers for frames still wanted
            self._bands = self._read.pop(frame_id, [])

    def __call__(self, image: np.ndarray) -> tuple[list, None]:
        return self._pool.submit(image, urgent=True).result(), None

    def read_area(self, image: np.ndarray, top: int, left: int) -> tuple[list, None]:
        """Words in an area of the current frame, read ahead if possible."""
        height, width = image.shape[:2]
        for band_top, band_bottom, job in self._bands:
            if band_top <= top and top + height <= band_bottom:
                self.stats["hits"] += 1
                return words_in_area(job.result(), band_top, top, left, height, width), None
        self.stats["misses"] += 1
        return self(image)

    def close(self) -> None:
        self._pool.close()
