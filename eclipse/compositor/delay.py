"""Delay buffer: smooth output from a scanner that can't always keep up.

Three steps run side by side:

    capture  -> frames waiting to be scanned  (steady rate)
    scanning -> frames safe to send           (as fast as it can)
    output   -> OBS, each frame `delay` late  (steady rate)

A slow scan uses up some of the delay instead of freezing the picture. Output only
ever receives frames from the compositor, so a raw frame has no way to reach it. A
frame that can't be scanned in time is dropped and the last safe frame stays up.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from eclipse.compositor.compositor import Compositor, Detector


MAX_BUFFER_BYTES = 6 * 1024**3
_JITTER_SLOTS = 1.5  # room for scans running late, on top of the delay itself
_MAX_BEHIND = 3  # due frames allowed to wait for later ticks before skipping


class _BufferLimit:
    """Frames to hold: enough for the delay and some jitter, within a byte limit."""

    def __init__(self, delay: float, fps: int, max_bytes: int) -> None:
        self._wanted = max(int(fps * delay * _JITTER_SLOTS) + 2, 4)
        self._max_bytes = max_bytes
        self.frames = self._wanted

    def note_frame(self, size: int) -> None:
        self.frames = max(min(self._wanted, self._max_bytes // max(size, 1)), 2)


def _buffer_limit(delay: float, fps: int, max_bytes: int) -> _BufferLimit:
    return _BufferLimit(delay, fps, max_bytes)


@dataclass
class PipelineStats:
    captured: int = 0
    scanned: int = 0
    dropped: int = 0  # never scanned in time, so never shown
    shown: int = 0  # distinct frames actually sent to the output
    skipped: int = 0  # scanned, but a newer frame was sent in the same tick
    repeated: int = 0  # output ticks that had no new frame
    max_backlog: int = 0
    over_limit: int = 0  # dropped because the buffer hit its memory limit
    backlog: int = 0
    grab_seconds: float = 0.0
    scan_seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)  # capture to output, in ms
    capture_seconds: float = 0.0


class DelayedPipeline:
    def __init__(
        self,
        source,
        sink,
        detect: Detector,
        compositor: Compositor,
        delay: float = 1.0,
        fps: int = 30,
        max_frames: int | None = None,
        max_buffer_bytes: int = MAX_BUFFER_BYTES,
    ) -> None:
        self._source = source
        self._sink = sink
        self._detect = detect
        self._compositor = compositor
        self._delay = delay
        self._interval = 1 / fps
        self._max_buffered = _buffer_limit(delay, fps, max_buffer_bytes)
        self._max_frames = max_frames
        self._waiting: deque = deque()
        self._safe: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self._scan_done = threading.Event()
        self._error: BaseException | None = None
        self.stats = PipelineStats()

    def _capture(self) -> None:
        try:
            started = time.monotonic()
            with self._source:
                next_grab = time.monotonic()
                while not self._stop.is_set():
                    if self._max_frames is not None and self.stats.captured >= self._max_frames:
                        break
                    grab_started = time.monotonic()
                    frame = self._source.read()
                    self.stats.grab_seconds += time.monotonic() - grab_started
                    with self._frame_ready:
                        self.stats.captured += 1
                        self._max_buffered.note_frame(frame.image.nbytes)
                        if len(self._waiting) + len(self._safe) >= self._max_buffered.frames:
                            self.stats.dropped += 1  # memory limit reached
                            self.stats.over_limit += 1
                            continue
                        self._waiting.append(frame)
                        self.stats.backlog = len(self._waiting)
                        self.stats.max_backlog = max(self.stats.max_backlog, len(self._waiting))
                        self._frame_ready.notify()
                    next_grab = max(next_grab + self._interval, time.monotonic())
                    time.sleep(max(next_grab - time.monotonic(), 0))
        except BaseException as error:
            self._error = error
            self._stop.set()
        finally:
            self.stats.capture_seconds = time.monotonic() - started
            self._capture_done.set()
            with self._frame_ready:
                self._frame_ready.notify()

    def _scan(self) -> None:
        try:
            while True:
                with self._frame_ready:
                    while not self._waiting:
                        if self._capture_done.is_set() or self._stop.is_set():
                            return
                        self._frame_ready.wait(0.1)
                    frame = self._waiting.popleft()

                # Already overdue, so skip to a newer one. Most scans are quick, so
                # one slow scan doesn't mean the frames behind it will be late too.
                if time.monotonic() > frame.ts + self._delay and self._waiting:
                    self.stats.dropped += 1
                    continue

                started = time.monotonic()
                output = self._compositor.process(frame, self._detect)
                self.stats.scanned += 1
                self.stats.scan_seconds += time.monotonic() - started
                with self._lock:
                    self._safe.append((frame.ts, output))
        except BaseException as error:
            self._error = error
            self._stop.set()
        finally:
            self._scan_done.set()

    def _next_output(self, current: np.ndarray | None) -> np.ndarray | None:
        """The next safe frame now due, or the current one if none is.

        Frames go out one per tick, so timing jitter doesn't waste them. Older ones
        are only skipped if output has really fallen behind.
        """
        due = time.monotonic() - self._delay
        chosen = None
        with self._lock:
            while self._safe and self._safe[0][0] <= due:
                if chosen is not None:
                    self.stats.skipped += 1
                chosen = self._safe.popleft()
                behind = sum(1 for ts, _ in self._safe if ts <= due)
                if behind <= _MAX_BEHIND:
                    break
        if chosen is None:
            if current is not None:
                self.stats.repeated += 1
            return current
        ts, current = chosen
        self.stats.shown += 1
        self.stats.latencies.append((time.monotonic() - ts) * 1000)
        return current

    def _finished(self) -> bool:
        with self._lock:
            return self._scan_done.is_set() and not self._safe

    def run(self) -> PipelineStats:
        threads = [
            threading.Thread(target=self._capture, name="eclipse-capture", daemon=True),
            threading.Thread(target=self._scan, name="eclipse-scan", daemon=True),
        ]
        current = None
        with self._sink:
            for thread in threads:
                thread.start()
            try:
                while not self._finished() and not self._stop.is_set():
                    current = self._next_output(current)
                    if current is None:
                        time.sleep(self._interval / 2)
                    else:
                        self._sink.send(current)
            except KeyboardInterrupt:
                pass
            finally:
                self._stop.set()
                with self._frame_ready:
                    self._frame_ready.notify_all()
                for thread in threads:
                    thread.join(timeout=5)
        if self._error is not None:
            raise self._error
        return self.stats
