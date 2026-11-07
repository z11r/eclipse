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
from typing import NamedTuple

import numpy as np

from eclipse.compositor.compositor import Compositor, Detector
from eclipse.types import Frame


MAX_BUFFER_BYTES = 6 * 1024**3
_JITTER_SLOTS = 1.5  # room for scans running late, on top of the delay itself
_MAX_BEHIND = 3  # due frames allowed to wait for later ticks before skipping
_FRESH_ROW_STEP = 8  # compare every 8th row to tell a new picture from a repeat
# A new frame that would wait longer than this share of the delay to be scanned is dropped
# on arrival. Dropping late, after waiting, makes the detector compare against an older
# frame, which costs more, which makes the next frames later still: a real screen fell to
# 26 fps that way. Dropping on arrival keeps each scanned frame next to the one before.
_SHED_AT_WAIT = 0.5
_SCAN_SMOOTHING = 0.1  # weight of the newest scan in the running estimate of scan time
_SAMPLES_KEPT = 4096  # timing samples held until the stats log collects them
_LONGEST_IDLE = 0.01  # seconds the output waits between looks while it has nothing to send


def _samples() -> deque:
    return deque(maxlen=_SAMPLES_KEPT)


class _Queued(NamedTuple):
    frame: Frame
    fresh: bool  # its pixels differ from the frame captured before it


class _SafeFrame(NamedTuple):
    ts: float
    image: np.ndarray
    frame_id: int
    fresh: bool  # its pixels differ from the frame captured before it
    size: int  # bytes counted against the buffer for this frame


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
    captured: int = 0  # frames read from the source
    capture_attempts: int = 0  # capture ticks, whether or not they read a frame
    capture_skips: int = 0  # ticks that read nothing, as there was no room for a frame
    admitted: int = 0  # frames that joined the queue
    discarded: int = 0  # frames read but refused, for a source that can't skip a read
    scanned: int = 0
    dropped: int = 0  # read but never scanned in time, so never shown
    shown: int = 0  # distinct frames actually sent to the output
    skipped: int = 0  # scanned, but a newer frame was sent in the same tick
    repeated: int = 0  # output ticks that had no new frame
    max_backlog: int = 0
    over_limit: int = 0  # capture ticks lost to the memory limit
    backlog: int = 0
    in_flight_frames: int = 0  # queued, being scanned, or waiting to go out
    in_flight_bytes: int = 0
    peak_in_flight_bytes: int = 0
    grab_seconds: float = 0.0
    scan_seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)  # capture to output, in ms
    capture_seconds: float = 0.0
    fresh: int = 0  # captured frames whose pixels differ from the frame before
    fresh_shown: int = 0  # sent frames that showed the viewer something new
    held: int = 0  # scans that failed, so the last safe frame stayed up
    id_gaps: int = 0  # source frames missing between two sent frames
    detect_seconds: float = 0.0  # the detector alone, within scan_seconds
    composite_seconds: float = 0.0  # the rest of scan_seconds
    last_sent_id: int = -1
    prepare_failures: int = 0
    shed: int = 0  # dropped on arrival because no reader could reach them in time
    scan_ages: deque = field(default_factory=_samples)  # ms from capture to scan start
    slack: deque = field(default_factory=_samples)  # ms left when safe; below 0 is late
    present_intervals: deque = field(default_factory=_samples)  # ms between new pictures


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
        self._waiting: deque[_Queued] = deque()
        self._safe: deque[_SafeFrame] = deque()
        self._last_rows: np.ndarray | None = None
        self._last_presented: float | None = None
        self._scan_estimate = 0.0  # seconds per scan, smoothed
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self._scan_done = threading.Event()
        self._error: BaseException | None = None
        self.stats = PipelineStats()

    def _room_for_a_frame(self) -> bool:
        with self._lock:
            return self.stats.in_flight_frames < self._max_buffered.frames

    def _pace(self, next_grab: float) -> float:
        next_grab = max(next_grab + self._interval, time.monotonic())
        self._stop.wait(max(next_grab - time.monotonic(), 0))
        return next_grab

    def _capture(self) -> None:
        started = time.monotonic()
        try:
            with self._source:
                next_grab = time.monotonic()
                while not self._stop.is_set():
                    if self._max_frames is not None and self.stats.captured >= self._max_frames:
                        break
                    # A source that keeps its own time has to be read to stay current.
                    paced = getattr(self._source, "paced", False)
                    self.stats.capture_attempts += 1
                    if paced or self._room_for_a_frame():
                        self._grab()
                    else:
                        with self._lock:
                            self.stats.capture_skips += 1
                            self.stats.over_limit += 1
                    if not paced:
                        next_grab = self._pace(next_grab)
        except BaseException as error:
            self._error = error
            self._stop.set()
        finally:
            self.stats.capture_seconds = time.monotonic() - started
            self._capture_done.set()
            with self._frame_ready:
                self._frame_ready.notify_all()

    def _grab(self) -> None:
        grab_started = time.monotonic()
        frame = self._source.read()
        self.stats.grab_seconds += time.monotonic() - grab_started
        if frame is None:  # nothing new on screen yet
            return
        fresh = self._is_fresh(frame)
        size = frame.image.nbytes
        # Room first, so no reading ahead starts on a frame that will be refused anyway.
        room = self._room_for_a_frame()
        shed = room and (self._scan_would_be_late() or not self._prepare(frame))
        with self._frame_ready:
            self.stats.captured += 1
            self.stats.fresh += fresh
            self._max_buffered.note_frame(size)
            if not room or self.stats.in_flight_frames >= self._max_buffered.frames:
                self.stats.dropped += 1
                self.stats.discarded += 1
                self.stats.over_limit += 1
                return
            if shed:
                self.stats.dropped += 1  # no reader could reach it in time
                self.stats.shed += 1
                return
            self._waiting.append(_Queued(frame, fresh))
            self._note_in_flight(1, size)
            self.stats.admitted += 1
            self.stats.backlog = len(self._waiting)
            self.stats.max_backlog = max(self.stats.max_backlog, len(self._waiting))
            self._frame_ready.notify_all()

    def _note_in_flight(self, frames: int, size: int) -> None:
        """Call with the lock held."""
        stats = self.stats
        stats.in_flight_frames += frames
        stats.in_flight_bytes += size
        stats.peak_in_flight_bytes = max(stats.peak_in_flight_bytes, stats.in_flight_bytes)

    def _is_fresh(self, frame: Frame) -> bool:
        """Whether this frame's pixels differ from the last one's.

        A polling source can grab one picture twice, and that must not count as motion.
        """
        rows = frame.image[::_FRESH_ROW_STEP]
        last = self._last_rows
        self._last_rows = rows
        return last is None or last.shape != rows.shape or not np.array_equal(last, rows)

    def _scan_would_be_late(self) -> bool:
        """Whether a frame arriving now would wait too long behind those already waiting."""
        with self._lock:
            waiting = len(self._waiting)
        return waiting * self._scan_estimate > self._delay * _SHED_AT_WAIT

    def _prepare(self, frame: Frame) -> bool:
        """Let the detector start early. False means drop this frame now.

        Dropping is always safe, as the last safe frame stays up. If preparing fails, the
        frame still goes to the scanner, so that only costs the head start.
        """
        prepare = getattr(self._detect, "prepare", None)
        if prepare is None:
            return True
        try:
            return prepare(frame) is not False
        except Exception:
            self.stats.prepare_failures += 1
            return True

    def _timed_detect(self, frame: Frame) -> list:
        started = time.monotonic()
        try:
            return self._detect(frame)
        finally:
            self.stats.detect_seconds += time.monotonic() - started

    def _scan(self) -> None:
        try:
            while True:
                with self._frame_ready:
                    while not self._waiting:
                        if self._capture_done.is_set() or self._stop.is_set():
                            return
                        self._frame_ready.wait(0.1)
                    frame, fresh = self._waiting.popleft()
                size = frame.image.nbytes

                # Already overdue, so skip to a newer one. Most scans are quick, so
                # one slow scan doesn't mean the frames behind it will be late too.
                if time.monotonic() > frame.ts + self._delay and self._waiting:
                    with self._lock:
                        self.stats.dropped += 1
                        self._note_in_flight(-1, -size)
                    continue

                started = time.monotonic()
                self.stats.scan_ages.append((started - frame.ts) * 1000)
                detect_before = self.stats.detect_seconds
                output = self._compositor.process(frame, self._timed_detect)
                spent = time.monotonic() - started
                self._scan_estimate += _SCAN_SMOOTHING * (spent - self._scan_estimate)
                self.stats.scanned += 1
                self.stats.scan_seconds += spent
                self.stats.composite_seconds += spent - (
                    self.stats.detect_seconds - detect_before
                )
                self.stats.held = self._compositor.held_frames
                with self._lock:
                    self._safe.append(_SafeFrame(frame.ts, output, frame.frame_id, fresh, size))
                self.stats.slack.append((frame.ts + self._delay - time.monotonic()) * 1000)
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
                self._note_in_flight(-1, -chosen.size)
                behind = sum(1 for safe in self._safe if safe.ts <= due)
                if behind <= _MAX_BEHIND:
                    break
        if chosen is None:
            if current is not None:
                self.stats.repeated += 1
            return current
        self.stats.shown += 1
        self.stats.fresh_shown += chosen.fresh
        if self.stats.last_sent_id >= 0:
            self.stats.id_gaps += max(chosen.frame_id - self.stats.last_sent_id - 1, 0)
        self.stats.last_sent_id = chosen.frame_id
        self.stats.latencies.append((time.monotonic() - chosen.ts) * 1000)
        return chosen.image

    def _note_presented(self) -> None:
        now = time.monotonic()
        if self._last_presented is not None:
            self.stats.present_intervals.append((now - self._last_presented) * 1000)
        self._last_presented = now

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
                    latest = self._next_output(current)
                    is_new = latest is not current
                    current = latest
                    if current is None:
                        # A stop can't wait out a long frame interval.
                        self._stop.wait(min(self._interval / 2, _LONGEST_IDLE))
                    else:
                        self._sink.send(current)
                        if is_new:
                            self._note_presented()
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
