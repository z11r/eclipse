"""Delay buffer: smooth output from a scanner that can't always keep up.

Three steps run side by side:

    capture  -> frames waiting to be scanned  (steady rate)
    scanning -> frames safe to send           (as fast as it can)
    output   -> OBS, each frame `delay` late  (steady rate)

A slow scan uses up some of the delay instead of freezing the picture. Output only
ever receives frames from the compositor, so a raw frame has no way to reach it. A
frame that can't be scanned in time is dropped and the last safe frame stays up.

Every picture passes the OutputGate on its way out. Nothing the gate turns down is
ever sent: black goes out instead. Panic and new settings work through the gate, so
they take effect at the output, however far behind the scanner is.

Shutting down closes the gate first, and sends black while the output still works.
Python can't stop a thread stuck inside a native OCR call. If a worker doesn't finish
in time, it is left behind as a daemon, reported in stats.stalled_workers, and the
closed gate stops anything it produces from ever going out.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np

from eclipse.compositor.compositor import Compositor, Detector
from eclipse.compositor.gate import OutputGate
from eclipse.types import Detection, Frame, Rendered


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
_JOIN_SECONDS = 5.0  # how long shutdown waits for the workers
_LONGEST_IDLE = 0.01  # seconds the output waits between looks while it has nothing to send
_NO_FRAME_YET = (1080, 1920, 3)  # the shape of black if a panic beats the first frame


def _samples() -> deque:
    return deque(maxlen=_SAMPLES_KEPT)


class _Queued(NamedTuple):
    frame: Frame
    fresh: bool  # its pixels differ from the frame captured before it


class _SafeFrame(NamedTuple):
    ts: float  # capture time of the frame this tick stands for, which sets when it's due
    rendered: Rendered
    fresh: bool
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
    rendered_scanned: int = 0  # results from an OCR read
    rendered_cached: int = 0  # results that reused earlier reads
    rendered_held: int = 0  # results that fell back on an older safe image
    rendered_blackout: int = 0  # results with nothing safe to show
    invalidated: int = 0  # results or frames thrown out by a panic or stricter settings
    event_errors: int = 0  # catches that couldn't be reported to the panel
    final_black_sent: bool = False
    stalled_workers: tuple[str, ...] = ()


class PolicyChange:
    """A change for the scanner to make between frames. Wait on it to hear how it went."""

    def __init__(self, apply: Callable[[], None], generation: int) -> None:
        self.apply = apply
        self.generation = generation
        self.error: BaseException | None = None
        self._finished = threading.Event()

    def finish(self, error: BaseException | None = None) -> None:
        self.error = error
        self._finished.set()

    def wait(self, seconds: float) -> bool:
        return self._finished.wait(seconds)


class _TimedDetector:
    """Runs the detector, timing it and reporting what it found."""

    def __init__(self, detect: Detector, stats: PipelineStats) -> None:
        self._detect = detect
        self._stats = stats
        self.on_found: Callable[[Frame, list[Detection]], None] | None = None

    @property
    def reused_read(self) -> bool:
        return getattr(self._detect, "reused_read", False)

    def __call__(self, frame: Frame) -> list[Detection]:
        started = time.monotonic()
        try:
            found = self._detect(frame)
        finally:
            self._stats.detect_seconds += time.monotonic() - started
        if self.on_found is not None:
            try:
                self.on_found(frame, found)
            except Exception:
                self._stats.event_errors += 1  # a broken log must not stop the stream
        return found


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
        gate: OutputGate | None = None,
        join_seconds: float = _JOIN_SECONDS,
    ) -> None:
        self.stats = PipelineStats()
        self.gate = gate if gate is not None else OutputGate()
        self.last_output: str | None = None  # scanned, cached, held, blackout or black
        self.output_error: str | None = None  # type name only, as with scan errors
        self.policy_error: str | None = None
        self.last_scan_at: float | None = None
        self._source = source
        self._sink = sink
        self._detect = detect
        self._timed = _TimedDetector(detect, self.stats)
        self._compositor = compositor
        self._delay = delay
        self._interval = 1 / fps
        self._max_buffered = _buffer_limit(delay, fps, max_buffer_bytes)
        self._max_frames = max_frames
        self._join_seconds = join_seconds
        self._waiting: deque[_Queued] = deque()
        self._safe: deque[_SafeFrame] = deque()
        self._last_rows: np.ndarray | None = None
        self._scan_estimate = 0.0  # seconds per scan, smoothed
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._capture_done = threading.Event()
        self._scan_done = threading.Event()
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._threads: list[threading.Thread] = []
        self._pending_policy: PolicyChange | None = None
        self._generation = 0  # the policy the scanner has really applied
        self._invalidations_seen = 0
        self._frame_shape: tuple[int, ...] | None = None
        self._black: np.ndarray | None = None
        self._sent_any = False
        self._presented_id: int | None = None
        self._last_presented: float | None = None

    @property
    def on_found(self):
        return self._timed.on_found

    @on_found.setter
    def on_found(self, callback) -> None:
        self._timed.on_found = callback

    @property
    def running(self) -> bool:
        return self._started.is_set() and not self._stop.is_set() and not self._scan_done.is_set()

    @property
    def sink_connected(self) -> bool:
        return self._sent_any and self.output_error is None

    def change_policy(self, apply: Callable[[], None], stricter: bool) -> PolicyChange:
        """Have the scanner run apply() between frames.

        A stricter change covers the output straight away, without waiting for that
        to happen: nothing checked under the old settings gets out from this moment.
        If apply() fails, the output stays covered until a later change succeeds.
        """
        with self._frame_ready:
            change = PolicyChange(apply, self.gate.advance(stricter))
            replaced, self._pending_policy = self._pending_policy, change
            self._frame_ready.notify_all()
        if replaced is not None:
            replaced.finish()  # a newer change takes its place
        if not self._started.is_set() or self._scan_done.is_set():
            self._apply_pending_policy()  # no scanner running to do it
        return change

    def _apply_pending_policy(self) -> None:
        with self._lock:
            change, self._pending_policy = self._pending_policy, None
        if change is None:
            return
        try:
            change.apply()
        except Exception as error:
            # Type only: an error message could hold a keyword.
            self.policy_error = type(error).__name__
            change.finish(error)
            return
        self._generation = change.generation
        self.policy_error = None
        change.finish()

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
        self._frame_shape = frame.image.shape
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

    def _next_to_scan(self) -> _Queued | None:
        """The next frame to scan, or None once there is nothing more to do."""
        while True:
            self._apply_pending_policy()
            with self._frame_ready:
                if self._stop.is_set():
                    return None
                if self._waiting:
                    return self._waiting.popleft()
                if self._capture_done.is_set():
                    return None
                self._frame_ready.wait(0.1)

    def _scan(self) -> None:
        try:
            while True:
                queued = self._next_to_scan()
                if queued is None:
                    return
                frame, fresh = queued
                size = frame.image.nbytes

                if self.gate.captured_too_early(frame.ts):
                    with self._lock:
                        self.stats.invalidated += 1
                        self._note_in_flight(-1, -size)
                    continue

                # Already overdue, so skip to a newer one. Most scans are quick, so
                # one slow scan doesn't mean the frames behind it will be late too.
                if time.monotonic() > frame.ts + self._delay and self._waiting:
                    with self._lock:
                        self.stats.dropped += 1
                        self._note_in_flight(-1, -size)
                    continue

                invalidations = self.gate.invalidations
                if invalidations != self._invalidations_seen:
                    # Don't let a failed scan fall back on something no longer allowed.
                    self._invalidations_seen = invalidations
                    self._compositor.forget()

                started = time.monotonic()
                self.stats.scan_ages.append((started - frame.ts) * 1000)
                detect_before = self.stats.detect_seconds
                rendered = self._compositor.render(frame, self._timed, self._generation)
                finished = time.monotonic()
                spent = finished - started
                self._scan_estimate += _SCAN_SMOOTHING * (spent - self._scan_estimate)
                self.last_scan_at = finished
                self.stats.scanned += 1
                self.stats.scan_seconds += spent
                self.stats.composite_seconds += spent - (
                    self.stats.detect_seconds - detect_before
                )
                self.stats.held = self._compositor.held_frames
                self._count_result(rendered)
                with self._lock:
                    if self._stop.is_set():
                        self._note_in_flight(-1, -size)  # nobody is left to show it
                        return
                    self._safe.append(_SafeFrame(frame.ts, rendered, fresh, size))
                self.stats.slack.append((frame.ts + self._delay - time.monotonic()) * 1000)
        except BaseException as error:
            self._error = error
            self._stop.set()
        finally:
            self._scan_done.set()

    def _count_result(self, rendered: Rendered) -> None:
        name = f"rendered_{rendered.state}"
        setattr(self.stats, name, getattr(self.stats, name) + 1)

    def _next_output(self, current: _SafeFrame | None) -> _SafeFrame | None:
        """The next safe frame now due, or the current one if none is.

        Frames go out one per tick, so timing jitter doesn't waste them. Older ones
        are only skipped if output has really fallen behind.
        """
        due = time.monotonic() - self._delay
        chosen = None
        with self._lock:
            while self._safe and self._safe[0].ts <= due:
                candidate = self._safe.popleft()
                self._note_in_flight(-1, -candidate.size)
                if self.gate.outdated(candidate.rendered):
                    self.stats.invalidated += 1
                    continue
                if chosen is not None:
                    self.stats.skipped += 1
                chosen = candidate
                behind = sum(1 for safe in self._safe if safe.ts <= due)
                if behind <= _MAX_BEHIND:
                    break
        if chosen is None:
            if current is not None:
                self.stats.repeated += 1
            return current
        return chosen

    def _black_for(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._black is None or self._black.shape != shape:
            self._black = np.zeros(shape, dtype=np.uint8)
        return self._black

    def _submit(self, image: np.ndarray, may_send: Callable[[], bool] | None = None) -> bool:
        """Hand one image to the output. Returns True if black went out in its place."""
        if getattr(self._sink, "checks_before_send", False):
            blacked = bool(self._sink.send(image, may_send))
        else:
            blacked = may_send is not None and not may_send()
            self._sink.send(self._black_for(image.shape) if blacked else image)
        self._sent_any = True
        return blacked

    def _send_black(self, shape: tuple[int, ...] | None = None) -> None:
        ticket = self.gate.black_ticket()
        self._submit(self._black_for(shape or self._frame_shape or _NO_FRAME_YET))
        self.last_output = "black"
        self.gate.confirm_black(ticket)

    def _send_safe(self, safe: _SafeFrame, is_new: bool) -> None:
        rendered = safe.rendered
        if is_new:
            # Before the send, so a sink can tell which frame it is being handed.
            self._note_sending(rendered)
        ticket = 0

        def may_send() -> bool:
            nonlocal ticket
            ticket = self.gate.black_ticket()
            return self.gate.allows(rendered)

        if self._submit(rendered.image, may_send):
            self.last_output = "black"
            self.gate.confirm_black(ticket)
            return
        self.last_output = rendered.state
        if is_new:
            self._note_shown(safe)
        self._note_presented(rendered)

    def _note_sending(self, rendered: Rendered) -> None:
        """Record which source frame is going out. A held image is still its old frame."""
        stats = self.stats
        if rendered.state == "blackout":
            return
        if stats.last_sent_id >= 0:
            stats.id_gaps += max(rendered.source_frame_id - stats.last_sent_id - 1, 0)
        stats.last_sent_id = rendered.source_frame_id

    def _note_shown(self, safe: _SafeFrame) -> None:
        self.stats.shown += 1
        self.stats.fresh_shown += safe.fresh

    def _note_presented(self, rendered: Rendered) -> None:
        """Track what viewers really see. A held image is still the old frame."""
        if rendered.state == "blackout" or rendered.source_frame_id == self._presented_id:
            return
        self._presented_id = rendered.source_frame_id
        now = time.monotonic()
        if self._last_presented is not None:
            self.stats.present_intervals.append((now - self._last_presented) * 1000)
        self._last_presented = now
        self.stats.latencies.append((now - rendered.ts) * 1000)

    def _tick(self, current: _SafeFrame | None) -> _SafeFrame | None:
        """Send one picture, and return the frame to compare the next one with."""
        safe = self._next_output(current)
        if safe is not None and self.gate.allows(safe.rendered):
            self._send_safe(safe, safe is not current)
            return safe
        if safe is None and not self.gate.panicked and not self._sent_any:
            # Nothing has gone out, so nothing needs hiding. A panic can't wait long, though.
            self._stop.wait(min(self._interval / 2, _LONGEST_IDLE))
            return None
        self._send_black(safe.rendered.image.shape if safe is not None else None)
        return None if safe is None or self.gate.outdated(safe.rendered) else safe

    def _finished(self) -> bool:
        with self._lock:
            return self._scan_done.is_set() and not self._safe

    def _drop_queued(self) -> None:
        """Call with the lock held. Work nobody will see is only in the way now."""
        for frame, _ in self._waiting:
            self.stats.dropped += 1
            self._note_in_flight(-1, -frame.image.nbytes)
        for safe in self._safe:
            self._note_in_flight(-1, -safe.size)
        self._waiting.clear()
        self._safe.clear()

    def _send_final_black(self) -> None:
        """Leave black up, while the output still works. Never raises."""
        if not self._sent_any and not self.gate.panicked:
            return  # nothing was ever shown, so there is nothing to replace
        try:
            self._send_black()
            self.stats.final_black_sent = True
        except BaseException as error:
            if self.output_error is None:
                self.output_error = type(error).__name__
            if self._error is None and not isinstance(error, KeyboardInterrupt):
                self._error = error

    def _shut_down(self) -> None:
        self.gate.close()
        self._stop.set()
        with self._frame_ready:
            self._drop_queued()
            self._frame_ready.notify_all()
        cancel = getattr(self._detect, "cancel", None)
        if cancel is not None:
            try:
                cancel()
            except Exception:
                pass  # the closed gate protects the output either way
        self._send_final_black()
        give_up_at = time.monotonic() + self._join_seconds
        for thread in self._threads:
            thread.join(timeout=max(give_up_at - time.monotonic(), 0))
        self.stats.stalled_workers = tuple(
            thread.name for thread in self._threads if thread.is_alive()
        )
        if self.stats.stalled_workers:
            names = ", ".join(self.stats.stalled_workers)
            print(f"Still running after shutdown: {names}. Output stays closed to them.")
        with self._lock:
            self._drop_queued()  # anything a worker added on its way out

    def run(self) -> PipelineStats:
        self._threads = [
            threading.Thread(target=self._capture, name="eclipse-capture", daemon=True),
            threading.Thread(target=self._scan, name="eclipse-scan", daemon=True),
        ]
        current = None
        with self._sink:
            self._started.set()
            for thread in self._threads:
                thread.start()
            try:
                while not self._finished() and not self._stop.is_set():
                    current = self._tick(current)
            except KeyboardInterrupt:
                pass
            except BaseException as error:
                self.output_error = type(error).__name__
                if self._error is None:
                    self._error = error
            finally:
                self._shut_down()
        if self._error is not None:
            raise self._error
        return self.stats
