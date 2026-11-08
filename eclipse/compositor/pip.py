"""Scans and covers a screen with a camera in its corner. See eclipse/capture/dual.py.

The screen and the camera each keep their own detector, as each needs its own idea of
what changed. The camera's scan runs on its own thread, beside the screen's, and the
screen never waits for it unless it must:

- A frame repeating a camera picture already scanned reuses its covers.
- A camera scan still running when the screen's is done doesn't hold the frame up.
  The corner keeps showing the last camera picture that was scanned, with its covers,
  so the screen stays at full rate and only the corner slows down.
- The frame waits only when no scanned picture is fit to show: at the start, after
  new settings or a cut, or once the last one is more than _STALE_SECONDS old.

The camera picture in the corner is always the one its covers were found in. The
detector records which one that is for each frame, and the compositor pastes that.

Plain Frames, from the screen-only and webcam-only views, go to the screen's detector
or the full-size camera's.
"""

from __future__ import annotations

import math
import queue
import threading
from concurrent.futures import Future
from typing import NamedTuple

import numpy as np

from eclipse.capture.dual import DualFrame
from eclipse.compositor.compositor import PADDING, Compositor, Detector
from eclipse.types import Detection, Frame

# How far the corner may fall behind the screen before the screen waits for it.
_STALE_SECONDS = 0.5


class _Worker:
    """One daemon thread running jobs in order. A daemon, so an OCR call stuck at
    shutdown is left behind like the pipeline's own workers, not waited on forever."""

    def __init__(self, name: str) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, work, *args) -> Future:
        future: Future = Future()
        self._jobs.put((future, work, args))
        return future

    def _run(self) -> None:
        while True:
            future, work, args = self._jobs.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(work(*args))
            except BaseException as error:
                future.set_exception(error)


class _Scanned(NamedTuple):
    """A camera picture and everything to cover in it, in its scanned pixels."""

    webcam_id: int
    ts: float  # when the camera took it
    pip: np.ndarray  # the picture to paste
    scan_size: tuple[int, int]  # (height, width) its covers are measured in
    boxes: list[Detection]
    reused: bool  # its scan needed no OCR


class _Job(NamedTuple):
    future: Future
    epoch: int
    frame: DualFrame


class PictureInPictureDetector:
    """Finds what to cover in each view, in the pixels of the picture that goes out."""

    def __init__(
        self, screen: Detector, webcam: Detector, webcam_alone: Detector | None = None
    ) -> None:
        self._screen = screen
        self._webcam = webcam  # the camera at its scanned size, for the corner
        # The camera at full size, for the webcam-only view. Its own, so switching views
        # doesn't make either one start over.
        self._webcam_alone = webcam_alone if webcam_alone is not None else webcam
        self._worker = _Worker("eclipse-webcam-scan")
        self._job: _Job | None = None
        self._scanned: _Scanned | None = None  # the newest camera picture fit to show
        self._epoch = 0  # goes up when old camera results stop counting
        self._shown: tuple[int, _Scanned] | None = None  # (frame_id, what goes in its corner)
        self.reused_read = False
        self._counts = {
            "webcam_scans": 0,  # camera pictures read for the corner
            "webcam_repeats": 0,  # frames whose camera picture was already scanned
            "webcam_behind": 0,  # frames that showed an older camera picture than they had
            "webcam_waits": 0,  # frames that waited for the camera's scan
            "hidden_by_camera": 0,
        }

    @property
    def stats(self) -> dict:
        numbers = dict(getattr(self._screen, "stats", {}))
        for name, value in dict(getattr(self._webcam, "stats", {})).items():
            numbers[f"webcam_{name}"] = value
        if self._webcam_alone is not self._webcam:
            for name, value in dict(getattr(self._webcam_alone, "stats", {})).items():
                numbers[f"webcam_alone_{name}"] = value
        numbers.update(self._counts)
        return numbers

    def prepare(self, frame: Frame) -> bool:
        if frame.source != "screen":
            return True
        prepare = getattr(self._screen, "prepare", None)
        return prepare is None or prepare(frame)

    def _detectors(self) -> list:
        unique = []
        for detector in (self._screen, self._webcam, self._webcam_alone):
            if all(detector is not seen for seen in unique):
                unique.append(detector)
        return unique

    def configure(self, keywords=(), enabled=None) -> None:
        # The same settings pass or fail the same way for all, so checking the screen's
        # first means a bad one changes none.
        keywords = tuple(keywords)
        enabled = None if enabled is None else tuple(enabled)
        for detector in self._detectors():
            configure = getattr(detector, "configure", None)
            if configure is not None:
                configure(keywords, enabled)
        self.forget()  # covers found under the old settings no longer count

    def forget(self) -> None:
        """Stop using camera results found so far, even one still being found."""
        self._epoch += 1
        self._scanned = None
        self._shown = None

    def cancel(self) -> None:
        for detector in self._detectors():
            cancel = getattr(detector, "cancel", None)
            if cancel is not None:
                cancel()

    def shown_camera(self, frame: Frame) -> np.ndarray:
        """The camera picture whose covers this frame got. Raises if it has none."""
        if self._shown is None or self._shown[0] != frame.frame_id:
            raise RuntimeError("no scanned camera picture for this frame")
        return self._shown[1].pip

    def __call__(self, frame: Frame) -> list[Detection]:
        if isinstance(frame, DualFrame) and frame.webcam is not None:
            return self._both(frame)
        detector = self._webcam_alone if frame.source == "webcam" else self._screen
        found = detector(frame)
        self.reused_read = getattr(detector, "reused_read", False)
        return found

    def _both(self, frame: DualFrame) -> list[Detection]:
        self._shown = None
        if self._job is None:
            if self._scanned_is(frame.webcam_id):
                self._counts["webcam_repeats"] += 1
            else:
                self._start(frame)  # runs beside the screen's scan
        error = None
        screen_found: list[Detection] = []
        try:
            screen_found = self._screen(frame)
        except BaseException as failure:
            error = failure
        try:
            camera = self._camera_for(frame)
        except BaseException as failure:
            raise error or failure
        if error is not None:
            raise error
        self._shown = (frame.frame_id, camera)
        self.reused_read = getattr(self._screen, "reused_read", False) and camera.reused
        return self._screen_boxes(frame, screen_found) + self._camera_boxes(frame, camera)

    def _scanned_is(self, webcam_id: int) -> bool:
        return self._scanned is not None and self._scanned.webcam_id == webcam_id

    def _fit_to_show(self, frame: DualFrame) -> bool:
        scanned = self._scanned
        return (
            scanned is not None
            and scanned.pip.shape == frame.pip.shape
            and scanned.ts >= frame.webcam_ts - _STALE_SECONDS
        )

    def _camera_for(self, frame: DualFrame) -> _Scanned:
        """The camera picture and covers to show with this frame, waiting only if none
        already scanned is fit to show."""
        while True:
            if self._job is not None and (self._job.future.done() or not self._fit_to_show(frame)):
                if not self._job.future.done():
                    self._counts["webcam_waits"] += 1
                self._finish_job()  # raises if the scan failed
            if self._fit_to_show(frame):
                if self._scanned.webcam_id != frame.webcam_id:
                    self._counts["webcam_behind"] += 1
                    if self._job is None:
                        self._start(frame)  # catch up from this picture, for later frames
                return self._scanned
            if self._job is None:
                self._start(frame)

    def _start(self, frame: DualFrame) -> None:
        camera = Frame(frame.webcam, frame.webcam_ts, "webcam", frame.webcam_id)
        future = self._worker.submit(self._scan_webcam, camera)
        self._job = _Job(future, self._epoch, frame)

    def _finish_job(self) -> None:
        job, self._job = self._job, None
        boxes = job.future.result()  # raises the scan's error: the frame is held
        if job.epoch != self._epoch:
            return  # found under settings or a view that no longer apply
        frame = job.frame
        self._scanned = _Scanned(
            frame.webcam_id,
            frame.webcam_ts,
            frame.pip,
            frame.webcam.shape[:2],
            boxes,
            getattr(self._webcam, "reused_read", False),
        )

    def _scan_webcam(self, camera: Frame) -> list[Detection]:
        self._counts["webcam_scans"] += 1
        return self._webcam(camera)

    def _screen_boxes(self, frame: DualFrame, found: list[Detection]) -> list[Detection]:
        """Drop covers wholly under the camera, which hides them anyway. Kept, they would
        black out part of a face for as long as the text sat behind it."""
        left, top = frame.pip_at
        height, width = frame.pip.shape[:2]
        kept = []
        for box in found:
            under = (
                box.x >= left
                and box.y >= top
                and box.x + box.w <= left + width
                and box.y + box.h <= top + height
            )
            if under:
                self._counts["hidden_by_camera"] += 1
            else:
                kept.append(box)
        return kept

    def _camera_boxes(self, frame: DualFrame, camera: _Scanned) -> list[Detection]:
        """The camera's covers, moved and scaled into the corner. Rounded outwards."""
        left, top = frame.pip_at
        shown_h, shown_w = camera.pip.shape[:2]
        scan_h, scan_w = camera.scan_size
        across, down = shown_w / scan_w, shown_h / scan_h
        boxes = []
        for box in camera.boxes:
            x1 = max(math.floor(box.x * across), 0)
            y1 = max(math.floor(box.y * down), 0)
            x2 = min(math.ceil((box.x + box.w) * across), shown_w)
            y2 = min(math.ceil((box.y + box.h) * down), shown_h)
            boxes.append(
                Detection(
                    left + x1,
                    top + y1,
                    max(x2 - x1, 1),
                    max(y2 - y1, 1),
                    box.category,
                    box.confidence,
                    box.ttl_ms,
                    frame.frame_id,
                )
            )
        return boxes


class PictureInPictureCompositor(Compositor):
    """Puts the scanned camera picture in its corner, then covers everything found.

    The picture pasted is the one `camera` found the covers in, which may be a little
    older than the frame's own. Pass the same detector the pipeline scans with.
    """

    def __init__(self, camera: PictureInPictureDetector, padding: int = PADDING) -> None:
        super().__init__(padding)
        self._camera = camera

    def forget(self) -> None:
        super().forget()
        self._camera.forget()  # a camera picture from before a cut mustn't come back

    def _canvas(self, frame: Frame) -> np.ndarray:
        output = frame.image.copy()
        if isinstance(frame, DualFrame) and frame.pip is not None:
            pip = self._camera.shown_camera(frame)
            left, top = frame.pip_at
            height, width = pip.shape[:2]
            output[top : top + height, left : left + width] = pip
        return output
