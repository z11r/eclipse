"""Screen and webcam at once, sent as one picture with the camera in a corner.

Each input reads on its own thread, so a camera waiting for its next picture never
holds up the screen, and a still screen never holds up the camera. A new frame goes
out whenever an input in the current view has something new.

The camera is not pasted in here. A DualFrame carries the raw screen as its image,
plus the camera picture to scan and the same picture at its size on screen. The text
layer reads each one by itself, in its own pixels and with its own noise floor, and
the compositor puts them together only after both are covered. So a face moving in
the corner never makes the screen's text be read again, and a still screen behind it
costs nothing.

Both inputs keep running whatever the view, so switching between the screen alone,
both, and the camera alone takes effect on the next frame. See set_view().
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np

from eclipse.types import Frame

View = Literal["screen", "both", "webcam"]
VIEWS: tuple[View, ...] = ("screen", "both", "webcam")
# What each view shows. A switch to a view that drops one must cut off older frames.
VIEW_SOURCES = {"screen": {"screen"}, "both": {"screen", "webcam"}, "webcam": {"webcam"}}

PIP_SIZE = 0.25  # the camera's width, as a share of the screen's
_MARGIN = 0.02  # gap between the camera and the screen's edges, as a share of its width
# The camera is read at twice the size it is shown at. Text too small to read there is
# too small for a viewer to read in the corner, and a smaller picture reads faster.
_SCAN_OVER_SHOWN = 2
_READ_WAIT = 0.1  # seconds read() waits for news, so a still picture can't block shutdown
_START_WAIT = 30.0  # seconds a camera gets to open; Windows cameras can take over 10


def drops_a_source(old: View, new: View) -> bool:
    """Whether switching from old to new stops showing something old showed."""
    return not VIEW_SOURCES[old] <= VIEW_SOURCES[new]


@dataclass(frozen=True, eq=False)
class DualFrame(Frame):
    """A screen frame, with the camera picture that goes in its corner.

    image is the screen alone, so anything reading a Frame reads the screen. The
    camera pictures are shared by every DualFrame made while the camera stood still:
    treat them as read-only too.
    """

    webcam: np.ndarray | None = None  # the camera picture to scan
    pip: np.ndarray | None = None  # the same picture, at its size on screen
    pip_at: tuple[int, int] = (0, 0)  # where its top left corner goes on the screen
    webcam_id: int = -1  # which camera picture this is; repeats while the camera waits
    webcam_ts: float = 0.0  # when the camera took it


def pip_layout(
    screen_shape: tuple[int, ...], camera_shape: tuple[int, ...], size: float,
    output_size: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Where the camera goes, at what size, and at what size to scan it.

    Returns ((left, top), (width, height) shown, (width, height) scanned). The
    camera sits in the bottom right corner and keeps its own shape.
    """
    if output_size is not None and min(output_size) <= 0:
        raise ValueError("output_size dimensions must be positive")
    screen_h, screen_w = screen_shape[:2]
    camera_h, camera_w = camera_shape[:2]
    shown_w = max(min(int(screen_w * size), screen_w), 2)
    shown_h = max(round(shown_w * camera_h / camera_w), 2)
    if shown_h > screen_h:  # a tall camera on a wide screen
        shown_h = screen_h
        shown_w = max(round(shown_h * camera_w / camera_h), 2)
    margin = int(screen_w * _MARGIN)
    left = max(screen_w - shown_w - margin, 0)
    top = max(screen_h - shown_h - margin, 0)
    # Only the final virtual-camera pixels need oversampling. Retina capture
    # dimensions can be much larger than that output. None retains old sizing.
    output_scale = 1.0 if output_size is None else min(
        1.0, output_size[0] / screen_w, output_size[1] / screen_h
    )
    scan_w = min(camera_w, max(2, math.ceil(shown_w * output_scale * _SCAN_OVER_SHOWN)))
    scan_h = min(camera_h, max(2, round(scan_w * camera_h / camera_w)))
    return (left, top), (shown_w, shown_h), (scan_w, scan_h)


class _Feed:
    """Reads one source on its own thread and keeps the newest frame."""

    def __init__(self, source, name: str, interval: float, news: threading.Condition) -> None:
        self.source = source
        self.frame: Frame | None = None
        self.count = 0  # frames taken so far; read() compares it to spot news
        self.failure: BaseException | None = None
        self.finished = False
        self._interval = interval
        self._news = news
        self._entered = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        # Opened on its own thread: some capture libraries only work on the thread
        # that opened them.
        self._thread.start()

    def wait_open(self, seconds: float) -> None:
        if not self._entered.wait(seconds) and self.failure is None:
            raise RuntimeError(f"{self._thread.name} didn't open in time")
        if self.failure is not None:
            raise self.failure

    def stop(self, seconds: float) -> None:
        self._stop.set()
        self._thread.join(timeout=seconds)

    def _run(self) -> None:
        try:
            with self.source:
                self._entered.set()
                due = time.monotonic()
                while not self._stop.is_set():
                    frame = self.source.read()
                    if frame is not None:
                        with self._news:
                            self.frame = frame
                            self.count += 1
                            self._news.notify_all()
                    elif getattr(self.source, "finished", False):
                        with self._news:
                            self.finished = True
                            self._news.notify_all()
                        return
                    if not getattr(self.source, "paced", False):
                        # A polling source returns at once, so keep it to the frame rate.
                        due = max(due + self._interval, time.monotonic())
                        self._stop.wait(max(due - time.monotonic(), 0))
        except BaseException as error:
            with self._news:
                self.failure = error
                self._news.notify_all()
        finally:
            self._entered.set()


class DualCapture:
    """Captures a screen and a camera together, sent as the current view.

    `screen` is any screen source (or a clip standing in for one), `webcam` any camera
    source (or a clip standing in for one). If either fails, read() raises, so the
    pipeline stops on black, as it does for one source.

    The views: "both" makes DualFrames, "screen" plain screen Frames, and "webcam"
    plain camera Frames at the camera's full size, tagged "webcam".
    """

    paced = True  # read() waits for news from either input, so the pipeline mustn't
    views = VIEWS

    def __init__(
        self, screen, webcam, pip_size: float = PIP_SIZE, fps: int = 60, view: View = "both",
        *, scan_output_size: tuple[int, int] | None = None,
    ) -> None:
        if not 0 < pip_size <= 1:
            raise ValueError("pip_size must be above 0 and at most 1")
        if view not in VIEWS:
            raise ValueError(f"view must be one of {', '.join(VIEWS)}")
        self._pip_size = pip_size
        self._scan_output_size = scan_output_size
        self._view: View = view
        self._refresh = False  # send the current pictures once, even with no news
        self._refresh_from = -math.inf  # the earliest time the cut allows
        self._news = threading.Condition()
        self._screen = _Feed(screen, "eclipse-screen", 1 / fps, self._news)
        self._webcam = _Feed(webcam, "eclipse-webcam", 1 / fps, self._news)
        self._seen = (0, 0)  # feed counts as of the last frame made
        self._next_id = 0
        self._layout_for = None
        self._layout = None
        self._camera_for = -1
        self._camera = (None, None)  # (scanned, shown) pictures of the newest camera frame
        self.finished = False
        self._stats = {
            "webcam_resize_seconds": 0.0,
            "webcam_frames": 0,
            "screen_frames": 0,
            "view_switches": 0,
        }

    @property
    def view(self) -> View:
        with self._news:
            return self._view

    def set_view(self, view: View, cut: Callable[[], float] | None = None) -> None:
        """Show this view from the next frame on.

        cut, if given, runs once no frame of the old view can be made any more, and
        must stop those already made from going out, returning the earliest capture time
        still allowed: pass the gate's cut() when the new view drops a source. The current pictures go out at once, so a still screen
        can't leave the output waiting.
        """
        if view not in VIEWS:
            raise ValueError(f"view must be one of {', '.join(VIEWS)}")
        with self._news:
            # read() takes the view and its pictures under this lock, so every frame of
            # the old view was taken before this point, and before the cut.
            self._view = view
            self._refresh_from = cut() if cut is not None else -math.inf
            self._refresh = True
            self._stats["view_switches"] += 1
            self._news.notify_all()

    @property
    def stats(self) -> dict:
        numbers = dict(getattr(self._screen.source, "stats", {}))
        for name, value in dict(getattr(self._webcam.source, "stats", {})).items():
            numbers[f"webcam_{name}"] = value
        numbers.update(self._stats)
        return numbers

    def __enter__(self) -> "DualCapture":
        self._screen.start()
        self._webcam.start()
        try:
            self._screen.wait_open(_START_WAIT)
            self._webcam.wait_open(_START_WAIT)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._screen.stop(2.0)
        self._webcam.stop(2.0)

    def read(self) -> Frame | None:
        """The newest pictures in the current view, or None if nothing in it is new."""
        give_up_at = time.monotonic() + _READ_WAIT
        with self._news:
            while True:
                for feed in (self._screen, self._webcam):
                    if feed.failure is not None:
                        raise RuntimeError(f"{feed.source.__class__.__name__} stopped") from (
                            feed.failure
                        )
                counts = (self._screen.count, self._webcam.count)
                if self._screen.finished and counts[0] == self._seen[0]:
                    self.finished = True  # a clip standing in for the screen ran out
                    return None
                view = self._view
                wanted = VIEW_SOURCES[view]
                new_screen = counts[0] != self._seen[0]
                new_camera = counts[1] != self._seen[1]
                ready = ("screen" not in wanted or self._screen.frame is not None) and (
                    "webcam" not in wanted or self._webcam.frame is not None
                )
                news = ("screen" in wanted and new_screen) or ("webcam" in wanted and new_camera)
                if ready and (news or self._refresh):
                    break
                left = give_up_at - time.monotonic()
                if left <= 0:
                    return None
                self._news.wait(left)
            screen, camera = self._screen.frame, self._webcam.frame
            # The first frame after a switch shows the new view's pictures as they are
            # now, which is after any cut, so the time is taken here, under the lock. Its
            # inputs are the newest there are, so nothing on them has changed since.
            now = max(time.monotonic(), self._refresh_from) if self._refresh else None
            self._refresh = False
            self._seen = counts
        self._stats["screen_frames"] += new_screen
        self._stats["webcam_frames"] += new_camera
        if view == "screen":
            return self._single(screen, now if now is not None else screen.ts)
        if view == "webcam":
            return self._single(camera, now if now is not None else camera.ts)
        return self._combine(screen, camera, new_screen, new_camera, now)

    def _single(self, frame: Frame, ts: float) -> Frame:
        single = Frame(frame.image, ts, frame.source, self._next_id)
        self._next_id += 1
        return single

    def _combine(
        self,
        screen: Frame,
        camera: Frame,
        new_screen: bool,
        new_camera: bool,
        ts: float | None,
    ) -> DualFrame:
        shapes = (screen.image.shape, camera.image.shape)
        if shapes != self._layout_for:
            self._layout_for = shapes
            self._layout = pip_layout(*shapes, self._pip_size, self._scan_output_size)
            self._camera_for = -1
            at, shown, scanned = self._layout
            print(
                f"Camera {camera.image.shape[1]}x{camera.image.shape[0]} shown at "
                f"{shown[0]}x{shown[1]} in the corner, scanned at {scanned[0]}x{scanned[1]}"
            )
        at, shown, scanned = self._layout
        if camera.frame_id != self._camera_for:
            # Once per camera picture, here on the capture thread, never on the scanner's.
            started = time.monotonic()
            scan = camera.image
            if (scan.shape[1], scan.shape[0]) != scanned:
                scan = cv2.resize(scan, scanned, interpolation=cv2.INTER_AREA)
            pip = cv2.resize(scan, shown, interpolation=cv2.INTER_AREA)
            scan.flags.writeable = False
            pip.flags.writeable = False
            self._camera = (scan, pip)
            self._camera_for = camera.frame_id
            self._stats["webcam_resize_seconds"] += time.monotonic() - started
        scan, pip = self._camera
        if ts is None:
            # The age of whatever is new in this frame. A part that hasn't changed is
            # still what its source shows now, so its old capture time says nothing.
            parts = ((screen, new_screen), (camera, new_camera))
            ts = min(frame.ts for frame, new in parts if new)
        frame = DualFrame(
            screen.image,
            ts,
            screen.source,
            self._next_id,
            webcam=scan,
            pip=pip,
            pip_at=at,
            webcam_id=camera.frame_id,
            webcam_ts=camera.ts,
        )
        self._next_id += 1
        return frame
