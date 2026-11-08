"""Reads a recorded clip as Frames, so test clips can replace live input."""

from __future__ import annotations

import time

import cv2

from eclipse.types import Frame, Source

_FALLBACK_FPS = 30.0  # for files that don't say their frame rate


class ClipCapture:
    def __init__(self, path: str, source: Source = "screen") -> None:
        self._path = path
        self._source = source
        self._video = None
        self._next_id = 0

    def __enter__(self) -> "ClipCapture":
        self._video = cv2.VideoCapture(self._path)
        if not self._video.isOpened():
            raise FileNotFoundError(f"can't open clip: {self._path}")
        self._fps = self._video.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, *exc) -> None:
        self._video.release()

    def read(self) -> Frame | None:
        """Return the next frame, or None at the end of the clip.

        ts is clip time, so blur timing matches the video however fast we read it.
        """
        ok, image = self._video.read()
        if not ok:
            return None
        frame = Frame(image, self._next_id / self._fps, self._source, self._next_id)
        self._next_id += 1
        return frame


class LiveClip:
    """Plays a recorded clip into the live pipeline as if it were a camera.

    Unlike ClipCapture, which the eval reads as fast as it can, this plays at the
    clip's own speed and stamps each frame with time.monotonic(), the clock the delay
    buffer runs on. Frame ids keep counting up across loops, so a rewind never looks
    like an old frame coming back.
    """

    paced = True  # read() waits for the clip's next frame, so the pipeline mustn't

    def __init__(
        self, path: str, source: Source = "screen", loop: bool = False, clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self._path = path
        self._source = source
        self._loop = loop
        self._clock = clock
        self._sleep = sleep
        self._video = None
        self._next_id = 0
        self._due: float | None = None
        self.interval = 1 / _FALLBACK_FPS
        self.finished = False  # the clip ran out and won't loop: the pipeline stops
        self.loops = 0
        self.stats = {"grab_seconds": 0.0}

    def __enter__(self) -> "LiveClip":
        self._video = cv2.VideoCapture(self._path)
        if not self._video.isOpened():
            raise FileNotFoundError(f"can't open clip: {self._path}")
        fps = self._video.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.interval = 1 / fps
        return self

    def __exit__(self, *exc) -> None:
        if self._video is not None:
            self._video.release()

    def read(self) -> Frame | None:
        """Wait for the clip's next frame time and return that frame.

        Returns None once the clip has ended without --loop, and sets finished.
        """
        if self.finished:
            return None
        self._wait_for_due()
        started = self._clock()
        ok, image = self._video.read()
        if not ok and self._loop:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._video.read()
            if not ok:
                raise RuntimeError(f"clip has no frames: {self._path}")
            self.loops += 1
        self.stats["grab_seconds"] += self._clock() - started
        if not ok or image is None:
            if self._next_id == 0:
                raise RuntimeError(f"clip has no frames: {self._path}")
            self.finished = True
            return None
        height, width = image.shape[:2]
        if width % 2 or height % 2:  # the virtual camera needs even sizes
            image = image[: height & ~1, : width & ~1]
        if self._next_id == 0:
            print(
                f"Playing {self._path} at {image.shape[1]}x{image.shape[0]}, "
                f"{1 / self.interval:.0f} fps{', looping' if self._loop else ''}"
            )
        frame = Frame(image, self._clock(), self._source, self._next_id)
        self._next_id += 1
        return frame

    def _wait_for_due(self) -> None:
        now = self._clock()
        if self._due is None:
            self._due = now
        elif now - self._due > self.interval:
            self._due = now  # fell behind: carry on from here rather than rush to catch up
        else:
            self._sleep(max(self._due - now, 0))
        self._due += self.interval
