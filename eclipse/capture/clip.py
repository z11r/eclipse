"""Reads a recorded clip as Frames, so test clips can replace live input."""

from __future__ import annotations

import cv2

from eclipse.types import Frame, Source


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
