"""Webcam capture: turns a camera into a stream of Frames."""

from __future__ import annotations

import time

import cv2

from eclipse.platform import camera_backend
from eclipse.types import Frame


class WebcamCapture:
    """Reads one camera. Pick the real camera, not OBS Virtual Camera, or the
    output loops back into the input."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._size = (width, height)
        self._video = None
        self._next_id = 0
        self.stats = {"grab_seconds": 0.0}

    def __enter__(self) -> "WebcamCapture":
        self._video = cv2.VideoCapture(self._index, camera_backend())
        if not self._video.isOpened():
            raise RuntimeError(
                f"can't open camera {self._index}. Check it isn't in use by another app, "
                "and on macOS that Camera access is allowed in Privacy & Security"
            )
        self._video.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        self._video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, not a stale one
        return self

    def __exit__(self, *exc) -> None:
        if self._video is not None:
            self._video.release()

    def read(self) -> Frame:
        """Return the newest camera frame. Raises if the camera stops, since the
        pipeline expects a Frame every time."""
        ts = time.monotonic()
        ok, image = self._video.read()
        self.stats["grab_seconds"] += time.monotonic() - ts
        if not ok or image is None:
            raise RuntimeError(f"camera {self._index} stopped sending frames")
        height, width = image.shape[:2]
        if width % 2 or height % 2:  # the virtual camera needs even sizes
            image = image[: height & ~1, : width & ~1]
        if self._next_id == 0:
            print(f"Capturing camera {self._index} at {image.shape[1]}x{image.shape[0]}")
        frame = Frame(image, ts, "webcam", self._next_id)
        self._next_id += 1
        return frame
