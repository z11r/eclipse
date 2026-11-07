"""Screen capture: turns the display into a stream of Frames."""

from __future__ import annotations

import time

import cv2
import mss
import numpy as np

from eclipse.platform import use_full_screen_resolution
from eclipse.types import Frame


def to_bgr(raw: np.ndarray, max_width: int) -> np.ndarray:
    """Convert an mss BGRA grab to BGR, shrinking it to fit max_width.

    Dimensions are rounded down to even numbers for the virtual camera.
    """
    height, width = raw.shape[:2]
    scale = min(1.0, max_width / width)
    new_width = int(width * scale) & ~1
    new_height = int(height * scale) & ~1
    if scale == 1.0:
        width, height = new_width, new_height  # keep even sizes without resizing
    image = cv2.cvtColor(raw[:height, :width], cv2.COLOR_BGRA2BGR)  # far faster than slicing
    if (new_width, new_height) == (width, height):
        return image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


class ScreenCapture:
    def __init__(self, monitor: int = 1, max_width: int = 1920, retina: bool = False) -> None:
        self._monitor = monitor
        self._max_width = max_width
        self._retina = retina
        self.stats = {"grab_seconds": 0.0, "resize_seconds": 0.0}
        self._next_id = 0
        self._mss = None

    def __enter__(self) -> "ScreenCapture":
        if self._retina:
            use_full_screen_resolution()
        self._mss = mss.MSS()
        return self

    def __exit__(self, *exc) -> None:
        self._mss.close()

    def read(self) -> Frame:
        ts = time.monotonic()
        raw = np.asarray(self._mss.grab(self._mss.monitors[self._monitor]))
        grabbed = time.monotonic()
        frame = Frame(to_bgr(raw, self._max_width), ts, "screen", self._next_id)
        self.stats["grab_seconds"] += grabbed - ts
        self.stats["resize_seconds"] += time.monotonic() - grabbed
        if self._next_id == 0:
            height, width = frame.image.shape[:2]
            print(f"Capturing {raw.shape[1]}x{raw.shape[0]}, scanning {width}x{height}")
        self._next_id += 1
        return frame
