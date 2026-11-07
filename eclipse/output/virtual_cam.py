"""Output: sends frames to the OBS virtual camera."""

from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np
import pyvirtualcam

# The camera driver rejects some sizes, including 1512x982 (a MacBook screen),
# with "Pixel buffer size mismatch". Always output this one size instead.
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080


def scale_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize, shrinking in two linear steps through 1.5 times the final size.

    This has to be cheap on one thread. OpenCV runs one parallel job per process, so when
    another thread is mid-call, a resize quietly runs alone. INTER_AREA then takes 48 ms and
    misses two output ticks. Two linear steps take 1.9 ms, stay closest to INTER_AREA
    (35.5 dB), and leave as much small text readable.
    """
    middle = (width * 3 // 2, height * 3 // 2)
    if middle[0] < image.shape[1] and middle[1] < image.shape[0]:
        image = cv2.resize(image, middle, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale image to fit width x height, keeping its shape, with black bars."""
    image_height, image_width = image.shape[:2]
    if (image_width, image_height) == (width, height):
        return image
    scale = min(width / image_width, height / image_height)
    new_width = round(image_width * scale)
    new_height = round(image_height * scale)
    resized = scale_to(image, new_width, new_height)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


class _NoCamera:
    device = "nowhere"

    def send(self, image: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass


class VirtualCam:
    """Opens on the first frame. Call send() with already-redacted images only."""

    checks_before_send = True  # send() takes a last check, for callers that want one

    def __init__(self, fps: int = 30) -> None:
        self._fps = fps
        self._interval = 1 / fps
        self._next_tick: float | None = None
        self._camera = None
        self._last_image = None
        self._last_fitted = None
        self.stats = {
            "fit_seconds": 0.0,
            "send_seconds": 0.0,
            "fits": 0,
            "sends": 0,
            "late_ticks": 0,
        }

    def __enter__(self) -> "VirtualCam":
        return self

    def __exit__(self, *exc) -> None:
        if self._camera is not None:
            self._camera.close()

    def send(self, image: np.ndarray, may_send: Callable[[], bool] | None = None) -> bool:
        """Send one frame, and return True if black went out in its place.

        Fitting and waiting for the tick both take time, and a panic can arrive during
        either. So may_send is asked as the very last step before the frame leaves.
        """
        if self._camera is None:
            self._camera = self._open_camera()
            print(f"Sending {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} to {self._camera.device}")
        started = time.monotonic()
        if image is not self._last_image:  # a repeated frame is already fitted
            self._last_fitted = fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            self._last_image = image
            self.stats["fits"] += 1
        fitted = time.monotonic()
        self._wait_for_tick(fitted)
        blacked = may_send is not None and not may_send()
        sending = time.monotonic()
        self._camera.send(np.zeros_like(self._last_fitted) if blacked else self._last_fitted)
        self.stats["fit_seconds"] += fitted - started
        self.stats["send_seconds"] += time.monotonic() - sending
        self.stats["sends"] += 1
        return blacked

    def _open_camera(self):
        return pyvirtualcam.Camera(
            width=OUTPUT_WIDTH,
            height=OUTPUT_HEIGHT,
            fps=self._fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )

    def _wait_for_tick(self, now: float) -> None:
        """Hold until the next output tick, on a fixed clock.

        Fitting costs time for a new picture and nothing for a repeat, so timing from
        the end of the last send would make the frame intervals uneven. After a stall the
        clock restarts from now, so frames never leave in a burst to catch up.
        """
        if self._next_tick is None:
            self._next_tick = now
        if now < self._next_tick:
            time.sleep(self._next_tick - now)
        self._next_tick += self._interval
        if now > self._next_tick:
            self.stats["late_ticks"] += 1
            self._next_tick = now + self._interval


class DiscardingCam(VirtualCam):
    """Does all of VirtualCam's work but sends nothing, for timing runs on a real screen."""

    def _open_camera(self):
        return _NoCamera()
