"""Output: sends frames to the OBS virtual camera."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pyvirtualcam

# The camera driver rejects some sizes, including 1512x982 (a MacBook screen),
# with "Pixel buffer size mismatch". Always output this one size instead.
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale image to fit width x height, keeping its shape, with black bars."""
    image_height, image_width = image.shape[:2]
    if (image_width, image_height) == (width, height):
        return image
    scale = min(width / image_width, height / image_height)
    new_width = round(image_width * scale)
    new_height = round(image_height * scale)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


class VirtualCam:
    """Opens on the first frame. Call send() with already-redacted images only."""

    def __init__(self, fps: int = 30) -> None:
        self._fps = fps
        self._camera = None
        self._last_image = None
        self._last_fitted = None
        self.stats = {"fit_seconds": 0.0, "send_seconds": 0.0, "fits": 0, "sends": 0}

    def __enter__(self) -> "VirtualCam":
        return self

    def __exit__(self, *exc) -> None:
        if self._camera is not None:
            self._camera.close()

    def send(self, image: np.ndarray) -> None:
        if self._camera is None:
            self._camera = pyvirtualcam.Camera(
                width=OUTPUT_WIDTH,
                height=OUTPUT_HEIGHT,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            print(f"Sending {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} to {self._camera.device}")
        started = time.monotonic()
        if image is not self._last_image:  # a repeated frame is already fitted
            self._last_fitted = fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            self._last_image = image
            self.stats["fits"] += 1
        fitted = time.monotonic()
        self._camera.send(self._last_fitted)
        self.stats["fit_seconds"] += fitted - started
        self.stats["send_seconds"] += time.monotonic() - fitted
        self.stats["sends"] += 1
        self._camera.sleep_until_next_frame()
