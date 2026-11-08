"""Output: sends frames to the OBS virtual camera."""

from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np
import pyvirtualcam

from eclipse.platform import stamped_virtual_camera

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


def bgr_to_uyvy(image: np.ndarray) -> np.ndarray:
    """BT.601 video range UYVY, as rows of width * 2 bytes.

    The same colors pyvirtualcam makes with libyuv (within 1 in 255), so switching
    camera paths doesn't change what viewers see. About 4.4 ms for 1080p.
    """
    height, width = image.shape[:2]
    luma = cv2.transform(image, _LUMA)
    pairs = cv2.resize(image, (width // 2, height), interpolation=cv2.INTER_AREA)
    chroma = cv2.transform(pairs, _CHROMA)
    left, right = cv2.split(luma.reshape(height, width // 2, 2))
    blue, red = cv2.split(chroma)
    return cv2.merge([blue, left, red, right]).reshape(height, width * 2)


# Rows are B, G, R weights, then the offset. cv2.transform rounds to nearest.
_LUMA = np.array([[25, 129, 66, 16 * 256]], np.float32) / 256
_CHROMA = np.array([[112, -74, -38, 128 * 256], [-18, -94, 112, 128 * 256]], np.float32) / 256


class PyVirtualCamera:
    """pyvirtualcam, which converts each frame and stamps it with the moment it leaves."""

    def __init__(self, width: int, height: int, fps: float) -> None:
        self._camera = pyvirtualcam.Camera(
            width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR
        )
        self.device = self._camera.device

    def prepare(self, image: np.ndarray) -> np.ndarray:
        return image

    def send(self, image: np.ndarray, due: float) -> None:
        self._camera.send(image)

    def close(self) -> None:
        self._camera.close()


def open_camera(width: int, height: int, fps: float):
    """OBS Virtual Camera. Where we can, each frame is stamped with the tick it is due.

    OBS's Video Capture Device paces frames by those stamps when its Use Buffering box
    is ticked. Without it, OBS shows whatever arrived last at each of its own 60 Hz
    ticks, and frames that land near one of them are doubled or skipped at random.
    """
    return stamped_virtual_camera(width, height) or PyVirtualCamera(width, height, fps)


class _NoCamera:
    device = "nowhere"
    prepare = staticmethod(bgr_to_uyvy)  # the work the macOS camera does before each tick

    def send(self, image: np.ndarray, due: float) -> None:
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
        self._black = None
        self._last_image = None
        self._last_prepared = None
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
            black = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            self._black = self._camera.prepare(black)
            print(f"Sending {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} to {self._camera.device}")
        started = time.monotonic()
        if image is not self._last_image:  # a repeated frame is already prepared
            fitted = fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            self._last_prepared = self._camera.prepare(fitted)
            self._last_image = image
            self.stats["fits"] += 1
        prepared = time.monotonic()
        due = self._wait_for_tick(prepared)
        blacked = may_send is not None and not may_send()
        sending = time.monotonic()
        self._camera.send(self._black if blacked else self._last_prepared, due)
        self.stats["fit_seconds"] += prepared - started
        self.stats["send_seconds"] += time.monotonic() - sending
        self.stats["sends"] += 1
        return blacked

    def _open_camera(self):
        return open_camera(OUTPUT_WIDTH, OUTPUT_HEIGHT, self._fps)

    def _wait_for_tick(self, now: float) -> float:
        """Hold until the next output tick, on a fixed clock, and return the tick's time.

        Preparing costs time for a new picture and nothing for a repeat, so timing from
        the end of the last send would make the frame intervals uneven. After a stall
        that misses a whole tick, the frame waits for the next tick on the same grid, so
        frames never leave in a burst to catch up. Keeping one grid matters to OBS: it
        paces frames by these times, and a grid that jumped would eat into the margin
        that absorbs late wake-ups.
        """
        if self._next_tick is None:
            self._next_tick = now
        if now > self._next_tick + self._interval:
            self.stats["late_ticks"] += 1
            missed = int((now - self._next_tick) / self._interval) + 1
            self._next_tick += missed * self._interval
        if now < self._next_tick:
            time.sleep(self._next_tick - now)
        due = self._next_tick
        self._next_tick += self._interval
        return due


class DiscardingCam(VirtualCam):
    """Does all of VirtualCam's work but sends nothing, for timing runs on a real screen."""

    def _open_camera(self):
        return _NoCamera()
