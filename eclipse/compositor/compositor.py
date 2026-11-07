"""Compositor: the last stop before output, and where fail closed is enforced.

Every frame goes through process(). It either returns a frame with all active
detections hidden, or, if anything went wrong, the last safe frame. The raw frame
is never returned unless its scan finished cleanly.

Use one Compositor per source, since boxes are in that source's pixels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from eclipse.types import Detection, Frame

Detector = Callable[[Frame], list[Detection]]

PADDING = 10  # the spec's minimum, in pixels
FILL = (0, 0, 0)  # BGR. Fixed, so a cover carries nothing from the pixels under it.


class Compositor:
    def __init__(self, padding: int = PADDING) -> None:
        self._padding = padding
        self._active: dict[tuple[int, int, int, int], float] = {}  # box -> expiry time
        self._last_safe: np.ndarray | None = None
        self.held_frames = 0

    def process(self, frame: Frame, detect: Detector) -> np.ndarray:
        """Scan a frame and return what is safe to send."""
        try:
            self._remember(frame, detect(frame))
            output = self._redact(frame)
        except Exception as error:
            return self._hold(frame, error)
        self._last_safe = output
        return output

    def _remember(self, frame: Frame, detections: list[Detection]) -> None:
        height, width = frame.image.shape[:2]
        for detection in detections:
            outside = (
                detection.x >= width
                or detection.y >= height
                or detection.x + detection.w <= 0
                or detection.y + detection.h <= 0
            )
            if detection.w <= 0 or detection.h <= 0 or outside:
                # A secret was found but we can't tell where it is.
                raise ValueError("detection box is not inside the frame")
            box = (detection.x, detection.y, detection.w, detection.h)
            expiry = frame.ts + detection.ttl_ms / 1000
            self._active[box] = max(expiry, self._active.get(box, expiry))
        self._active = {box: end for box, end in self._active.items() if end >= frame.ts}

    def covered_areas(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        """Each active box as left, top, right, bottom, padded and clipped to the frame."""
        return [
            (
                max(x - self._padding, 0),
                max(y - self._padding, 0),
                min(x + w + self._padding, width),
                min(y + h + self._padding, height),
            )
            for x, y, w, h in self._active
        ]

    def _redact(self, frame: Frame) -> np.ndarray:
        output = frame.image.copy()
        height, width = output.shape[:2]
        for left, top, right, bottom in self.covered_areas(height, width):
            # A solid fill, because blurred text can sometimes be recovered.
            output[top:bottom, left:right] = FILL
        return output

    def _hold(self, frame: Frame, error: Exception) -> np.ndarray:
        if self.held_frames == 0:
            # Type only: an error message could contain the secret itself.
            print(f"Scan failed ({type(error).__name__}), holding the last safe frame")
        self.held_frames += 1
        if self._last_safe is not None:
            return self._last_safe
        return np.zeros_like(frame.image)
