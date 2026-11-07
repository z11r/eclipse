"""Compositor: the last stop before output, and where fail closed is enforced.

Every frame goes through render(). It either returns a frame with that frame's
detections hidden, or, if anything went wrong, the last safe frame, still labelled as
the frame it came from. The raw frame is never returned unless its scan finished
cleanly.

The compositor keeps no boxes between frames. A screen position goes stale as soon
as anything moves, and holding old ones paints trails behind moving text. A detector
must report everything to cover on every frame, and it decides how long a cover
lasts, because it can tell where the text went.

Use one Compositor per source, since boxes are in that source's pixels.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from eclipse.types import Detection, Frame, Rendered

Detector = Callable[[Frame], list[Detection]]

PADDING = 10  # the spec's minimum, in pixels
FILL = (0, 0, 0)  # BGR. Fixed, so a cover carries nothing from the pixels under it.


class Compositor:
    def __init__(self, padding: int = PADDING) -> None:
        self._padding = padding
        self._boxes: list[tuple[int, int, int, int]] = []  # this frame's boxes
        self._last_safe: Rendered | None = None
        self.held_frames = 0

    def process(self, frame: Frame, detect: Detector) -> np.ndarray:
        """Scan a frame and return what is safe to send."""
        return self.render(frame, detect).image

    def render(self, frame: Frame, detect: Detector, generation: int = 0) -> Rendered:
        """Scan a frame and return what is safe to send, labelled with what it shows."""
        try:
            self._boxes = self._checked(frame, detect(frame))
            image = self._redact(frame)
        except Exception as error:
            return self._hold(frame, error, generation)
        state = "cached" if getattr(detect, "reused_read", False) else "scanned"
        self._last_safe = Rendered(
            image, state, frame.source, frame.frame_id, frame.ts, generation
        )
        return self._last_safe

    def forget(self) -> None:
        """Stop holding the last safe image, as it may no longer be safe to show."""
        self._last_safe = None

    def _checked(
        self, frame: Frame, detections: list[Detection]
    ) -> list[tuple[int, int, int, int]]:
        height, width = frame.image.shape[:2]
        boxes = []
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
            if not detection.ttl_ms >= 0:  # also catches NaN
                raise ValueError("detection ttl_ms must not be negative")
            if detection.frame_id != frame.frame_id:
                # Its position belongs to some other picture.
                raise ValueError("detection is from a different frame")
            boxes.append((detection.x, detection.y, detection.w, detection.h))
        return boxes

    def covered_areas(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        """Each box as left, top, right, bottom, padded and clipped to the frame."""
        return [
            (
                max(x - self._padding, 0),
                max(y - self._padding, 0),
                min(x + w + self._padding, width),
                min(y + h + self._padding, height),
            )
            for x, y, w, h in self._boxes
        ]

    def _redact(self, frame: Frame) -> np.ndarray:
        output = frame.image.copy()
        height, width = output.shape[:2]
        for left, top, right, bottom in self.covered_areas(height, width):
            # A solid fill, because blurred text can sometimes be recovered.
            output[top:bottom, left:right] = FILL
        return output

    def _hold(self, frame: Frame, error: Exception, generation: int) -> Rendered:
        if self.held_frames == 0:
            # Type only: an error message could contain the secret itself.
            print(f"Scan failed ({type(error).__name__}), holding the last safe frame")
        self.held_frames += 1
        if self._last_safe is not None:
            return replace(self._last_safe, state="held")
        return Rendered(np.zeros_like(frame.image), "blackout", None, -1, frame.ts, generation)
