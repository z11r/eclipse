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

import cv2
import numpy as np

from eclipse.types import Detection, Frame, Rendered

Detector = Callable[[Frame], list[Detection]]

PADDING = 10  # the spec's minimum, in pixels
FILL = (0, 0, 0)  # BGR. Fixed, so a cover carries nothing from the pixels under it.
HELD = "held"  # an area to show as the last safe picture showed it, not covered black

BLACK, BLUR = "black", "blur"
STYLES = (BLACK, BLUR)  # how a cover looks. Every style hides exactly the same areas.
_RING = 6  # pixels just outside a cover that a blurred cover takes its colours from
_FROST = 0.12  # share of grey mixed in, a frosted-glass tint
_FROST_TO = 128.0  # the grey it is tinted towards
_KNOTS = 8  # colour steps along each edge of a blurred cover, before they are blended
_SMALL = 4  # a blurred cover is drawn at this fraction of its size, then stretched
_LINE = 40  # cover pixels per line of made-up blurred words
_INK_SHARE = 0.6  # how strongly the made-up words show over the background
_INK_LIGHT, _INK_DARK = (215, 215, 215), (45, 45, 45)
_SEED = 20261108  # fixed, so the made-up words are the same every frame


def frosted(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    """A blurred look for a cover, made only from the pixels around it.

    Real blur of what's under a cover can sometimes be undone, so nothing under it is
    used. Colours are taken from a thin ring just outside the cover, which the output
    shows anyway, blended across it, with made-up blurred words on top so it reads as
    blurred text (see _smudge). The image passed must already have
    every cover painted, so a ring that crosses another cover picks up its fill, never
    its secret. With no ring on any side, as for a cover over the whole frame, black.
    """
    height, width = image.shape[:2]
    rows, columns = bottom - top, right - left
    # Worked out on a small grid and stretched to size, as it is only a soft gradient.
    grid_rows, grid_columns = min(rows, _KNOTS), min(columns, _KNOTS)
    above = _profile(image[top - _RING : top, left:right], 0, grid_columns) if top >= _RING else None
    below = _profile(image[bottom : bottom + _RING, left:right], 0, grid_columns) if bottom + _RING <= height else None
    before = _profile(image[top:bottom, left - _RING : left], 1, grid_rows) if left >= _RING else None
    after = _profile(image[top:bottom, right : right + _RING], 1, grid_rows) if right + _RING <= width else None
    fill = np.zeros((grid_rows, grid_columns, 3), np.float32)
    sides = 0
    if above is not None or below is not None:  # each column runs from above to below
        first, last = above if above is not None else below, below if below is not None else above
        share = np.linspace(0.0, 1.0, grid_rows, dtype=np.float32)[:, None, None]
        fill += first[None] * (1 - share) + last[None] * share
        sides += 1
    if before is not None or after is not None:  # each row runs from before to after
        first, last = before if before is not None else after, after if after is not None else before
        share = np.linspace(0.0, 1.0, grid_columns, dtype=np.float32)[None, :, None]
        fill += first[:, None] * (1 - share) + last[:, None] * share
        sides += 1
    if not sides:
        return np.zeros((rows, columns, 3), np.uint8)
    fill /= sides
    # Drawn at a quarter size and stretched, which costs little and softens it further.
    small = max(rows // _SMALL, 1), max(columns // _SMALL, 1)
    background = cv2.resize(fill, small[::-1], interpolation=cv2.INTER_LINEAR)
    words = _smudge(*small, rows)[..., None]
    light = float(fill.mean()) < 128  # light smudges on a dark background, and back
    ink = np.float32(_INK_LIGHT if light else _INK_DARK)
    look = background * (1 - words) + ink * words
    look = np.clip(look * (1 - _FROST) + _FROST_TO * _FROST, 0, 255).astype(np.uint8)
    return cv2.resize(look, (columns, rows), interpolation=cv2.INTER_LINEAR)


def _smudge(small_rows: int, small_columns: int, rows: int) -> np.ndarray:
    """Made-up blurred words, 0 to _INK_SHARE, so a blurred cover looks like blurred text.

    Their shapes come from a fixed seed and the cover's size only, never from the
    screen, so they say nothing about what is hidden. They start at the cover's own
    left edge, so they move with it and stay still while it does.
    """
    lines = max(1, round(rows / _LINE))
    line = small_rows / lines
    margin = line * 0.25
    mask = np.zeros((small_rows, small_columns), np.float32)
    shapes = np.random.default_rng(_SEED)
    for index in range(lines):
        top = int(index * line + line * 0.32)
        bottom = max(int(index * line + line * 0.68), top + 1)
        x, end = margin, small_columns - margin
        while x < end:
            word = line * shapes.uniform(1.0, 3.5)
            mask[top:bottom, int(x) : int(min(x + word, end))] = 1
            x += word + max(line * 0.45, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(line * 0.2, 0.7))
    return mask * _INK_SHARE


def _profile(strip: np.ndarray, axis: int, steps: int) -> np.ndarray:
    """A ring strip's colour along its length, averaged down to a few steps, so no detail
    at the cover's edge, like the tip of a letter, shows up as a streak."""
    profile = strip.mean(axis=axis, dtype=np.float32)[None]  # 1 x length x 3
    return cv2.resize(profile, (steps, 1), interpolation=cv2.INTER_AREA)[0]


class Compositor:
    def __init__(self, padding: int = PADDING, style: str = BLACK) -> None:
        self._padding = padding
        self.style = style
        self._boxes: list[tuple[int, int, int, int]] = []  # this frame's boxes
        self._held: list[tuple[int, int, int, int]] = []  # this frame's held areas
        self._last_safe: Rendered | None = None
        self.held_frames = 0
        self.held_areas = 0  # areas shown as the last safe picture had them: a local stutter

    def process(self, frame: Frame, detect: Detector) -> np.ndarray:
        """Scan a frame and return what is safe to send."""
        return self.render(frame, detect).image

    def render(self, frame: Frame, detect: Detector, generation: int = 0) -> Rendered:
        """Scan a frame and return what is safe to send, labelled with what it shows."""
        try:
            found = detect(frame)
            self._boxes = self._checked(frame, [d for d in found if d.category != HELD])
            self._held = self._checked(frame, [d for d in found if d.category == HELD])
            image = self._redact(frame)
        except Exception as error:
            return self._hold(frame, error, generation)
        state = "cached" if getattr(detect, "reused_read", False) else "scanned"
        self._last_safe = Rendered(
            image, state, frame.source, frame.frame_id, frame.ts, generation
        )
        return self._last_safe

    @property
    def style(self) -> str:
        return self._style

    @style.setter
    def style(self, style: str) -> None:
        """Change how covers look from the next frame on. Never changes what is covered."""
        if style not in STYLES:
            raise ValueError(f"cover style must be one of {STYLES}")
        self._style = style  # one assignment, so a frame mid-render sees one style or the other

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

    def _canvas(self, frame: Frame) -> np.ndarray:
        """A fresh copy of the picture to cover. Subclasses may build it from more parts."""
        return frame.image.copy()

    def _redact(self, frame: Frame) -> np.ndarray:
        output = self._canvas(frame)
        height, width = output.shape[:2]
        style = self._style  # read once, so every cover in this frame looks the same
        areas = self.covered_areas(height, width)
        for left, top, right, bottom in areas:
            # A solid fill, because blurred text can sometimes be recovered.
            output[top:bottom, left:right] = FILL
        if style == BLUR:
            # Painted over the black, from pixels already safe to show: see frosted().
            # Every cover is black first, so no ring can reach a secret under another:
            # it finds black, or a blur already made only from safe pixels.
            for left, top, right, bottom in areas:
                if right > left and bottom > top:
                    output[top:bottom, left:right] = frosted(output, left, top, right, bottom)
        if self._held:
            # Not checked yet, so it shows what the last safe picture showed there: that
            # went out already, so nothing new can. With no such picture, black.
            last = self._last_safe
            usable = last is not None and last.image.shape == output.shape
            pad = self._padding  # as much margin as a cover gets, for faint text edges
            for x, y, w, h in self._held:
                top, left = max(y - pad, 0), max(x - pad, 0)
                bottom, right = min(y + h + pad, height), min(x + w + pad, width)
                output[top:bottom, left:right] = (
                    last.image[top:bottom, left:right] if usable else FILL
                )
            self.held_areas += len(self._held)
        return output

    def _hold(self, frame: Frame, error: Exception, generation: int) -> Rendered:
        if self.held_frames == 0:
            # Type only: an error message could contain the secret itself.
            print(f"Scan failed ({type(error).__name__}), holding the last safe frame")
        self.held_frames += 1
        if self._last_safe is not None:
            return replace(self._last_safe, state="held")
        return Rendered(np.zeros_like(frame.image), "blackout", None, -1, frame.ts, generation)
