"""A synthetic scrolling page with planted fake secrets, and OCR that only sees pixels.

Every line of text is a row of solid blocks in the line's own colour. SceneReader
works out what it was handed from those colours alone, so like real OCR it can't read
a line that is cut off, or outside the crop it was given. That lets tests drive the
real detector, tracks and compositor with no OCR model and no randomness.

Secrets here are fake. Never plant a real one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from eclipse.types import Frame, Source

BACKGROUND = (30, 30, 30)
LINE_HEIGHT = 20
LINE_PITCH = 40
BLOCK_WIDTH, BLOCK_STEP = 30, 34
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOUR = (235, 235, 235)
WORDS = "notes on the plan for friday review each draft before we ship it to team".split()
FAKE_EMAIL = "demo@example.com"
FAKE_KEY = "sk-FAKE-KEY-1234567890"
FAKE_PHONE = "(780) 555-0123"


@dataclass(frozen=True)
class Line:
    text: str
    x: int
    y: int  # in the document, or on screen for the side pane
    w: int
    h: int
    colour: tuple[int, int, int]


def _colour(index: int) -> tuple[int, int, int]:
    return 200, 100 + index, 255


def _ends_on_a_block(width: int) -> int:
    return width // BLOCK_STEP * BLOCK_STEP + BLOCK_WIDTH


class ScrollScene:
    """A scrolling document, with a still side pane to its right.

    Lines are coloured blocks for SceneReader, or real text for real OCR if font_scale
    is set.
    """

    def __init__(
        self,
        secrets: dict[int, str] | None = None,
        side_secrets: dict[int, str] | None = None,
        width: int = 960,
        height: int = 540,
        pane: tuple[int, int] = (40, 640),
        side: tuple[int, int] = (680, 940),
        lines: int = 90,
        font_scale: float = 0.0,
        pitch: int = LINE_PITCH,
    ) -> None:
        self.width, self.height = width, height
        self._font_scale = font_scale
        self._thickness = max(round(font_scale * 2), 1)
        self.pane, self.side = pane, side
        secrets = {6: FAKE_EMAIL, 30: FAKE_KEY} if secrets is None else secrets
        side_secrets = {3: FAKE_PHONE} if side_secrets is None else side_secrets
        self.secrets = {*secrets.values(), *side_secrets.values()}
        rng = np.random.default_rng(7)
        self.lines = []
        for index in range(lines):
            words = " ".join(WORDS[(index * 5 + step * 3) % len(WORDS)] for step in range(6))
            text = secrets.get(index, f"{index} {words}" if font_scale else f"line {index}")
            limit = pane[1] - pane[0] - 100
            self.lines.append(self._line(text, 20, 20 + index * pitch, limit, index, rng))
        self.side_lines = [
            self._line(
                side_secrets.get(index, f"side note {index}"),
                side[0] + 10,
                20 + index * pitch,
                side[1] - side[0] - 60,
                lines + index,
                rng,
            )
            for index in range(10)
        ]
        self.document = np.full(
            (self.lines[-1].y + height + pitch, pane[1] - pane[0], 3), BACKGROUND, np.uint8
        )
        for line in self.lines:
            self._draw(self.document, line, line.x)
        self._side_image = np.full((height, width, 3), BACKGROUND, np.uint8)
        for line in self.side_lines:
            self._draw(self._side_image, line, line.x)
        self._by_colour = {
            (colour[0] << 16) | (colour[1] << 8) | colour[2]: line
            for line in [*self.lines, *self.side_lines]
            for colour in [line.colour]
        }

    def _line(self, text: str, x: int, y: int, limit: int, index: int, rng) -> Line:
        if self._font_scale:
            (width, height), baseline = cv2.getTextSize(
                text, FONT, self._font_scale, self._thickness
            )
            return Line(text, x, y, width, height + baseline, TEXT_COLOUR)
        width = _ends_on_a_block(int(rng.integers(min(limit, 120), limit)))
        return Line(text, x, y, width, LINE_HEIGHT, _colour(index))

    def _draw(self, image: np.ndarray, line: Line, x: int) -> None:
        if self._font_scale:
            (_, height), _ = cv2.getTextSize(line.text, FONT, self._font_scale, self._thickness)
            cv2.putText(
                image,
                line.text,
                (x, line.y + height),
                FONT,
                self._font_scale,
                line.colour,
                self._thickness,
                cv2.LINE_AA,
            )
            return
        # Blocks with gaps, so that lines differ from row to row like text does.
        for left in range(x, x + line.w, BLOCK_STEP):
            image[line.y : line.y + line.h, left : left + BLOCK_WIDTH] = line.colour

    def image(self, offset: int) -> np.ndarray:
        image = self._side_image.copy()
        image[:, self.pane[0] : self.pane[1]] = self.document[offset : offset + self.height]
        return image

    def frame(
        self, offset: int, frame_id: int = 0, ts: float = 0.0, source: Source = "screen"
    ) -> Frame:
        return Frame(self.image(offset), ts, source, frame_id)

    def line_at(self, code: int) -> Line | None:
        return self._by_colour.get(code)

    def secret_boxes(self, offset: int) -> dict[str, tuple[int, int, int, int]]:
        """Each visible secret's left, top, right, bottom on screen, cut at the edges."""
        found = {}
        for line in self.lines:
            top, bottom = max(line.y - offset, 0), min(line.y - offset + line.h, self.height)
            if bottom > top and line.text in self.secrets:
                left = self.pane[0] + line.x
                found[line.text] = (left, top, left + line.w, bottom)
        for line in self.side_lines:
            if line.text in self.secrets:
                found[line.text] = (line.x, line.y, line.x + line.w, line.y + line.h)
        return found


class SceneReader:
    """Fake OCR for a ScrollScene. Each read reports the whole lines it can see.

    blind holds text it fails to read, and wobble moves each box edge by up to that
    many pixels per read, which is how real OCR behaves. miss_rate is the chance of
    missing a line it has already read once. A secret missed on its very first read
    is a different problem, which tracking can't help with.
    """

    def __init__(self, scene: ScrollScene, wobble: int = 0, miss_rate: float = 0.0) -> None:
        self.scene = scene
        self.blind: set[str] = set()
        self.wobble = wobble
        self.miss_rate = miss_rate
        self._seen: set[str] = set()
        self.calls = 0
        self.read_sizes: list[tuple[int, int]] = []
        self._rng = np.random.default_rng(3)

    def __call__(self, image: np.ndarray) -> tuple[list, None]:
        self.calls += 1
        self.read_sizes.append(image.shape[:2])
        pixels = image.astype(np.int32)
        codes = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
        found = []
        for code in np.unique(codes[::2, ::3]):
            line = self.scene.line_at(int(code))
            if line is None or line.text in self.blind:
                continue
            rows, columns = np.nonzero(codes == code)
            top, bottom = int(rows.min()), int(rows.max()) + 1
            left, right = int(columns.min()), int(columns.max()) + 1
            if bottom - top < line.h or right - left < line.w:
                continue  # cut off, so there's no reading it yet
            if line.text in self._seen and self._rng.random() < self.miss_rate:
                continue
            self._seen.add(line.text)
            if self.wobble:
                left, top, right, bottom = (
                    edge + int(self._rng.integers(-self.wobble, self.wobble + 1))
                    for edge in (left, top, right, bottom)
                )
            corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
            found.append((corners, line.text, 0.9))
        return found, None
