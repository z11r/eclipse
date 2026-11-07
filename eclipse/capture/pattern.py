"""A moving test pattern with a fake secret, for checking OBS and redaction safely."""

from __future__ import annotations

import time

import cv2
import numpy as np

from eclipse.types import Detection, Frame

FAKE_SECRET = "sk-FAKE-KEY-1234567890"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 1.2
_THICKNESS = 2


class TestPattern:
    __test__ = False  # not a pytest class, despite the name

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self._width = width
        self._height = height
        self._next_id = 0

    def __enter__(self) -> "TestPattern":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def read(self) -> Frame:
        ts = time.monotonic()
        image = np.full((self._height, self._width, 3), 40, dtype=np.uint8)
        bar_x = self._next_id * 8 % self._width
        image[:, bar_x : bar_x + 40] = (0, 200, 255)
        cv2.putText(
            image,
            f"ECLIPSE TEST PATTERN  frame {self._next_id}",
            (60, self._height // 2),
            _FONT,
            _SCALE,
            (255, 255, 255),
            _THICKNESS,
        )
        x, y, _, text_height = self._secret_box(self._next_id)
        cv2.putText(
            image,
            FAKE_SECRET,
            (x, y + text_height),
            _FONT,
            _SCALE,
            (255, 255, 255),
            _THICKNESS,
        )
        frame = Frame(image, ts, "screen", self._next_id)
        self._next_id += 1
        return frame

    def detect(self, frame: Frame) -> list[Detection]:
        """Stands in for the text layer: reports exactly where the fake secret is."""
        x, y, width, height = self._secret_box(frame.frame_id)
        return [
            Detection(x, y, width, height, "api_key", 1.0, 100, frame.frame_id),
        ]

    def _secret_box(self, frame_id: int) -> tuple[int, int, int, int]:
        (width, height), baseline = cv2.getTextSize(FAKE_SECRET, _FONT, _SCALE, _THICKNESS)
        x = 60 + frame_id * 4 % (self._width - width - 120)
        return x, self._height // 2 + 80, width, height + baseline
