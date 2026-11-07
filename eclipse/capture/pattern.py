"""A moving test pattern, for checking the OBS link without showing a real screen."""

from __future__ import annotations

import time

import cv2
import numpy as np

from eclipse.types import Frame


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
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
        )
        frame = Frame(image, ts, "screen", self._next_id)
        self._next_id += 1
        return frame
