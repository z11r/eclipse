"""Failures found by an outside review, kept here so they stay fixed.

Each one failed on the code as reviewed. Everything is generated at test time, in
temporary folders, and every secret is fake.
"""

import time

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.platform import IS_MAC, AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.types import Frame
from eval import run_eval

FAKE_EMAIL = "alex.fake@example.com"


@pytest.mark.skipif(not IS_MAC, reason="uses Apple Vision on generated pixels")
def test_a_readable_email_added_during_a_scroll_is_not_left_visible():
    height, width = 900, 1600
    rows = np.random.default_rng(19).integers(150, 205, size=(height, 1, 1), dtype=np.uint8)
    before = np.full((height, width, 3), 240, np.uint8)
    before[:, :250] = np.tile(rows, (1, 250, 3))
    plain = np.roll(before, -6, axis=0)
    plain[-6:] = 180
    marker = np.zeros((height, width), np.uint8)
    cv2.putText(
        marker, FAKE_EMAIL, (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 1, cv2.LINE_AA
    )
    now = np.maximum(plain.astype(int) - (marker.astype(int) * 50 // 255)[:, :, None], 0)
    now = now.astype(np.uint8)
    glyphs = (now != plain).any(axis=2)

    engine = AppleVisionReader()
    words, _ = engine(now)
    assert any("example.com" in word[1].lower() for word in words), "Vision can't read it"

    calls = []

    def reader(image):
        calls.append(image.shape)
        return engine(image)

    detector = TextDetector(reader=reader, pixel_scale=2)
    compositor = Compositor()
    compositor.process(Frame(before, 0.0, "screen", 0), detector)
    calls.clear()
    output = compositor.process(Frame(now, 1 / 60, "screen", 1), detector)

    assert detector.stats["scrolls"] == 1
    assert calls, "the new text was never read"
    same = (output == now).all(axis=2)
    assert int((same & glyphs & output.any(axis=2)).sum()) == 0
