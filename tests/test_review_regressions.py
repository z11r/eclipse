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


class TimedSource:
    def __init__(self):
        self.next_id = 0
        self.times = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        now = time.monotonic()
        self.times.append(now)
        frame = Frame(np.full((60, 80, 3), 150, np.uint8), now, "screen", self.next_id)
        self.next_id += 1
        return frame


class ClosingSink:
    def __init__(self):
        self.images = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def send(self, image):
        self.images.append(image.copy())
        time.sleep(0.002)


def under_memory_pressure():
    source, sink = TimedSource(), ClosingSink()
    pipeline = DelayedPipeline(
        source, sink, lambda frame: [], Compositor(), delay=0.08, fps=50, max_frames=30,
        max_buffer_bytes=60 * 80 * 3 * 2,
    )
    return source, sink, pipeline.run()


def test_buffer_pressure_keeps_capture_pacing():
    source, _, stats = under_memory_pressure()
    assert stats.over_limit > 0
    # 30 grabs at 50 Hz span 0.58 s, and this allows 80 ms of scheduling tolerance.
    assert source.times[-1] - source.times[0] >= 0.50


def test_the_evaluator_accounts_for_the_content_of_a_held_frame(tmp_path):
    path = tmp_path / "held-content.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (240, 100))
    assert writer.isOpened()
    first = np.full((100, 240, 3), 30, np.uint8)
    cv2.putText(first, "FAKE SECRET", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    for image in (first, np.full_like(first, 30), np.full_like(first, 30)):
        writer.write(image)
    writer.release()
    labels = {"secrets": [{
        "id": "fake", "text": "FAKE SECRET", "category": "keyword",
        "first_frame": 0, "last_frame": 0, "box": [0, 20, 220, 50], "move": [0, 0],
    }]}
    showing = []

    def misses_it_then_fails(frame):
        if frame.frame_id:
            raise RuntimeError("injected failure")
        return []

    report = run_eval.evaluate(
        str(path), labels, misses_it_then_fails,
        on_frame=lambda raw, output: showing.append(bool(output.max() > 100)),
    )
    assert report["leaked_frames"] == sum(showing) == 3
