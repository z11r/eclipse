"""A camera's sensor noise must not force a full read of every frame."""

import numpy as np
import pytest

from eclipse.text_detect import TextDetector
from eclipse.text_detect.detection import Detector
from eclipse.types import Frame


class CountingReader:
    def __init__(self):
        self.areas = []

    def __call__(self, image):
        self.areas.append(image.shape[:2])
        return [], None


def camera_frame(image, frame_id=0):
    return Frame(image, 0.0, "webcam", frame_id)


def scene(height=240, width=320):
    return np.full((height, width, 3), 128, dtype=np.uint8)


def with_noise(image, seed, amount=6):
    noise = np.random.default_rng(seed).integers(-amount, amount + 1, image.shape)
    return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)


def test_sensor_noise_alone_is_not_read_again():
    reader = CountingReader()
    detect = TextDetector(reader=reader, noise_floor=8)
    base = scene()
    for seed in range(5):
        detect(camera_frame(with_noise(base, seed), seed))
    assert len(reader.areas) == 1


def test_without_a_floor_noise_reads_the_whole_frame_every_time():
    reader = CountingReader()
    detect = TextDetector(reader=reader)
    base = scene()
    for seed in range(3):
        detect(camera_frame(with_noise(base, seed), seed))
    assert reader.areas == [(240, 320)] * 3


def test_new_text_is_read_and_only_where_it_appeared():
    reader = CountingReader()
    detect = TextDetector(reader=reader, noise_floor=8)
    base = scene()
    detect(camera_frame(with_noise(base, 0)))
    written = base.copy()
    written[100:112, 40:120:3] = 20  # thin dark strokes, like a line of text
    detect(camera_frame(with_noise(written, 1), 1))
    assert len(reader.areas) == 2
    height, width = reader.areas[1]
    assert height < 240 and width < 320


def test_slow_change_cannot_creep_past_the_floor():
    reader = CountingReader()
    detect = TextDetector(reader=reader, noise_floor=8)
    image = scene()
    detect(camera_frame(image))
    for step in range(1, 6):  # 4 levels a frame: each under the floor, 20 in all
        image = image.copy()
        image[60:80, 60:120] -= 4
        detect(camera_frame(image, step))
    assert len(reader.areas) > 1


def test_noise_floor_must_be_a_nonnegative_integer():
    with pytest.raises(ValueError):
        Detector(noise_floor=-1)
