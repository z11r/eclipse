import cv2
import numpy as np
import pytest

from eclipse.platform import AppleVisionReader
from eclipse.text_detect.styled_text import colour_regions, contrast_bgra


def styled_page(style):
    image = np.full((200, 700, 4), (30, 30, 30, 255), np.uint8)
    colour = (0, 180, 240, 255) if style != 'selected' else (255, 255, 255, 255)
    if style == 'selected':
        cv2.rectangle(image, (45, 60), (620, 110), (180, 70, 20, 255), -1)
    cv2.putText(image, 'demo@example.com', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 2)
    if style == 'underlined':
        cv2.line(image, (50, 108), (480, 108), colour, 2)
    return image


@pytest.mark.parametrize('style', ['amber', 'underlined', 'selected'])
def test_coloured_text_gets_bounded_geometry_preserving_contrast_crop(style):
    pixels = styled_page(style)
    original = pixels.copy()
    regions = colour_regions(pixels)
    assert regions
    assert any(left <= 50 and top <= 75 and right >= 450 and bottom >= 100
               for left, top, right, bottom in regions)
    for left, top, right, bottom in regions:
        crop = contrast_bgra(pixels[top:bottom, left:right])
        assert crop.shape == (bottom-top, right-left, 4)
        assert crop[:, :, 0].max() - crop[:, :, 0].min() == 255
        assert np.array_equal(crop[:, :, 0], crop[:, :, 1])
    assert np.array_equal(pixels, original)


def test_plain_text_needs_no_colour_retry():
    image = np.full((200, 700, 4), 255, np.uint8)
    cv2.putText(image, 'demo@example.com', (30, 100), 0, 1, (0, 0, 0, 255), 2)
    assert colour_regions(image) == []


def test_retry_keeps_fast_results_and_translates_accurate_boxes():
    class FakeReader(AppleVisionReader):
        def __init__(self):
            self.calls = []

        def _read_vision(self, image, level):
            self.calls.append(level)
            text = 'fast@example.com' if level == self.FAST else 'demo@example.com'
            return [([[1, 2], [10, 2], [10, 8], [1, 8]], text, .9)], None

    pixels = styled_page('underlined')
    reader = FakeReader()
    words, _ = reader.read_bgra(pixels)
    assert reader.calls[0] == reader.FAST
    assert reader.ACCURATE in reader.calls
    left, top, _, _ = colour_regions(pixels)[0]
    assert words[0][1] == 'fast@example.com'
    assert words[1][0][0] == [left+1, top+2]


def test_failed_retry_is_not_reported_as_successful_empty_scan():
    class Broken(AppleVisionReader):
        def __init__(self):
            pass

        def _read_vision(self, image, level):
            if level == self.ACCURATE:
                raise RuntimeError('Vision failed')
            return [], None

    with pytest.raises(RuntimeError):
        Broken().read_bgra(styled_page('amber'))
