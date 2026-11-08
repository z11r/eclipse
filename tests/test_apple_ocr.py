import sys

import cv2
import numpy as np
import pytest

from eclipse import platform
from eclipse.compositor.compositor import Compositor
from eclipse.platform import AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.text_detect.detection import DetectionError
from eclipse.types import Frame

needs_mac = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")


def page_with(text, x=200, y=300, width=1600, height=900):
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (240, 240, 240), 3)
    return image


def test_only_runs_on_a_mac(monkeypatch):
    monkeypatch.setattr(platform, "IS_MAC", False)
    with pytest.raises(RuntimeError):
        AppleVisionReader()


@needs_mac
def test_reads_text_and_places_it_from_the_top_left():
    lines, timings = AppleVisionReader()(page_with("alex.fake@example.com"))
    assert timings is None
    corners, text, confidence = lines[0]
    assert "example.com" in text
    assert 0 <= confidence <= 1
    assert 160 <= corners[0][0] <= 240  # left, near where it was drawn
    assert 240 <= corners[0][1] <= 310  # top: y counts down from the top


@needs_mac
def test_words_get_their_own_position_even_after_a_wide_gap():
    image = np.full((900, 2400, 3), 30, dtype=np.uint8)
    for word, x in (("0016", 100), ("alex.fake@example.com", 1000)):
        cv2.putText(image, word, (x, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (240, 240, 240), 3)
    words, _ = AppleVisionReader()(image)
    email = next(word for word in words if "example" in word[1])
    assert 900 <= email[0][0][0] <= 1100


@needs_mac
def test_cover_reaches_the_end_of_a_secret_after_a_wide_gap():
    image = np.full((900, 2400, 3), 30, dtype=np.uint8)
    for word, x in (("0016", 100), ("value", 400), ("alex.fake@example.com", 800)):
        cv2.putText(image, word, (x, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (240, 240, 240), 3)
    width = cv2.getTextSize("alex.fake@example.com", cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0][0]
    found = TextDetector(reader=AppleVisionReader(), pixel_scale=2)(
        Frame(image, 0.0, "screen", 0)
    )
    box = next(d for d in found if d.category == "email")
    assert box.x <= 810
    assert box.x + box.w >= 800 + width - 10


@needs_mac
def test_blank_image_is_a_clean_empty_result():
    lines, _ = AppleVisionReader()(np.zeros((200, 300, 3), dtype=np.uint8))
    assert lines == []


@needs_mac
def test_reader_works_with_the_secret_rules_and_a_cropped_area():
    detect = TextDetector(reader=AppleVisionReader(), pixel_scale=2)
    blank = np.full((900, 1600, 3), 30, dtype=np.uint8)
    assert detect(Frame(blank, 0.0, "screen", 0)) == []
    found = detect(Frame(page_with("alex.fake@example.com"), 0.03, "screen", 1))
    assert "email" in {detection.category for detection in found}
    box = next(d for d in found if d.category == "email")
    assert 150 <= box.x <= 420 and 240 <= box.y <= 320


def test_a_vision_failure_counts_as_a_failed_scan():
    class Failing(AppleVisionReader):
        def __init__(self):
            pass

        def __call__(self, image):
            raise RuntimeError("Apple OCR failed")

    frame = Frame(np.full((100, 200, 3), 200, dtype=np.uint8), 0.0, "screen", 0)
    with pytest.raises(DetectionError):
        TextDetector(reader=Failing(), pixel_scale=2)(frame)
    compositor = Compositor()
    assert not compositor.process(frame, TextDetector(reader=Failing(), pixel_scale=2)).any()


@needs_mac
def test_colour_text_on_white_is_read_through_the_direct_path():
    image = np.full((400, 1600, 3), 255, dtype=np.uint8)
    cv2.putText(image, "alex.fake@example.com", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 200), 3)
    words, _ = AppleVisionReader()(image)
    assert any("example" in word[1] for word in words)


@needs_mac
def test_cropped_area_is_read_without_distortion():
    image = page_with("alex.fake@example.com", x=300, y=400, width=1601, height=901)
    words, _ = AppleVisionReader()(image[350:430, 280:1100])
    assert any("example" in word[1] for word in words)


@needs_mac
def test_small_change_is_noticed_however_faint():
    detect = TextDetector(reader=AppleVisionReader(), pixel_scale=2)
    image = page_with("hello there")
    detect(Frame(image, 0.0, "screen", 0))
    changed = image.copy()
    changed[10, 10, 2] = changed[10, 10, 2] + 1  # one level, one pixel, one channel
    detect(Frame(changed, 0.03, "screen", 1))
    assert detect.stats["partial_scans"] == 1


@needs_mac
@pytest.mark.parametrize('style', ['plain', 'amber', 'underlined', 'selected'])
def test_styled_email_is_recognized_and_covered(style):
    """Run on macOS: exercise native Vision, rules and final cover placement."""
    image = np.full((900, 1600, 3), 30, dtype=np.uint8)
    text = 'alex.fake@example.com'
    x, y, scale, thickness = 200, 300, 1.6, 3
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    colour = (240, 240, 240) if style in ('plain', 'selected') else (0, 180, 240)
    if style == 'selected':
        cv2.rectangle(image, (x-5, y-height-5), (x+width+5, y+baseline+5), (180, 70, 20), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)
    if style == 'underlined':
        cv2.line(image, (x, y+5), (x+width, y+5), colour, 2)
    frame = Frame(image, 0.0, 'screen', 0)
    detections = TextDetector(reader=AppleVisionReader(), pixel_scale=2)(frame)
    emails = [d for d in detections if d.category == 'email']
    assert emails, 'Styled fake email was not detected'
    assert any(d.x <= x+10 and d.x+d.w >= x+width-10 for d in emails)
    output = Compositor().process(frame, lambda f: detections)
    assert not output[y-height:y, x:x+width].any()


@needs_mac
def test_email_in_sentence_keeps_surrounding_words_visible():
    image = np.full((500, 2600, 3), 240, np.uint8)
    prefix = 'You have not interacted with '
    email = 'demo@example.com'
    suffix = ' before. Add this invitation?'
    x, y, scale, thickness = 50, 250, 1.4, 2
    cv2.putText(image, prefix+email+suffix, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), thickness)
    prefix_width = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    email_width = cv2.getTextSize(email, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    detections = TextDetector(reader=AppleVisionReader(), pixel_scale=2)(Frame(image, 0, 'screen', 0))
    boxes = [d for d in detections if d.category == 'email']
    assert boxes
    assert all(d.x > x+prefix_width-80 and d.x+d.w < x+prefix_width+email_width+80 for d in boxes)


@needs_mac
@pytest.mark.parametrize('secret', ['password=INVALID_DEMO_8842', 'API_KEY=DEMO_UNUSED_219B'])
@pytest.mark.parametrize('foreground,background', [
    ((0, 0, 200), (0, 255, 255)),
    ((130, 0, 130), (255, 0, 255)),
    ((240, 240, 240), (30, 30, 30)),
])
def test_styled_labeled_credential_is_detected_without_hiding_sentence(secret, foreground, background):
    image = np.full((500, 2400, 3), 255, np.uint8)
    prefix, suffix = 'The dummy field ', ' inside the sentence.'
    x, y, scale, thickness = 50, 250, 1.3, 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    prefix_width = cv2.getTextSize(prefix, font, scale, thickness)[0][0]
    secret_width, secret_height = cv2.getTextSize(secret, font, scale, thickness)[0]
    start = x+prefix_width
    cv2.putText(image, prefix, (x, y), font, scale, (20, 20, 20), thickness)
    cv2.rectangle(image, (start, y-secret_height-5), (start+secret_width+5, y+8), background, -1)
    cv2.putText(image, secret, (start, y), font, scale, foreground, thickness)
    cv2.putText(image, suffix, (start+secret_width, y), font, scale, (20, 20, 20), thickness)
    found = TextDetector(reader=AppleVisionReader(), pixel_scale=2)(Frame(image, 0, 'screen', 0))
    boxes = [d for d in found if d.category == 'api_key']
    assert boxes
    assert all(d.x >= start-70 and d.x+d.w <= start+secret_width+70 for d in boxes)
