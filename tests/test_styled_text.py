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


@pytest.mark.parametrize('duplicate', [False, True])
def test_ordinary_prose_does_not_become_generic_key_when_ocr_words_join(duplicate):
    from eclipse.text_detect.detection import Detector

    sentence = 'A short electronic or SMS message sent from one mobile device or computer to another'
    regions = []
    left = 10
    for text in sentence.split():
        right = left + len(text) * 10
        regions.append((left, 20, right, 40, text))
        if duplicate:
            regions.append((left+1, 21, right+1, 41, text))
        left = right + 8
    detector = Detector(reader=lambda image: ([], None))
    assert not any(category == 'api_key' for category, _, _ in detector._find_areas(regions))


def test_real_generic_key_in_one_observation_still_matches():
    from eclipse.text_detect.detection import Detector

    detector = Detector(reader=lambda image: ([], None))
    token = 'Ab9x' * 20
    assert any(category == 'api_key' for category, _, _ in
               detector._find_areas([(10, 20, 800, 40, token)]))


def test_split_structured_key_and_email_still_match():
    from eclipse.text_detect.detection import Detector

    detector = Detector(reader=lambda image: ([], None))
    for pieces, expected in [(['sk-', 'a' * 24], 'api_key'),
                             (['demo@', 'example.com'], 'email')]:
        regions = [(10, 20, 90, 40, pieces[0]), (95, 20, 300, 40, pieces[1])]
        assert any(category == expected for category, _, _ in detector._find_areas(regions))


@pytest.mark.parametrize('number', ['500', '17', '20', '28'])
def test_overlapping_numeric_observations_do_not_manufacture_codes(number):
    from eclipse.text_detect.detection import Detector

    detector = Detector(reader=lambda image: ([], None))
    regions = [(10, 20, 50, 40, number), (11, 21, 51, 41, number),
               (60, 20, 160, 40, 'pages of text')]
    assert not detector._find_areas(regions)
    assert all(not (a[0] < b[2] and b[0] < a[2])
               for group in detector._groups(regions)
               for a, b in zip(group, group[1:]))


@pytest.mark.parametrize('pieces,category', [
    (['123', '456'], '2fa'),
    (['+1 (780)', '555-0123'], 'phone'),
    (['PASSWORD=', 'fake-pass-123'], 'api_key'),
    (['4111 1111', '1111 1111'], 'card'),
    (['demo@', 'example.com'], 'email'),
])
def test_adjacent_secret_pieces_survive_overlapping_ocr_alternatives(pieces, category):
    from eclipse.text_detect.detection import Detector

    detector = Detector(reader=lambda image: ([], None))
    regions = [(10, 20, 100, 40, pieces[0]), (11, 21, 101, 41, pieces[0]),
               (105, 20, 260, 40, pieces[1]), (106, 21, 261, 41, pieces[1])]
    assert any(kind == category for kind, _, _ in detector._find_areas(regions))


def test_broad_email_sentence_is_replaced_only_with_confirmed_precise_results():
    class Reader(AppleVisionReader):
        def __init__(self, recognize=True):
            self.recognize = recognize

        def _read_vision(self, pixels, level):
            assert level == self.ACCURATE
            text = 'demo@example.com' if self.recognize else 'ordinary words'
            box = [[180, 8], [380, 8], [380, 28], [180, 28]]
            return [(box, 'You have not interacted with', .95), (box, text, .95),
                    (box, 'before.', .95)], None

    pixels = np.zeros((100, 1000, 4), np.uint8)
    original = [([[20, 20], [950, 20], [950, 40], [20, 40]],
                 'You have not interacted with demo@example.com before.', .9)]
    precise = Reader()._refine_secret_boxes(pixels, original)
    assert precise[1][1] == 'demo@example.com'
    assert precise[0][0][0] == [192, 20]
    assert precise[0][0][1][0] < 950
    assert Reader(False)._refine_secret_boxes(pixels, original) == original


def test_normal_word_box_does_not_trigger_extra_recognition():
    class Reader(AppleVisionReader):
        def __init__(self):
            pass

        def _read_vision(self, *args):
            raise AssertionError('normal word must not be reread')

    words = [([[20, 20], [220, 20], [220, 40], [20, 40]], 'demo@example.com', .9)]
    assert Reader()._refine_secret_boxes(np.zeros((100, 1000, 4), np.uint8), words) == words


def test_line_width_word_box_is_refined_but_missing_other_secret_keeps_fallback():
    class Reader(AppleVisionReader):
        def __init__(self):
            pass

        def _read_vision(self, pixels, level):
            return [([[50, 8], [250, 8], [250, 28], [50, 28]], 'demo@example.com', .9)], None

    pixels = np.zeros((100, 1000, 4), np.uint8)
    box = [[20, 20], [950, 20], [950, 40], [20, 40]]
    assert Reader()._refine_secret_boxes(pixels, [(box, 'demo@example.com', .9)])[0][0] != box
    sentence = [(box, 'demo@example.com PASSWORD=fake-secret-123', .9)]
    assert Reader()._refine_secret_boxes(pixels, sentence) == sentence


@pytest.mark.parametrize('pieces', [
    ['demo@example.com'], ['demo@', 'example.com'], ['demo', '@', 'example', '.', 'com'],
])
def test_email_in_sentence_does_not_expand_to_surrounding_words(pieces):
    from eclipse.text_detect.detection import Detector

    words = ['You', 'have', 'not', 'interacted', 'with'] + pieces + ['before.', 'Do', 'you', 'want', 'to', 'add', 'this', 'invitation?']
    regions = []
    x = 10
    for word in words:
        right = x + len(word)*10
        regions.append((x, 20, right, 40, word))
        x = right+8
    start = regions[5][0]
    end = regions[5+len(pieces)-1][2]
    found = Detector(reader=lambda image: ([], None))._find_areas(regions)
    emails = [rect for kind, rect, _ in found if kind == 'email']
    assert emails
    assert all(rect[0] >= start and rect[2] <= end for rect in emails)
    assert min(rect[0] for rect in emails) == start
    assert max(rect[2] for rect in emails) == end


def test_separate_emails_in_same_sentence_both_remain_covered():
    from eclipse.text_detect.detection import Detector

    regions = [(10, 20, 180, 40, 'first@example.com'),
               (188, 20, 220, 40, 'and'),
               (228, 20, 408, 40, 'second@example.com')]
    found = Detector(reader=lambda image: ([], None))._find_areas(regions)
    emails = [rect for kind, rect, _ in found if kind == 'email']
    assert any(rect[0] == 10 for rect in emails)
    assert any(rect[2] == 408 for rect in emails)
    assert not any(rect[0] <= 188 and rect[2] >= 220 for rect in emails)


def word_regions(words):
    regions, x = [], 10
    for word in words:
        right = x+len(word)*10
        regions.append((x, 20, right, 40, word))
        x = right+8
    return regions


@pytest.mark.parametrize('words', [list('12345'), list('67890'),
    [str(i) for i in range(1, 11)], [str(i) for i in range(11, 21)],
    [str(i) for i in range(21, 31)]])
def test_chart_cells_do_not_become_numeric_secrets(words):
    from eclipse.text_detect.detection import Detector
    found = Detector(reader=lambda image: ([], None))._find_areas(word_regions(words))
    assert not {kind for kind, _, _ in found} & {'2fa', 'phone', 'card'}


@pytest.mark.parametrize('secret', ['password=INVALID_DEMO_8842', 'API_KEY=DEMO_UNUSED_219B'])
def test_inline_credentials_do_not_expand_into_surrounding_sentence(secret):
    from eclipse.text_detect.detection import Detector
    regions = word_regions(['The', 'dummy', 'field', secret, 'inside', 'the', 'sentence.'])
    found = Detector(reader=lambda image: ([], None))._find_areas(regions)
    key_boxes = [rect for kind, rect, _ in found if kind == 'api_key']
    assert key_boxes
    assert all(rect[0] >= regions[3][0] and rect[2] <= regions[3][2] for rect in key_boxes)


@pytest.mark.parametrize('text,category', [
    ('482913', '2fa'), ('Your code: 4829', '2fa'), ('OTP 91234', '2fa'),
    ('+1 (780) 555-0123', 'phone'), ('PASSWORD=fake-pass-123', 'api_key'),
    ('API_KEY=DEMO_UNUSED_219B', 'api_key'), ('4111 1111 1111 1111', 'card'),
])
def test_real_numeric_and_labeled_secrets_still_detected(text, category):
    from eclipse.text_detect.detection import classify_text
    assert category in classify_text(text)


def test_original_colour_accurate_pass_can_recover_key_missing_from_normalized_pass():
    class Reader(AppleVisionReader):
        def __init__(self):
            self.accurate_calls = 0
        def _read_vision(self, pixels, level):
            if level == self.FAST:
                return [], None
            self.accurate_calls += 1
            # Simulate recognition succeeding only on original color pixels.
            if not np.array_equal(pixels[:, :, 0], pixels[:, :, 1]):
                return [([[10, 8], [180, 8], [180, 28], [10, 28]], 'API_KEY=DEMO_UNUSED_219B', .9)], None
            return [], None
    reader = Reader()
    words, _ = reader.read_bgra(styled_page('amber'))
    assert any(text == 'API_KEY=DEMO_UNUSED_219B' for _, text, _ in words)
    assert reader.accurate_calls >= 2
