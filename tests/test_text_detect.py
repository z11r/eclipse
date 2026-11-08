import typing

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.text_detect import TTL_MS, TextDetector
from eclipse.text_detect.detection import CATEGORIES, DetectionError
from eclipse.types import Category, Detection, Frame


class FakeReader:
    def __init__(self, text):
        self.text = text

    def __call__(self, image):
        if self.text is None:
            raise RuntimeError("sk-FAKE-OCR-DIED")
        return [([[20, 20], [120, 20], [120, 40], [20, 40]], self.text, 0.9)], None


def frame(frame_id=7):
    return Frame(np.zeros((80, 200, 3), dtype=np.uint8), 0.0, "screen", frame_id)


def test_returns_detections_tied_to_the_frame():
    found = TextDetector(reader=FakeReader("demo@example.com"))(frame())
    assert found
    assert all(isinstance(detection, Detection) for detection in found)
    assert all(detection.frame_id == 7 for detection in found)
    assert all(detection.ttl_ms == TTL_MS for detection in found)


def test_two_fa_uses_the_shared_category_name():
    found = TextDetector(reader=FakeReader("Your code: 482913"))(frame())
    assert "two_fa" in {detection.category for detection in found}


def test_every_detector_category_is_a_shared_category():
    names = {"2fa": "two_fa"}
    allowed = set(typing.get_args(Category))
    assert {names.get(category, category) for category in CATEGORIES} <= allowed


def test_unchanged_frame_keeps_reporting_its_boxes():
    detect = TextDetector(reader=FakeReader("demo@example.com"))
    still = frame()
    assert detect(still)
    assert detect(still)


def test_keywords_are_found():
    detect = TextDetector(keywords=["fake street"], reader=FakeReader("12 Fake Street"))
    assert "keyword" in {detection.category for detection in detect(frame())}


def test_ocr_failure_raises_for_the_compositor():
    with pytest.raises(DetectionError):
        TextDetector(reader=FakeReader(None))(frame())


def test_ocr_failure_never_lets_the_frame_out(capsys):
    raw = frame()
    raw.image[:] = 200
    compositor = Compositor()
    output = compositor.process(raw, TextDetector(reader=FakeReader(None)))
    assert not output.any()
    assert compositor.held_frames == 1
    assert "sk-FAKE" not in capsys.readouterr().out


class PageReader:
    """Fake OCR for a page of text, seeing only the area it is handed."""

    def __init__(self, lines):
        self.lines = lines  # (text, left, top, right, bottom)
        self.heights = []
        self.widths = []

    def __call__(self, image):
        self.heights.append(image.shape[0])
        self.widths.append(image.shape[1])
        blue, green, red = (int(value) for value in image[0, 0])
        position = blue * 65536 + green * 256 + red
        offset_y, offset_x = position // 1024, position % 1024
        height, width = image.shape[:2]
        found = []
        for text, left, top, right, bottom in self.lines:
            inside = (
                offset_y <= top and bottom <= offset_y + height
                and offset_x <= left and right <= offset_x + width
            )
            if inside:
                x1, x2, y1, y2 = left - offset_x, right - offset_x, top - offset_y, bottom - offset_y
                found.append(([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, 0.9))
        return found, None


def page(height=400, width=600):
    """A blank page where each pixel holds its own position, for PageReader."""
    rows, columns = np.mgrid[0:height, 0:width]
    position = rows * 1024 + columns
    return np.dstack([position // 65536, position // 256 % 256, position % 256]).astype(np.uint8)


def page_frame(image, frame_id=0):
    return Frame(image, 0.0, "screen", frame_id)


def boxes(detect, image):
    return {(d.x, d.y, d.w, d.h, d.category) for d in detect(page_frame(image))}


def test_only_changed_rows_are_read_again():
    reader = PageReader([("demo@example.com", 20, 300, 180, 320)])
    detect = TextDetector(reader=reader)
    image = page()
    boxes(detect, image)
    image[50, 100] = 255
    boxes(detect, image)
    assert reader.heights[0] == 400
    assert reader.heights[1] < 80


def test_unchanged_screen_is_not_read_again():
    reader = PageReader([("demo@example.com", 20, 300, 180, 320)])
    detect = TextDetector(reader=reader)
    image = page()
    boxes(detect, image)
    boxes(detect, image)
    assert len(reader.heights) == 1


def test_still_secret_stays_reported_while_other_rows_change():
    reader = PageReader([("demo@example.com", 20, 300, 180, 320)])
    detect = TextDetector(reader=reader)
    image = page()
    first = boxes(detect, image)
    assert first
    for row in (40, 90, 140):
        image[row, 100] = 255
        assert boxes(detect, image) == first


def test_text_that_changes_is_read_whole_even_if_taller_than_the_margin():
    reader = PageReader([("demo@example.com", 20, 200, 180, 300)])
    detect = TextDetector(reader=reader)
    image = page()
    first = boxes(detect, image)
    image[210, 100] = 255  # near the top of a 100px tall line
    assert boxes(detect, image) == first


def test_secret_that_leaves_the_screen_is_dropped():
    reader = PageReader([("demo@example.com", 20, 300, 180, 320)])
    detect = TextDetector(reader=reader)
    image = page()
    assert boxes(detect, image)
    reader.lines = []
    image[300:320, 20:180] = np.flip(image[300:320, 20:180], axis=1)  # its pixels are gone
    assert boxes(detect, image)  # one clean read isn't proof
    assert boxes(detect, image) == set()


def test_one_changed_pixel_does_not_end_a_cover_when_ocr_misses_it():
    reader = PageReader([("demo@example.com", 20, 300, 180, 320)])
    detect = TextDetector(reader=reader)
    image = page()
    first = boxes(detect, image)
    reader.lines = []
    image[310, 100] = 255  # a blinking cursor, say, not new text
    assert boxes(detect, image) == first
    assert boxes(detect, image) == first


def test_new_secret_on_a_changed_row_is_found():
    reader = PageReader([])
    detect = TextDetector(reader=reader)
    image = page()
    assert boxes(detect, image) == set()
    reader.lines = [("demo@example.com", 20, 100, 180, 120)]
    image[110, 20:180] = 255  # new text changes all of its own pixels
    assert {box[4] for box in boxes(detect, image)} == {"email"}


def test_failed_band_scan_forces_a_full_scan_next_time():
    reader = PageReader([("demo@example.com", 20, 300, 180, 320)])
    detect = TextDetector(reader=reader)
    image = page()
    boxes(detect, image)
    image[50, 100] = 255

    def broken(_):
        raise RuntimeError

    detect._detector._reader = broken
    with pytest.raises(DetectionError):
        detect(page_frame(image))
    detect._detector._reader = reader
    assert boxes(detect, image)
    assert reader.heights[-1] == 400


def test_most_of_the_screen_changing_is_a_full_scan():
    reader = PageReader([])
    detect = TextDetector(reader=reader)
    image = page()
    boxes(detect, image)
    image[10:380, 100] = 255
    boxes(detect, image)
    assert reader.heights[-1] == 400


def test_box_covers_the_secret_not_the_whole_line():
    text = "contact " + "demo@example.com" + " about the project please"
    width_per_char = 10
    reader = PageReader([(text, 20, 100, 20 + len(text) * width_per_char, 120)])
    (x, _, w, _, category), = boxes(TextDetector(reader=reader), page(width=700))
    assert category == "email"
    assert x <= 20 + 8 * width_per_char
    assert x + w >= 20 + 24 * width_per_char
    assert w < len(text) * width_per_char * 0.6


def test_text_far_apart_on_a_row_is_not_joined():
    reader = PageReader(
        [("src/module.py", 10, 100, 120, 120), ("demo@example.com", 400, 100, 560, 120)]
    )
    found = boxes(TextDetector(reader=reader), page())
    assert len(found) == 1
    assert next(iter(found))[0] >= 400


def test_split_secret_covers_both_pieces_only():
    reader = PageReader(
        [("demo@", 100, 100, 150, 120), ("example.com", 155, 100, 265, 120), ("far", 500, 100, 530, 120)]
    )
    found = boxes(TextDetector(reader=reader), page())
    assert found
    assert all(x + w <= 270 for x, _, w, _, _ in found)


@pytest.mark.parametrize("text", ["port = 8080", "timeout = 3000", "year = 2026", "2026-11-07"])
def test_ordinary_numbers_are_not_two_fa(text):
    reader = PageReader([(text, 20, 100, 200, 120)])
    assert boxes(TextDetector(reader=reader), page()) == set()


@pytest.mark.parametrize("text", ["482913", "123 456", "Your code: 4829", "OTP 91234"])
def test_real_looking_codes_are_two_fa(text):
    reader = PageReader([(text, 20, 100, 200, 120)])
    assert {box[4] for box in boxes(TextDetector(reader=reader), page())} == {"two_fa"}


def test_only_the_changed_columns_are_read_again():
    reader = PageReader([])
    detect = TextDetector(reader=reader)
    image = page(width=1000)
    boxes(detect, image)
    image[100, 500] = 255
    boxes(detect, image)
    assert reader.widths[-1] < 150


def test_changed_area_grows_to_hold_a_whole_line_of_known_text():
    line = ("demo@example.com", 100, 100, 900, 120)
    reader = PageReader([line])
    detect = TextDetector(reader=reader)
    image = page(width=1000)
    first = boxes(detect, image)
    image[110, 850] = 255
    assert boxes(detect, image) == first
    assert reader.widths[-1] >= 800


def test_secret_beside_the_change_stays_reported():
    reader = PageReader([("demo@example.com", 20, 100, 180, 120)])
    detect = TextDetector(reader=reader)
    image = page(width=1000)
    first = boxes(detect, image)
    image[110, 800] = 255
    assert boxes(detect, image) == first


def test_two_separate_changes_are_read_separately():
    reader = PageReader([])
    detect = TextDetector(reader=reader)
    image = page(height=700, width=1000)
    boxes(detect, image)
    image[100, 100] = 255
    image[600, 900] = 255
    boxes(detect, image)
    assert len(reader.heights) == 3
    assert max(reader.heights[1:]) < 80


def scrolling_page(lines, offset, height=500, width=700):
    """A document scrolled up by offset, with text sliding on and off the edges."""
    document = np.full((lines[-1][1] + height + 40, 600, 3), 40, dtype=np.uint8)
    for index, (_, top) in enumerate(lines):
        # Each line gets its own pattern, so a shift can be told from a change.
        pattern = (index * 37 % 90 + 30, index * 61 % 120 + 40)
        document[top + 2 : top + 16, 20:560:pattern[0]] = 230
        document[top + 2 : top + 16, 20 + index * 3 : 20 + pattern[1]] = 230
    image = page(height, width)
    image[:, 40:640] = document[offset : offset + height]
    return image


def scroll_lines():
    lines = [(f"line {index} plain text", 20 + index * 40) for index in range(40)]
    lines[6] = ("demo@example.com", lines[6][1])
    lines[20] = ("other@example.com", lines[20][1])
    return lines


class ScrollReader(PageReader):
    """Reads whatever lines of the document are inside the area it is handed."""

    def __init__(self, lines):
        super().__init__([])
        self.document = lines
        self.offset = 0

    def __call__(self, image):
        self.lines = [
            (text, 62, top - self.offset, 400, top - self.offset + 18)
            for text, top in self.document
        ]
        return super().__call__(image)


def test_scroll_moves_covers_without_reading_the_whole_page_again():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    reader.offset = 12
    found = detect(page_frame(scrolling_page(lines, 12)))
    assert detect.stats["scrolls"] == 1
    assert len(reader.heights) == 1  # nothing had to be read again
    assert any(d.y == 20 + 6 * 40 - 12 for d in found)


def test_secret_scrolling_onto_the_screen_is_found():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 200)))
    for offset in range(212, 420, 12):
        reader.offset = offset
        found = detect(page_frame(scrolling_page(lines, offset)))
    target = 20 + 20 * 40 - offset
    assert 0 <= target < 480  # the secret really did scroll onto the screen
    assert any(abs(d.y - target) <= 1 for d in found)


def test_secret_scrolling_off_the_screen_stops_being_reported():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    for offset in range(12, 276, 12):
        reader.offset = offset
        found = detect(page_frame(scrolling_page(lines, offset)))
    assert all(d.y >= 0 for d in found)
    assert not any(d.y + d.h < 0 for d in found)


def test_text_that_changes_during_a_scroll_is_read_again():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    changed = list(lines)
    changed[3] = ("new@example.com", lines[3][1])
    reader.document = changed
    reader.offset = 12
    image = scrolling_page(lines, 12)
    image[20 + 3 * 40 - 12 : 20 + 3 * 40 + 4, 60:600] = 255  # that line was rewritten
    found = detect(page_frame(image))
    assert any(abs(d.y - (20 + 3 * 40 - 12)) <= 1 for d in found)


def test_a_full_page_replacement_is_not_mistaken_for_a_scroll():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    rng = np.random.default_rng(1)
    image = page(500, 700)
    image[:, 40:640] = rng.integers(0, 255, (500, 600, 3))
    reader.document = [("demo@example.com", 300)]
    detect(page_frame(image))
    assert detect.stats["scrolls"] == 0
    assert reader.heights[-1] == 500


def scroll_to(detect, reader, lines, offsets):
    found = []
    for offset in offsets:
        reader.offset = offset
        found = detect(page_frame(scrolling_page(lines, offset)))
    return found


def uncovered_ink(image, boxes, top, bottom, left, right, background=40):
    """Pixels between those rows and columns that show something and no box covers."""
    ink = (image[top:bottom, left:right] != background).any(axis=2)
    for d in boxes:
        rows = slice(max(d.y - top, 0), max(d.y + d.h - top, 0))
        ink[rows, max(d.x - left, 0):max(d.x + d.w - left, 0)] = False
    return int(ink.sum())


def test_incoming_rows_stay_covered_while_scrolling():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    found = scroll_to(detect, reader, lines, [6])
    bars = [d for d in found if d.category == "unread"]
    assert bars
    assert uncovered_ink(scrolling_page(lines, 6), bars, 494, 500, 40, 640) == 0
    assert all(bar.y >= 494 and bar.y + bar.h <= 500 for bar in bars)


def test_incoming_rows_are_covered_over_their_text_only():
    """Plain background coming in can't show anything, so it needs no black."""
    lines = [("plain text", 20 + 80 * index) for index in range(12)]  # a blank line between
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    found = scroll_to(detect, reader, lines, [2])  # rows 498-500 come in, all blank
    assert not [d for d in found if d.category == "unread"]
    found = scroll_to(detect, reader, lines, [10])  # now the next line's top shows
    bars = [d for d in found if d.category == "unread"]
    assert bars and uncovered_ink(scrolling_page(lines, 10), bars, 480, 500, 40, 640) == 0
    assert sum(d.w * d.h for d in bars) < 0.5 * 600 * 20


def test_incoming_rows_stay_on_the_pane_when_something_beside_it_changes_too():
    """A screen preview and a level meter beside the pane change on every frame it
    scrolls, as OBS does in a live test. The rows held black must stay on the pane's own
    edge, not stretch over them or land below the pane, which ends at row 500."""
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    rng = np.random.default_rng(3)
    last = np.full((700, 1100, 3), 90, np.uint8)

    # Lines of word-like blocks across most of the pane, as text is.
    page = np.full((lines[-1][1] + 540, 600, 3), 40, np.uint8)
    for index, (_, top) in enumerate(lines):
        length = 300 + index * 37 % 240
        for x in range(20, 20 + length, 34 + index % 3 * 4):
            page[top + 2:top + 16, x:x + 28 + index % 4] = 230 - index % 5 * 10

    def screen(offset):
        nonlocal last
        image = np.full((700, 1100, 3), 90, np.uint8)
        image[:500, 40:640] = page[offset:offset + 500]
        # Smaller than the pane, as OBS's preview usually is beside a note.
        image[40:160, 860:1010] = cv2.resize(last, (150, 120), interpolation=cv2.INTER_AREA)
        image[400:680, 700:720] = 60
        image[int(rng.integers(420, 680)):680, 700:720] = 200  # the meter
        last = image
        return image

    detect(page_frame(screen(0)))
    for offset in (6, 12):
        reader.offset = offset
        image = screen(offset)
        found = detect(page_frame(image))
    bars = [d for d in found if d.category == "unread"]
    assert detect.stats["scrolls"] == 2
    assert bars
    assert all(bar.y >= 480 and bar.y + bar.h <= 500 for bar in bars)
    assert all(bar.x + bar.w <= 640 + 16 for bar in bars)
    ink = (image[488:500, 40:640] != 40).any(axis=2)
    covered = np.zeros_like(ink)
    for bar in bars:
        rows = slice(max(bar.y - 488, 0), max(bar.y + bar.h - 488, 0))
        covered[rows, max(bar.x - 40, 0):max(bar.x + bar.w - 40, 0)] = True
    assert not (ink & ~covered).any()


def test_scrolling_reads_incoming_rows_only_once_a_whole_line_has_come_in():
    """No OCR on every frame of a scroll: incoming rows are read once a whole line is in
    past the one the edge cuts, at most every other frame, and only those rows."""
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    reads = len(reader.heights)
    scroll_to(detect, reader, lines, [6])
    assert len(reader.heights) == reads  # part of one line: nothing to read yet
    offsets = list(range(12, 96, 6))
    scroll_to(detect, reader, lines, offsets)
    assert len(reader.heights) - reads <= (len(offsets) + 1) // 2
    assert all(height < 200 for height in reader.heights[reads:])


def test_a_slow_scroll_does_not_hold_rows_black_until_many_have_come_in():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    tallest = 0
    for offset in range(2, 80, 2):
        found = scroll_to(detect, reader, lines, [offset])
        bars = [d for d in found if d.category == "unread"]
        if bars:
            tallest = max(tallest, max(d.y + d.h for d in bars) - min(d.y for d in bars))
    assert tallest <= 24 + 8 + 2 * 6  # what's kept, what a read needs, and a few frames


def test_covers_still_move_on_frames_with_no_ocr():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    found = scroll_to(detect, reader, lines, [6, 12, 18])
    assert any(d.category == "email" and abs(d.y - (20 + 6 * 40 - 18)) <= 1 for d in found)


def test_unread_rows_are_read_and_uncovered_when_scrolling_stops():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    scroll_to(detect, reader, lines, [6, 12])
    reads = len(reader.heights)
    found = scroll_to(detect, reader, lines, [12])
    assert len(reader.heights) > reads
    assert not any(d.category == "unread" for d in found)


def test_incoming_secret_is_found_after_scrolling_stops():
    lines = [(f"line {index} plain", 20 + index * 40) for index in range(40)]
    lines[12] = ("demo@example.com", 20 + 12 * 40)  # just below the screen
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    found = scroll_to(detect, reader, lines, [30, 30])
    assert any(d.category == "email" and abs(d.y - (20 + 12 * 40 - 30)) <= 1 for d in found)


def test_failed_read_clears_unread_tracking_and_scans_everything():
    lines = scroll_lines()
    reader = ScrollReader(lines)
    detect = TextDetector(reader=reader)
    detect(page_frame(scrolling_page(lines, 0)))
    scroll_to(detect, reader, lines, [6])

    def broken(_):
        raise RuntimeError

    detect._detector._reader = broken
    with pytest.raises(DetectionError):
        detect(page_frame(scrolling_page(lines, 6) * 0 + 9))
    detect._detector._reader = reader
    detect(page_frame(scrolling_page(lines, 6)))
    assert reader.heights[-1] == 500


def covered_text(text, keywords=()):
    """Which characters of one OCR line a cover reaches, at 10 px per character."""
    width = len(text) * 10
    reader = PageReader([(text, 100, 100, 100 + width, 120)])
    found = TextDetector(keywords=keywords, reader=reader)(page_frame(page(width=700)))
    box = next(d for d in found)
    first = (box.x - 100) // 10
    last = (box.x + box.w - 100 + 9) // 10
    return text[max(first, 0) : last], box.category


def test_a_match_inside_a_word_covers_the_whole_word():
    covered, category = covered_text("see <demo@example.com>. for details about the plan")
    assert category == "email"
    assert "<demo@example.com>." in covered


def test_a_one_word_region_is_covered_edge_to_edge():
    reader = PageReader([("(demo@example.com)", 100, 100, 280, 120)])
    (x, _, w, _, _), = boxes(TextDetector(reader=reader), page(width=700))
    assert (x, x + w) == (100, 280)


def test_cover_lands_on_a_full_width_secret():
    covered, category = covered_text("call me on （780） 555-0123 please thanks")
    assert category == "phone"
    assert "（780） 555-0123" in covered
    assert "please" not in covered


def test_cover_stays_put_when_a_character_expands():
    # The ligature becomes two letters, which must not push the cover sideways.
    covered, category = covered_text("ﬁﬁﬁﬁ note demo@example.com end of line here")
    assert category == "email"
    assert "demo@example.com" in covered
    assert "ﬁ" not in covered


def test_keyword_with_spaces_inside_a_word_gets_a_cover_in_the_right_place():
    covered, category = covered_text(
        "the address is De mo Ma ple Avenue for now okay", ["Demo Maple Avenue"]
    )
    assert category == "keyword"
    assert "De mo Ma ple Avenue" in covered
    assert "okay" not in covered


def test_keyword_still_ignores_different_words():
    from eclipse.text_detect.detection import classify_text

    assert "keyword" not in classify_text("Demo Oak Avenue", ["Demo Maple Avenue"])
    assert "keyword" not in classify_text("hello world", ["", "   ", "\t"])


def test_a_window_dragged_up_or_down_gets_no_strip_over_what_it_uncovers():
    """The background a dragged window uncovers stays where it is. It isn't text
    scrolling in at a pane's edge, so it is read where it shows instead of held black."""
    rng = np.random.default_rng(5)
    desktop = rng.integers(0, 2, (700, 900, 1), dtype=np.uint8) * 200 + 30
    desktop = np.repeat(desktop, 3, axis=2)  # a busy, still background
    window = np.full((300, 500, 3), 40, np.uint8)
    for index, row in enumerate(range(20, 300, 40)):
        # Each row its own pattern, as text is, so no shift but the real one fits.
        window[row:row + 14, 30:470:(index * 37 % 90 + 30)] = 230
        window[row:row + 14, 30 + index * 3:30 + index * 61 % 120 + 40] = 230
    detect = TextDetector(reader=PageReader([]))
    strips = []
    for step in range(12):
        image = desktop.copy()
        top = 100 + 12 * step
        image[top:top + 300, 200:700] = window
        found = detect(page_frame(image))
        strips.append(sum(d.w * d.h for d in found if d.category == "unread"))
    assert sum(strips[2:]) == 0  # at most the drag's first frame looks like a scroll
