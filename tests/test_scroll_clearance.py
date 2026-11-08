"""Scrolling must never clear pixels that the scroll doesn't explain.

Motion estimates may be tolerant, but a pixel only counts as checked if it really is
old content in a new place. All text here is fake.
"""

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.text_detect import TextDetector
from eclipse.types import Frame
from eval.scroll_scene import FAKE_EMAIL, SceneReader, ScrollScene

NEW_TEXT = "new.fake@example.com"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ink(shape, place, scale=0.5, line=cv2.LINE_8):
    """A mask of the new text, as one-pixel strokes unless the scale makes them bolder."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.putText(mask, NEW_TEXT, place, FONT, scale, 255, 1, line)
    return mask


def painted(image, mask, contrast):
    """The image with the mask drawn on it, this much darker or lighter than what's there."""
    change = (mask.astype(int) * contrast // 255)[:, :, None]
    lighter = image.astype(int).mean() < 128
    result = image.astype(int) + change if lighter else image.astype(int) - change
    return np.clip(result, 0, 255).astype(np.uint8)


class ReaderOfPlantedText:
    """OCR that reads the new text only when it is handed all of it, like real OCR.

    Everything else goes to the wrapped reader.
    """

    def __init__(self, plain_reader=None):
        self.plain_reader = plain_reader
        self.template = None
        self.calls = 0
        self.hits = 0
        self.sizes = []

    def plant(self, image, mask):
        rows, columns = np.nonzero(mask)
        top, bottom = int(rows.min()) - 2, int(rows.max()) + 3
        left, right = int(columns.min()) - 2, int(columns.max()) + 3
        self.template = image[top:bottom, left:right].copy()

    def __call__(self, image):
        self.calls += 1
        self.sizes.append(image.shape[:2])
        found = []
        if self.plain_reader is not None:
            found.extend(self.plain_reader(image)[0])
        template = self.template
        if (
            template is not None
            and image.shape[0] >= template.shape[0]
            and image.shape[1] >= template.shape[1]
        ):
            cost = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
            left, top = cv2.minMaxLoc(cost)[2]
            height, width = template.shape[:2]
            if np.array_equal(image[top : top + height, left : left + width], template):
                self.hits += 1
                corners = [
                    [left, top], [left + width, top],
                    [left + width, top + height], [left, top + height],
                ]
                found.append((corners, NEW_TEXT, 1.0))
        return found, None


class Stage:
    def __init__(self, reader, pixel_scale=1):
        self.reader = reader
        self.detect = TextDetector(reader=reader, pixel_scale=pixel_scale)
        self.compositor = Compositor()
        self.next_id = 0

    def show(self, image):
        frame = Frame(image, self.next_id / 60, "screen", self.next_id)
        self.next_id += 1
        return self.compositor.process(frame, self.detect)


def visible_ink(raw, output, mask):
    return int(((output == raw).all(axis=2) & (mask > 0) & output.any(axis=2)).sum())


def striped_page(height=480, width=640, stripe_width=250):
    """Text-like rows on the left, and blank space on the right for new text to land on."""
    rows = np.random.default_rng(19).integers(150, 205, size=(height, 1, 1), dtype=np.uint8)
    image = np.full((height, width, 3), 240, np.uint8)
    image[:, :stripe_width] = np.tile(rows, (1, stripe_width, 3))
    return image


def scrolled(image, by=6):
    moved = np.roll(image, -by, axis=0)
    moved[-by:] = 180
    return moved


@pytest.mark.parametrize(
    "scale, contrast, line",
    [(0.5, 30, cv2.LINE_8), (0.7, 30, cv2.LINE_8), (0.5, 160, cv2.LINE_8), (0.8, 50, cv2.LINE_AA)],
    ids=["small-faint", "large-faint", "small-strong", "smooth-edged"],
)
def test_new_text_during_a_scroll_is_read_and_covered(scale, contrast, line):
    before = striped_page()
    plain = scrolled(before)
    mask = ink(plain.shape, (300, 220), scale, line)
    now = painted(plain, mask, contrast)
    reader = ReaderOfPlantedText()
    reader.plant(now, mask)
    stage = Stage(reader)
    stage.show(before)
    reads = reader.calls
    output = stage.show(now)
    assert stage.detect.stats["scrolls"] == 1
    assert reader.calls > reads
    assert reader.hits >= 1
    assert visible_ink(now, output, mask) == 0


def test_one_pixel_of_new_ink_is_enough_to_read_again():
    before = striped_page()
    now = scrolled(before)
    now[220, 400] = 200  # a dot on blank space, thinner than any erosion would leave
    reader = ReaderOfPlantedText()
    stage = Stage(reader)
    stage.show(before)
    reads = reader.calls
    stage.show(now)
    assert stage.detect.stats["scrolls"] == 1
    assert reader.calls > reads


def test_text_that_differs_only_in_colour_is_noticed():
    before = striped_page()
    plain = scrolled(before)
    mask = ink(plain.shape, (300, 220), 0.7)
    now = plain.copy()
    # (255, 233, 233) and (240, 240, 240) are close to the same grey.
    now[mask > 0] = (255, 225, 225)
    assert abs(int(cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)[mask > 0].mean()) - 240) < 16
    reader = ReaderOfPlantedText()
    reader.plant(now, mask)
    stage = Stage(reader)
    stage.show(before)
    output = stage.show(now)
    assert stage.detect.stats["scrolls"] == 1
    assert visible_ink(now, output, mask) == 0


def lit_scene():
    return ScrollScene(secrets={6: FAKE_EMAIL}, side_secrets={}, lines=60)


def test_a_small_change_during_a_scroll_reads_the_whole_line_around_it():
    scene = lit_scene()
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(scene.image(0))
    now = scene.image(6)
    line = scene.lines[3]
    left = scene.pane[0] + line.x
    top = line.y - 6
    now[top + 8 : top + 10, left + line.w // 2 : left + line.w // 2 + 2] = 255
    reader.sizes.clear()
    stage.show(now)
    assert stage.detect.stats["scrolls"] == 1
    assert max(width for _, width in reader.sizes) >= line.w


def test_a_cut_off_line_in_a_changed_area_is_read_with_all_of_its_height():
    """A band of rows must take in a whole text line, not just the rows that changed."""
    before = striped_page()
    plain = scrolled(before)
    mask = ink(plain.shape, (300, 220), 1.4)
    now = painted(plain, mask, 40)
    reader = ReaderOfPlantedText()
    reader.plant(now, mask)
    stage = Stage(reader)
    stage.show(before)
    reader.sizes.clear()
    output = stage.show(now)
    rows = np.flatnonzero(mask.any(axis=1))
    assert max(height for height, _ in reader.sizes) >= rows[-1] - rows[0] + 1
    assert visible_ink(now, output, mask) == 0


def test_new_text_inside_a_scrolling_pane_is_covered_on_its_first_frame():
    scene = lit_scene()
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(scene.image(0))
    for offset in range(6, 60, 6):
        stage.show(scene.image(offset))
    plain = scene.image(60)
    mask = ink(plain.shape, (scene.pane[0] + 40, 300))
    now = painted(plain, mask, 60)
    reader.plant(now, mask)
    output = stage.show(now)
    assert visible_ink(now, output, mask) == 0


def test_new_text_that_scrolls_with_the_page_stays_covered():
    scene = lit_scene()
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(scene.image(0))
    place = (scene.pane[0] + 40, 300)
    for step, offset in enumerate(range(6, 120, 6)):
        plain = scene.image(offset)
        mask = ink(plain.shape, (place[0], place[1] - step * 6))
        now = painted(plain, mask, 60)
        reader.plant(now, mask)
        assert visible_ink(now, stage.show(now), mask) == 0, offset


def test_a_notification_that_lasts_one_frame_is_covered_and_leaves_nothing_behind():
    scene = lit_scene()
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(scene.image(0))
    stage.show(scene.image(6))
    plain = scene.image(12)
    mask = ink(plain.shape, (scene.pane[0] + 40, 200))
    now = painted(plain, mask, 80)
    reader.plant(now, mask)
    output = stage.show(now)
    assert visible_ink(now, output, mask) == 0
    reader.template = None
    # Its cover needs two reads of the changed pixels before it comes off.
    for offset in (18, 24, 30):
        after = stage.show(scene.image(offset))
    rows, columns = np.nonzero(mask)
    area = (rows.min() - 18, rows.max() - 18, columns.min(), columns.max())
    assert after[area[0] : area[1], area[2] : area[3]].any()


def test_new_text_right_after_a_reversal_is_covered():
    scene = lit_scene()
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(scene.image(0))
    for offset in [*range(6, 96, 6), *range(90, 60, -6)]:
        stage.show(scene.image(offset))
    plain = scene.image(54)
    mask = ink(plain.shape, (scene.pane[0] + 40, 260))
    now = painted(plain, mask, 50)
    reader.plant(now, mask)
    assert visible_ink(now, stage.show(now), mask) == 0


def test_new_text_in_a_still_pane_beside_a_scrolling_one_is_covered():
    scene = lit_scene()
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(scene.image(0))
    stage.show(scene.image(6))
    plain = scene.image(12)
    mask = ink(plain.shape, (scene.side[0] + 5, 300), 0.3)
    now = painted(plain, mask, 50)
    reader.plant(now, mask)
    assert visible_ink(now, stage.show(now), mask) == 0


def test_a_still_pane_is_not_read_again_just_because_one_beside_it_scrolls():
    scene = ScrollScene(secrets={}, side_secrets={}, lines=60)
    stage_reader = SceneReader(scene)
    stage = Stage(stage_reader)
    stage.show(scene.image(0))
    reads = stage_reader.calls
    for offset in range(6, 42, 6):
        stage.show(scene.image(offset))
    assert stage.detect.stats["scrolls"] > 0
    # Only the rows scrolling in are read, once whole lines are in: never the still pane.
    assert stage_reader.calls - reads <= 3
    assert all(height < 120 for height, _ in stage_reader.read_sizes[reads:])


def shrink(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def spaced_scene():
    """Block lines with wide gaps, as dense real text fails the scroll check outright."""
    return ScrollScene(
        secrets={}, side_secrets={}, width=1512, height=982, pane=(40, 1200), side=(1240, 1500),
        lines=60, pitch=80,
    )


SCALE = 1920 / 3024


def test_fractional_scrolling_alone_does_not_flood_the_reader():
    scene = spaced_scene()
    reader = ReaderOfPlantedText()
    stage = Stage(reader)
    stage.show(shrink(scene.image(0), SCALE))
    reads = reader.calls
    frames = 0
    for offset in range(13, 13 * 22, 13):  # 8.25 px a frame once shrunk
        stage.show(shrink(scene.image(offset), SCALE))
        frames += 1
    assert stage.detect.stats["scrolls"] == frames
    # Incoming rows and the occasional recheck get read. The whole view doesn't, each frame.
    assert reader.calls - reads < frames / 2


def test_new_text_during_a_fractional_scroll_is_covered():
    scene = spaced_scene()
    reader = ReaderOfPlantedText()
    stage = Stage(reader)
    stage.show(shrink(scene.image(0), SCALE))
    for offset in range(13, 13 * 10, 13):
        stage.show(shrink(scene.image(offset), SCALE))
    plain = shrink(scene.image(13 * 10), SCALE)
    # In the gap below the line at 420 - 130 = 290, on the page's own background.
    mask = ink(plain.shape, (100, round(350 * SCALE)))
    now = painted(plain, mask, 50)
    reader.plant(now, mask)
    output = stage.show(now)
    assert stage.detect.stats["scrolls"] == 10
    assert reader.hits >= 1
    assert visible_ink(now, output, mask) == 0


def test_a_long_fractional_scroll_rechecks_the_whole_view_now_and_then():
    scene = spaced_scene()
    reader = ReaderOfPlantedText()
    stage = Stage(reader)
    stage.show(shrink(scene.image(0), SCALE))
    stage.reader.sizes.clear()
    for offset in range(13, 13 * 40, 13):
        stage.show(shrink(scene.image(offset), SCALE))
    height = shrink(scene.image(0), SCALE).shape[0]
    assert max(size[0] for size in stage.reader.sizes) > height * 0.8


def test_retina_scale_keeps_every_rule():
    before = striped_page(900, 1600)
    plain = scrolled(before)
    mask = ink(plain.shape, (300, 220), 0.8, cv2.LINE_AA)
    now = painted(plain, mask, 50)
    reader = ReaderOfPlantedText()
    reader.plant(now, mask)
    stage = Stage(reader, pixel_scale=2)
    stage.show(before)
    output = stage.show(now)
    assert reader.hits >= 1
    assert visible_ink(now, output, mask) == 0


def busy_beside(scene, image, index):
    """The side pane changes every frame, like a live OBS preview next to the document."""
    image = image.copy()
    left = scene.side[0] + 10 + (index * 7) % 60
    image[300:330, scene.side[0] : scene.side[1]] = 230
    image[305:325, left : left + 120] = (40 + index * 9) % 200
    return image


def test_a_scroll_is_followed_while_something_beside_it_keeps_changing():
    scene = ScrollScene(side_secrets={}, lines=60)
    stage = Stage(SceneReader(scene))
    stage.show(busy_beside(scene, scene.image(0), 0))
    for index, offset in enumerate(range(6, 66, 6), 1):
        stage.show(busy_beside(scene, scene.image(offset), index))
    # Before, the change beside the document made the whole area fail the scroll check,
    # so the document was read again from scratch on every frame.
    assert stage.detect.stats["scrolls"] == 10


def busy_area(scene, image, index):
    """A big part of the side pane redrawn every frame, like OBS's live preview: never a
    scroll or a move, and big enough to be an area of its own."""
    image = image.copy()
    rng = np.random.default_rng(index)
    image[150:420, scene.side[0] : scene.side[1]] = rng.integers(
        60, 200, (270, scene.side[1] - scene.side[0], 1), dtype=np.uint8
    )
    return image


def test_a_busy_area_is_left_out_of_following_but_still_read():
    scene = ScrollScene(side_secrets={}, lines=60)
    reader = ReaderOfPlantedText(SceneReader(scene))
    stage = Stage(reader)
    stage.show(busy_area(scene, scene.image(0), 0))
    frames = 0
    for index, offset in enumerate(range(6, 78, 6), 1):
        stage.show(busy_area(scene, scene.image(offset), index))
        frames += 1
    assert stage.detect.stats["scrolls"] == frames
    assert stage.detect.stats["areas_left_out"] > 0
    # New text in the left-out area is read and covered on the frame it appears.
    now = busy_area(scene, scene.image(78), 99)
    mask = ink(now.shape, (scene.side[0] + 10, 300), 0.5)
    now = painted(now, mask, 120)
    reader.plant(now, mask)
    output = stage.show(now)
    assert reader.hits >= 1
    assert visible_ink(now, output, mask) == 0


def test_a_scroll_is_followed_with_small_changes_far_from_it():
    scene = ScrollScene(side_secrets={}, lines=60)
    stage = Stage(SceneReader(scene))
    stage.show(scene.image(0))
    for index, offset in enumerate(range(6, 66, 6), 1):
        image = scene.image(offset)
        x = scene.side[0] + 20 + index * 9
        image[440:452, x : x + 8] = 0  # a cursor moving in the side pane
        stage.show(image)
    assert stage.detect.stats["scrolls"] == 10
