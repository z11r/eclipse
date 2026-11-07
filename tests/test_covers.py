"""The visible result of detector, tracks and compositor working together.

These check output pixels. A cover can be right in the bookkeeping and wrong on screen.
"""

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.output.virtual_cam import OUTPUT_HEIGHT, OUTPUT_WIDTH, fit
from eclipse.text_detect import TTL_MS, UNREAD_TTL_MS, TextDetector
from eclipse.text_detect.detection import Detector, DetectionError
from eclipse.types import Frame
from eval.scroll_scene import (
    BACKGROUND,
    FAKE_EMAIL,
    FAKE_KEY,
    FAKE_PHONE,
    LINE_HEIGHT,
    LINE_PITCH,
    SceneReader,
    ScrollScene,
)

EMAIL_TOP = 20 + 6 * LINE_PITCH  # where the fake email sits in the document


class Run:
    """Drives one scene through the real pipeline, a frame at a time."""

    def __init__(self, scene=None, keywords=(), wobble=0):
        self.scene = scene or ScrollScene()
        self.reader = SceneReader(self.scene, wobble=wobble)
        self.detect = TextDetector(keywords=keywords, reader=self.reader)
        self.compositor = Compositor()
        self.frame_id = 0
        self.found = []

    def show(self, image, source="screen"):
        frame = Frame(image, self.frame_id / 60, source, self.frame_id)
        self.frame_id += 1

        def detect(frame):
            self.found = self.detect(frame)
            return self.found

        return frame, self.compositor.process(frame, detect)

    def at(self, offset):
        return self.show(self.scene.image(offset))[1]

    def covers(self, category=None):
        return [d for d in self.found if category in (None, d.category)]

    def secrets(self):
        return [d for d in self.found if d.category != "unread"]

    def unread(self):
        return self.covers("unread")


def shown_secret_pixels(scene, offset, output):
    """Secret pixels the output still shows, across every visible secret."""
    raw = scene.image(offset)
    total = 0
    for left, top, right, bottom in scene.secret_boxes(offset).values():
        text = (raw[top:bottom, left:right] != BACKGROUND).any(axis=2)
        same = (output[top:bottom, left:right] == raw[top:bottom, left:right]).all(axis=2)
        total += int((text & same).sum())
    return total


def unexplained_cover(run, offset, output):
    """Black pixels that no secret on screen and no unread row accounts for.

    The scene has no black of its own, so all black in the output is a cover.
    """
    black = (output == 0).all(axis=2)
    explained = np.zeros(black.shape, dtype=bool)
    boxes = list(run.scene.secret_boxes(offset).values())
    boxes += [(d.x, d.y, d.x + d.w, d.y + d.h) for d in run.unread()]
    for left, top, right, bottom in boxes:
        explained[
            max(top - PADDING, 0) : bottom + PADDING, max(left - PADDING, 0) : right + PADDING
        ] = True
    return int((black & ~explained).sum())


def covered_rows(output, column):
    return np.flatnonzero((output[:, column] == 0).all(axis=1))


def walk(start, stop, step):
    return list(range(start, stop, step))


SCRIPTS = {
    "down": walk(6, 240, 6),
    "up": walk(600, 360, -6),
    "rapid": walk(40, 800, 40),
    "stopping": [*walk(6, 96, 6), *[90] * 8, *walk(96, 150, 6), *[144] * 8],
    "reversal": [*walk(6, 132, 6), *walk(120, 0, -6), *walk(6, 72, 6)],
}


@pytest.mark.parametrize("name", SCRIPTS)
def test_scrolling_shows_no_secret_pixels_and_leaves_no_trail(name):
    run = Run()
    offsets = SCRIPTS[name]
    run.at(offsets[0])
    for offset in offsets[1:]:
        output = run.at(offset)
        assert shown_secret_pixels(run.scene, offset, output) == 0, offset
        assert unexplained_cover(run, offset, output) == 0, offset
        # One place per secret, never a history. The fake key's digits also count as a
        # phone number, so one place can hold two categories.
        places = {(cover.x, cover.y) for cover in run.secrets()}
        assert len(places) <= len(run.scene.secret_boxes(offset))
        for cover in run.secrets():
            assert cover.h <= LINE_HEIGHT


def test_scrolling_covers_land_on_their_text_on_frames_with_no_ocr():
    run = Run()
    run.at(0)
    reads = run.reader.calls
    for offset in walk(6, 60, 6):
        run.at(offset)
        cover = next(d for d in run.secrets() if d.category == "email")
        assert cover.y == EMAIL_TOP - offset
        assert cover.h == LINE_HEIGHT
    assert run.reader.calls == reads


def test_a_new_scroll_does_not_trim_a_cover_at_the_edge_of_the_screen():
    # A scroll's first frame only has its changed rows to go on, and those can stop
    # short of the real edge. A cover must not lose its height to that estimate.
    run = Run()
    plain_read = run.reader

    def boxes_a_little_tall(image):
        found, timings = plain_read(image)
        taller = [
            ([[x, y] if index < 2 else [x, y + 2] for index, (x, y) in enumerate(box)], text, score)
            for box, text, score in found
        ]
        return taller, timings

    run.detect._detector._reader = boxes_a_little_tall
    run.at(0)
    for offset in walk(6, EMAIL_TOP, 6):
        run.at(offset)
    for _ in range(3):
        run.at(EMAIL_TOP)  # stopped with the line touching the top, which forgets the edges
    for offset in walk(EMAIL_TOP + 3, EMAIL_TOP + 21, 3):
        output = run.at(offset)
        assert shown_secret_pixels(run.scene, offset, output) == 0, offset


def test_a_cover_ends_once_its_text_has_scrolled_off_the_edge():
    run = Run()
    run.at(0)
    for offset in walk(40, 320, 40):
        run.at(offset)
    assert "email" not in {d.category for d in run.secrets()}
    output = run.at(280)
    assert unexplained_cover(run, 280, output) == 0  # nothing left pinned to the edge


def test_reversing_a_scroll_does_not_join_both_edges_into_one_band():
    run = Run()
    run.at(0)
    for offset in walk(6, 60, 6):
        run.at(offset)
    for offset in walk(54, 12, -6):
        output = run.at(offset)
        assert sum(d.h for d in run.unread()) < 150
        assert covered_rows(output, 300).size < 300  # the page stays mostly visible


def test_a_scrolling_pane_leaves_the_still_pane_beside_it_alone():
    scene = ScrollScene()
    run = Run(scene)
    run.at(0)
    phone_box = scene.secret_boxes(0)[FAKE_PHONE]
    for offset in [*walk(6, 180, 6), *walk(180, 60, -12)]:
        output = run.at(offset)
        assert shown_secret_pixels(scene, offset, output) == 0
        phone = next(d for d in run.secrets() if d.category == "phone")
        assert (phone.x, phone.y) == phone_box[:2]
        column = scene.side[0] + 10 + 5  # inside the still pane, under its first line
        assert covered_rows(output, column).size == LINE_HEIGHT + 2 * PADDING


def test_still_text_inside_the_scrolling_areas_bounds_stays_where_it_is():
    scene = ScrollScene(secrets={}, side=(680, 940))
    run = Run(scene)
    run.at(0)
    phone_box = scene.secret_boxes(0)[FAKE_PHONE]
    for index, offset in enumerate(walk(6, 120, 6)):
        image = scene.image(offset)
        image[300, 950 + index % 2] = 255  # stretches the changed area past the still pane
        output = run.show(image)[1]
        assert shown_secret_pixels(scene, offset, output) == 0
        phone = next(d for d in run.secrets() if d.category == "phone")
        assert (phone.x, phone.y) == phone_box[:2]


def test_new_text_coming_onto_the_screen_stays_covered_until_it_is_read():
    scene = ScrollScene(secrets={13: FAKE_EMAIL})  # just below the first screenful
    run = Run(scene)
    run.at(0)
    for offset in walk(6, 90, 6):
        output = run.at(offset)
        assert shown_secret_pixels(scene, offset, output) == 0
        assert run.unread()


def test_read_rows_lose_their_cover_on_the_same_frame():
    run = Run()
    run.at(0)
    heights = []
    for offset in walk(6, 200, 6):
        run.at(offset)
        heights.append(sum(d.h for d in run.unread()))
    assert max(heights) < 96  # the frame that reaches 96 rows reads them and cuts back
    assert heights[-1] < max(heights)
    assert min(heights[16:]) <= 24 + 6  # only the rows after the last whole line stay


def test_stopping_clears_the_unread_strip_with_no_ghost_band():
    run = Run()
    run.at(0)
    for offset in walk(6, 36, 6):
        run.at(offset)
    assert run.unread()
    output = run.at(30)
    assert not run.unread()
    assert covered_rows(output, 600).size == 0  # no secret there, so nothing covers it


def test_a_secret_in_the_unread_strip_keeps_its_cover_after_the_read():
    scene = ScrollScene(secrets={14: FAKE_EMAIL})
    run = Run(scene)
    run.at(0)
    for offset in walk(6, 78, 6):
        run.at(offset)
    output = run.at(72)  # scrolling stops with the whole line on screen
    assert not run.unread()
    assert [d.category for d in run.secrets()] == ["email", "phone"]
    assert shown_secret_pixels(scene, 72, output) == 0


def test_unread_covers_last_one_frame_and_secrets_keep_the_shared_ttl():
    run = Run()
    run.at(0)
    run.at(6)
    assert {d.ttl_ms for d in run.unread()} == {UNREAD_TTL_MS}
    assert {d.ttl_ms for d in run.secrets()} == {TTL_MS}


def blink_beside(image, left, top):
    """Something small changes next to the text, so its line gets read again."""
    changed = image.copy()
    changed[top + 8 : top + 12, left - 6 : left - 3] = 255
    return changed


def test_a_missed_read_on_unchanged_pixels_keeps_the_cover():
    run = Run()
    image = run.scene.image(0)
    run.show(image)
    left, top, *_ = run.scene.secret_boxes(0)[FAKE_EMAIL]
    run.reader.blind.add(FAKE_EMAIL)
    for index in range(40):
        current = blink_beside(image, left, top) if index % 2 else image
        output = run.show(current)[1]
        assert shown_secret_pixels(run.scene, 0, output) == 0
        assert "email" in {d.category for d in run.secrets()}


def test_a_lost_secret_is_checked_again_until_it_is_found():
    run = Run()
    image = run.scene.image(0)
    run.show(image)
    left, top, *_ = run.scene.secret_boxes(0)[FAKE_EMAIL]
    run.reader.blind.add(FAKE_EMAIL)
    run.show(blink_beside(image, left, top))
    reads = run.reader.calls
    for _ in range(40):
        run.show(blink_beside(image, left, top))
    assert run.reader.calls > reads  # still looking, with nothing else changing
    assert run.detect.stats["checks"] >= 1
    run.reader.blind.clear()
    for _ in range(20):
        run.show(blink_beside(image, left, top))
    assert run.detect.stats["checks"] >= 1
    checks = run.detect.stats["checks"]
    for _ in range(40):
        run.show(blink_beside(image, left, top))
    assert run.detect.stats["checks"] == checks  # found again, so no more checking


def test_a_lost_secret_keeps_moving_with_a_scroll():
    run = Run()
    image = run.scene.image(0)
    run.show(image)
    left, top, *_ = run.scene.secret_boxes(0)[FAKE_EMAIL]
    run.reader.blind.add(FAKE_EMAIL)
    run.show(blink_beside(image, left, top))
    assert "email" in {d.category for d in run.secrets()}
    run.show(image)
    for offset in walk(6, 90, 6):
        output = run.at(offset)
        assert shown_secret_pixels(run.scene, offset, output) == 0
        cover = next(d for d in run.secrets() if d.category == "email")
        assert abs(cover.y - (EMAIL_TOP - offset)) <= 1


def replaced_with_safe_text(run, offset=0):
    """The same page, with a plain line where the fake email was."""
    image = run.scene.image(offset)
    left, top, right, bottom = run.scene.secret_boxes(offset)[FAKE_EMAIL]
    image[top:bottom, left:right] = BACKGROUND
    image[top:bottom, left : left + 80] = (200, 99, 255)  # unknown to the fake OCR
    return image


def test_safe_replacement_clears_the_cover_only_after_a_second_read():
    run = Run()
    run.at(0)
    image = replaced_with_safe_text(run)
    run.show(image)
    assert "email" in {d.category for d in run.secrets()}  # one read isn't proof
    run.show(image)
    assert "email" not in {d.category for d in run.secrets()}


def test_a_removal_is_for_the_content_that_was_checked():
    run = Run()
    run.at(0)
    run.show(replaced_with_safe_text(run))
    assert "email" in {d.category for d in run.secrets()}
    # The secret comes straight back before the cover has come off.
    output = run.at(0)
    assert shown_secret_pixels(run.scene, 0, output) == 0
    assert "email" in {d.category for d in run.secrets()}


def test_a_cover_with_no_proof_grows_while_it_is_unsure():
    run = Run()
    run.at(0)
    settled = next(d for d in run.secrets() if d.category == "email")
    run.show(replaced_with_safe_text(run))
    unsure = next(d for d in run.secrets() if d.category == "email")
    assert unsure.y < settled.y and unsure.y + unsure.h > settled.y + settled.h


def test_a_secret_that_scrolls_away_ends_its_cover():
    run = Run()
    run.at(0)
    for offset in walk(40, 360, 40):
        run.at(offset)
    assert "email" not in {d.category for d in run.secrets()}


def test_text_that_jumps_without_a_clean_scroll_leaves_no_ghost():
    scene = ScrollScene(secrets={6: FAKE_EMAIL})
    run = Run(scene)
    run.at(0)
    target = 6 * LINE_PITCH - 40  # the email ends up on the first visible line
    image = scene.image(target)
    image[:, :20] = (10, 200, 40)  # and a wall of change rules out a clean scroll
    image[200:500, 300:630] = np.random.default_rng(1).integers(0, 255, (300, 330, 3))
    output = run.show(image)[1]
    covers = [d for d in run.secrets() if d.category == "email"]
    assert len(covers) == 1
    assert covers[0].y == EMAIL_TOP - target
    assert shown_secret_pixels(scene, target, output) == 0


def test_stationary_ocr_wobble_stays_one_steady_bounded_cover():
    run = Run(wobble=2)
    image = run.scene.image(0)
    left, top, right, bottom = run.scene.secret_boxes(0)[FAKE_EMAIL]
    run.show(image)
    edges = set()
    for index in range(60):
        output = run.show(blink_beside(image, left, top) if index % 2 else image)[1]
        emails = [d for d in run.secrets() if d.category == "email"]
        assert len(emails) == 1  # never a pile of near-identical boxes
        cover = emails[0]
        assert cover.x <= left + 2 and cover.y <= top + 2
        assert cover.x + cover.w >= right - 2 and cover.y + cover.h >= bottom - 2
        assert cover.x >= left - 3 and cover.y >= top - 3  # bounded by the wobble itself
        assert cover.x + cover.w <= right + 3 and cover.y + cover.h <= bottom + 3
        assert shown_secret_pixels(run.scene, 0, output) == 0
        edges.add(cover.y)
    assert len(edges) <= 5


def test_moving_text_with_no_scroll_keeps_exactly_one_padded_cover():
    scene = ScrollScene(secrets={6: FAKE_EMAIL}, side_secrets={}, lines=8)
    run = Run(scene)
    line = next(line for line in scene.lines if line.text == FAKE_EMAIL)

    def page(top):
        image = np.full((scene.height, scene.width, 3), BACKGROUND, np.uint8)
        image[top : top + LINE_HEIGHT, 60 : 60 + line.w] = line.colour
        return image

    for step in range(30):
        top = EMAIL_TOP + step * 6
        output = run.show(page(top))[1]
        rows = covered_rows(output, 100)
        assert rows.size == LINE_HEIGHT + 2 * PADDING, step
        assert rows[0] == top - PADDING


def test_failed_scan_holds_then_recovers_with_the_cover_still_on():
    run = Run()
    image = run.scene.image(0)
    safe = run.show(image)[1]

    def broken(_):
        raise RuntimeError

    run.detect._detector._reader = broken
    changed = image.copy()
    changed[0:4, 0:4] = 255
    held = run.show(changed)[1]
    assert run.compositor.held_frames == 1
    assert (held == safe).all()
    # OCR comes back unable to see the secret, on a full read of the next frame.
    run.detect._detector._reader = run.reader
    run.reader.blind.add(FAKE_EMAIL)
    recovered = run.show(image)[1]
    assert shown_secret_pixels(run.scene, 0, recovered) == 0


def test_resize_drops_every_position_from_the_old_size():
    run = Run()
    run.at(0)
    assert run.secrets()
    small = ScrollScene(secrets={}, side_secrets={}, width=640, height=360, pane=(20, 400),
                        side=(420, 620))
    run.reader.scene = small
    output = run.show(small.image(0))[1]
    assert run.found == []
    assert (output == small.image(0)).all()


def test_a_new_source_drops_every_position_from_the_old_one():
    run = Run()
    run.at(0)
    assert run.secrets()
    run.reader.blind.update({FAKE_EMAIL, FAKE_PHONE})
    run.show(run.scene.image(0), source="webcam")
    assert not run.secrets()


def test_new_settings_drop_covers_from_the_old_settings():
    scene = ScrollScene(secrets={6: FAKE_EMAIL}, side_secrets={})
    detector = Detector(reader=SceneReader(scene))
    image = scene.image(0)
    assert detector.detect(image)
    detector.configure(enabled={"phone"})
    assert detector.detect(image) == []


def test_a_replaced_scene_loses_the_old_covers_within_two_frames():
    first = ScrollScene()
    second = ScrollScene(secrets={}, side_secrets={})
    run = Run(first)
    run.at(0)
    assert run.secrets()
    other = np.full_like(first.image(0), 90)
    other[100:140, 50:500] = 200
    run.show(other)
    output = run.show(other)[1]
    assert not run.secrets()
    assert (output == other).all()
    assert second.secret_boxes(0) == {}


def test_failed_scan_on_a_new_size_does_not_keep_old_positions():
    run = Run()
    run.at(0)

    def broken(_):
        raise RuntimeError

    small = ScrollScene(secrets={}, side_secrets={}, width=640, height=360, pane=(20, 400),
                        side=(420, 620))
    run.detect._detector._reader = broken
    with pytest.raises(DetectionError):
        run.detect(Frame(small.image(0), 0.0, "screen", 99))
    run.detect._detector._reader = SceneReader(small)
    assert run.detect(Frame(small.image(0), 0.0, "screen", 100)) == []


def test_the_compositor_refuses_results_for_another_frame():
    run = Run()
    frame, _ = run.show(run.scene.image(0))
    stale = list(run.found)
    next_frame = Frame(run.scene.image(0), 1.0, "screen", frame.frame_id + 1)
    output = run.compositor.process(next_frame, lambda f: stale)
    assert run.compositor.held_frames == 1
    assert not (output == next_frame.image).all()


@pytest.mark.parametrize(
    "size", [(540, 960), (1080, 1920), (982, 1512), (360, 640)], ids=lambda s: f"{s[1]}x{s[0]}"
)
def test_secret_stays_opaque_after_the_final_resize_and_letterbox(size):
    height, width = size
    scene = ScrollScene(
        width=width,
        height=height,
        pane=(40, width * 2 // 3),
        side=(width * 2 // 3 + 20, width - 10),
        lines=120,
    )
    run = Run(scene)
    scale = min(OUTPUT_WIDTH / width, OUTPUT_HEIGHT / height)
    pad_x = (OUTPUT_WIDTH - round(width * scale)) // 2
    pad_y = (OUTPUT_HEIGHT - round(height * scale)) // 2
    run.at(0)
    for offset in walk(6, 150, 6):
        fitted = fit(run.at(offset), OUTPUT_WIDTH, OUTPUT_HEIGHT)
        assert fitted.shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)
        for left, top, right, bottom in scene.secret_boxes(offset).values():
            x1, x2 = int(left * scale) + pad_x, int(np.ceil(right * scale)) + pad_x
            y1, y2 = int(top * scale) + pad_y, int(np.ceil(bottom * scale)) + pad_y
            assert not fitted[y1:y2, x1:x2].any(), (offset, left, top)


def test_keeps_working_for_a_retina_scale_detector():
    scene = ScrollScene(width=1920, height=1080, pane=(80, 1280), side=(1360, 1880), lines=120)
    reader = SceneReader(scene)
    detect = TextDetector(reader=reader, pixel_scale=2)
    compositor = Compositor(padding=PADDING * 2)
    for index, offset in enumerate(walk(0, 240, 12)):
        frame = scene.frame(offset, frame_id=index)
        output = compositor.process(frame, detect)
        assert shown_secret_pixels(scene, offset, output) == 0
    assert compositor.held_frames == 0


class ShrunkPageReader:
    """Fake OCR for a shrunk scene: finds each line by the solid middle of its blocks."""

    def __init__(self, scene, scale):
        self.scene = scene
        self.scale = scale

    def __call__(self, image):
        found = []
        for line in self.scene.lines:
            rows, columns = np.nonzero((image == np.array(line.colour, np.uint8)).all(axis=2))
            if rows.size == 0:
                continue
            top, bottom = rows.min(), rows.max() + 1
            left, right = columns.min(), columns.max() + 1
            if bottom - top >= line.h * self.scale - 3 and right - left >= line.w * self.scale - 8:
                corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
                found.append((corners, line.text, 0.9))
        return found, None


@pytest.mark.parametrize("device_pixels", [10, 13, 17])
def test_a_long_scroll_in_fractions_of_a_pixel_keeps_the_cover_on_its_text(device_pixels):
    """Shrinking a Retina capture makes scrolls fractional, like 6.35 px a frame.

    Rounding each frame's shift used to slide the cover 14.7 px off its text at that
    speed, which is further than the padding.
    """
    scale = 1920 / 3024
    scene = ScrollScene(
        secrets={10: FAKE_EMAIL}, side_secrets={}, width=1512, height=982, pane=(40, 1200),
        side=(1240, 1500), lines=60,
    )
    line = next(line for line in scene.lines if line.text == FAKE_EMAIL)
    detect = TextDetector(reader=ShrunkPageReader(scene, scale))

    def shrunk(offset, frame_id):
        image = cv2.resize(
            scene.image(offset), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        return Frame(image, 0.0, "screen", frame_id)

    detect(shrunk(0, 0))
    for index in range(1, line.y // device_pixels - 1):
        offset = index * device_pixels
        cover = next(d for d in detect(shrunk(offset, index)) if d.category == "email")
        assert abs(cover.y - (line.y - offset) * scale) < PADDING / 2, index


def test_two_secrets_on_screen_move_independently_of_each_other():
    scene = ScrollScene(secrets={6: FAKE_EMAIL, 8: FAKE_KEY})
    run = Run(scene)
    run.at(0)
    for offset in walk(6, 72, 6):
        run.at(offset)
        found = {d.category: d for d in run.secrets()}
        assert found["email"].y == 20 + 6 * LINE_PITCH - offset
        assert found["api_key"].y == 20 + 8 * LINE_PITCH - offset
