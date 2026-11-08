"""A secret found late, but before its first frame goes out, is covered from that frame.

Frames wait unfinished in the delay (eclipse/compositor/retain.py) until their output
deadline is near, and covers found in later frames are traced back onto them. These
tests drive the real RetainedFrames, Compositor and DelayedPipeline. Timing is given
explicitly (take(due) is the clock) except in the few whole-pipeline runs, which keep
wide margins. Glyph pixels are known exactly, and every check is on the final pixels.
All secrets are fake.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.compositor.retain import (
    HIDDEN, SHOWN, UNEXPLAINED, Retained, RetainedFrames, compare,
)
from eclipse.text_detect import TextDetector
from eclipse.text_detect.detection import _ink_boxes
from eclipse.text_detect.incoming import IncomingRows
from eclipse.types import Detection, Frame

WHITE = 255
GLYPH_W, GLYPH_H = 60, 12


def glyphs(seed: int = 7) -> np.ndarray:
    """A block of fake 'text': dark strokes on white, with every row and column inked."""
    rng = np.random.default_rng(seed)
    block = rng.random((GLYPH_H, GLYPH_W)) < 0.35
    block[:, ::5] = True  # every column has ink somewhere, so no gap hides a fragment
    block[::3, :] |= rng.random((GLYPH_H // 3 + 1, GLYPH_W))[: block[::3].shape[0]] < 0.5
    block[np.arange(GLYPH_H), np.arange(GLYPH_H) % GLYPH_W] = True  # every row too
    return block


SECRET = glyphs()


def secret_box(x: int, y: int, frame_id: int, pad: int = 2) -> Detection:
    return Detection(x - pad, y - pad, GLYPH_W + 2 * pad, GLYPH_H + 2 * pad, "api_key", 1.0, 500, frame_id)


def record(image, frame_id, found, motions=(), source="screen", generation=0) -> Retained:
    frame = Frame(image, frame_id / 60, source, frame_id)
    return Retained(frame, True, image.nbytes, found, generation, 0.0, list(motions) if motions is not None else None)


def draw(compositor: Compositor, rec: Retained) -> np.ndarray:
    boxes = rec.boxes()
    if rec.found is None:
        def failed(_frame):
            raise RuntimeError("scan failed")
        return compositor.render(rec.frame, failed, rec.generation).image
    return compositor.render(rec.frame, lambda _frame: boxes, rec.generation).image


def shown_glyphs(raw: np.ndarray, output: np.ndarray, mask: np.ndarray) -> int:
    """Secret glyph pixels the output shows exactly as the source did."""
    return int((mask & (output == raw).all(axis=2)).sum())


def finish_all(frames: RetainedFrames, compositor=None):
    frames.flush()  # the stream ends: trace the last frame's finds too
    compositor = compositor or Compositor(padding=0)
    out = []
    while (rec := frames.take(float("inf"))) is not None:
        out.append((rec, draw(compositor, rec)))
    return out


# --- Scenes: images plus each frame's exact glyph mask -------------------------------

def filler_page(height, width, seed=3):
    """White page with grey filler lines every 24 rows, leaving room for the secret."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), WHITE, np.uint8)
    for top in range(4, height - 10, 24):
        line = rng.random((8, width - 20)) < 0.3
        page[top : top + 8, 10 : width - 10][line] = 90
    return page


def scrolling_pane(frames, speed, secret_row, pane=(20, 180), width=300, start=0):
    """A pane (rows pane[0]..pane[1]) over a page that scrolls up `speed` rows a frame.
    Above and below the pane is window chrome. The secret sits at page row secret_row,
    in a gap between filler lines, so it enters at the pane's bottom edge row by row."""
    top, bottom = pane
    view = bottom - top
    page = filler_page(secret_row + view + 200, width)
    page[secret_row - 4 : secret_row + GLYPH_H + 4] = WHITE  # a gap for it
    page[secret_row : secret_row + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    images, masks, offsets = [], [], []
    offset = start
    for index in range(frames):
        if index:
            offset += speed[index] if isinstance(speed, list) else speed
        image = np.full((bottom + 20, width, 3), 140, np.uint8)
        image[: top, :, 0] = 200  # chrome above, not the page's colours
        image[bottom:, :, 1] = 60  # chrome below
        image[top:bottom] = page[offset : offset + view]
        mask = np.zeros(image.shape[:2], bool)
        y = top + secret_row - offset
        for row in range(GLYPH_H):
            if top <= y + row < bottom:
                mask[y + row, 100 : 100 + GLYPH_W] = SECRET[row]
        images.append(image)
        masks.append(mask)
        offsets.append(offset)
    return images, masks, offsets


def textured_desktop(height, width, seed=11):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 90, (height, width, 3), dtype=np.uint8)


def window(height=80, width=160, seed=5):
    win = filler_page(height, width, seed)
    win[30 : 30 + GLYPH_H + 8, :] = WHITE
    win[34 : 34 + GLYPH_H, 50 : 50 + GLYPH_W][SECRET] = 20
    return win, (50, 34)  # where the secret is in the window


def paste(image, mask, win, glyph_at, x, y):
    """Paste a window at x, y, clipped to the frame; mark its secret's glyph pixels."""
    height, width = image.shape[:2]
    h, w = win.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    if x2 <= x1 or y2 <= y1:
        return None
    image[y1:y2, x1:x2] = win[y1 - y : y2 - y, x1 - x : x2 - x]
    gx, gy = x + glyph_at[0], y + glyph_at[1]
    for row in range(GLYPH_H):
        for col in range(GLYPH_W):
            if SECRET[row, col] and 0 <= gy + row < height and 0 <= gx + col < width:
                mask[gy + row, gx + col] = True
    return gx, gy


def window_area(x, y, dx, dy, w, h, shape):
    """The detector's moved area for a window at x, y that moved dx, dy: old and new place."""
    height, width = shape[:2]
    left, top = max(min(x, x - dx), 0), max(min(y, y - dy), 0)
    right, bottom = min(max(x + w, x + w - dx), width), min(max(y + h, y + h - dy), height)
    return (left, top, right - left, bottom - top, dx, dy)


# --- 1. Found after the old 0.3 s finish, before the output deadline ------------------

def test_secret_found_late_is_covered_from_its_first_output_frame():
    """Frames 0-29 show a still secret the scanner misses; frame 30, 0.5 s later at 60
    fps, finds it. Under the old 0.3 s finish, frames 0-11 had already been drawn."""
    image = filler_page(120, 300)
    image[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    mask = np.zeros(image.shape[:2], bool)
    mask[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W] = SECRET
    frames = RetainedFrames()
    for index in range(31):
        found = [secret_box(100, 50, index)] if index == 30 else []
        rec = record(image.copy(), index, found)
        frames.add(rec)
        frames.trace(rec)
    # Nothing has been finished yet: every frame is still inside a 1 s delay.
    assert frames.too_late == 0
    for rec, output in finish_all(frames):
        assert shown_glyphs(rec.frame.image, output, mask) == 0, rec.frame.frame_id


def test_a_cover_found_after_a_frame_was_finished_is_counted_not_lost_silently():
    image = filler_page(120, 300)
    image[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    frames = RetainedFrames()
    for index in range(3):
        frames.add(record(image.copy(), index, []))
    finished = frames.take(0.0)  # frame 0 is due at t=0 on this clock
    assert finished.frame.frame_id == 0
    late = record(image.copy(), 3, [secret_box(100, 50, 3)])
    frames.add(late)
    frames.trace(late)
    frames.flush()
    assert len(frames.oldest().added) == 1  # frame 1 still waiting: covered
    assert frames.too_late == 1  # frame 0 already went to output


class LateFinder:
    """A detector that sees the planted secret only from frame `first` on, and keeps
    its scans quick. Answers settled() with what it found, as a settled frame would."""

    def __init__(self, first, box=(20, 20, 40, 10)):
        self.first = first
        self.box = box

    def __call__(self, frame):
        if frame.frame_id < self.first:
            return []
        return [Detection(*self.box, "api_key", 1.0, 500, frame.frame_id)]

    def settled(self, frame, forget=True):
        return None


class SecretSource:
    """Frames whose top-left pixel holds their id, with a fake secret on all of them."""

    def __init__(self):
        self.next_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        image = np.full((60, 80, 3), 200, np.uint8)
        image[0, 0] = (self.next_id // 256, self.next_id % 256, 0)
        image[22:28, 22:58] = 123  # the secret's pixels
        frame = Frame(image, time.monotonic(), "screen", self.next_id)
        self.next_id += 1
        return frame


class Sink:
    def __init__(self):
        self.images = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        self.images.append(image)
        time.sleep(0.002)


def test_whole_pipeline_covers_frames_scanned_before_the_secret_was_found():
    """The detector first finds the secret on frame 15, 0.15 s after frame 0: past a
    0.05 s look-ahead, well inside a 0.4 s delay. No frame may show it."""
    sink = Sink()
    pipeline = DelayedPipeline(
        SecretSource(), sink, LateFinder(15), Compositor(padding=0), 0.4, 100, 30,
        look_ahead=0.05,
    )
    stats = pipeline.run()
    sent = sink.images[:-1]
    ids = [int(image[0, 0, 0]) * 256 + int(image[0, 0, 1]) for image in sent]
    assert 0 in ids  # the earliest frames really went out
    assert not any((image == 123).any() for image in sink.images)
    assert stats.traced_frames >= 14
    assert stats.traced_too_late == 0
    assert min(stats.slack) > 0  # every frame drawn before it was due


# --- 2. Text entering one pixel row at a time ----------------------------------------

def scroll_motion(pane, width, dy):
    top, bottom = pane
    return (0, top, width, bottom - top, 0, dy)


def run_scroll(images, offsets, found_at, box_at, pane=(20, 180), hint=True):
    frames = RetainedFrames()
    width = images[0].shape[1]
    for index, image in enumerate(images):
        motions = [] if index == 0 else (
            [scroll_motion(pane, width, -(offsets[index] - offsets[index - 1]))]
            if offsets[index] != offsets[index - 1] else []
        )
        found = [secret_box(*box_at(index), index)] if index in found_at else []
        rec = record(image, index, found, motions if hint else None)
        frames.add(rec)
        frames.trace(rec)
    return finish_all(frames)


def test_text_entering_one_row_at_a_time_is_covered_from_its_first_row():
    pane = (20, 180)
    secret_row = 200
    # Starts with the secret two rows below the pane's bottom edge.
    start = secret_row - (pane[1] - pane[0]) - 2
    images, masks, offsets = scrolling_pane(40, 1, secret_row, pane, start=start)
    first_visible = next(i for i, m in enumerate(masks) if m.any())
    whole = next(i for i, m in enumerate(masks) if m.sum() == SECRET.sum())
    assert 0 < first_visible < whole  # fragments show for many frames before it's whole
    found_at = set(range(whole, 40))  # found once every row shows, then kept

    def box_at(index):
        return 100, pane[0] + secret_row - offsets[index]

    results = run_scroll(images, offsets, found_at, box_at, pane)
    for (rec, output), mask in zip(results, masks):
        index = rec.frame.frame_id
        if index > whole:
            continue  # not found there in this test: the detector's own tracks cover it
        assert shown_glyphs(rec.frame.image, output, mask) == 0, index
    # No cover before the first row showed: tracing stops where the text wasn't visible.
    for rec, _ in results[:first_visible]:
        assert not rec.added


def test_scroll_reversal_maps_back_through_both_directions():
    pane = (20, 180)
    speeds = [0] + [3] * 12 + [-3] * 12  # down, then back up
    images, masks, offsets = scrolling_pane(25, speeds, 120, pane, start=0)
    assert all(m.any() for m in masks)

    def box_at(index):
        return 100, pane[0] + 120 - offsets[index]

    results = run_scroll(images, offsets, {24}, box_at, pane)
    for (rec, output), mask in zip(results, masks):
        assert shown_glyphs(rec.frame.image, output, mask) == 0, rec.frame.frame_id


def test_without_the_scroll_hint_moved_text_is_not_mapped_by_guesswork():
    """Missing motion metadata: only an unmoved place can be checked, and it fails, so
    nothing is invented. The frames then rely on what their own scans found."""
    pane = (20, 180)
    images, masks, offsets = scrolling_pane(10, 3, 120, pane)

    def box_at(index):
        return 100, pane[0] + 120 - offsets[index]

    results = run_scroll(images, offsets, {9}, box_at, pane, hint=False)
    assert not any(rec.added for rec, _ in results)


# --- 3. Window drags ------------------------------------------------------------------

def drag_scene(frames, dx, dy, start):
    win, glyph_at = window()
    shape = (240, 360, 3)
    images, masks, places, motions = [], [], [], []
    x, y = start
    for index in range(frames):
        if index:
            x, y = x + dx, y + dy
        image = textured_desktop(*shape[:2])
        mask = np.zeros(shape[:2], bool)
        place = paste(image, mask, win, glyph_at, x, y)
        images.append(image)
        masks.append(mask)
        places.append(place)
        motions.append([] if index == 0 else [window_area(x, y, dx, dy, *win.shape[1::-1], shape)])
    return images, masks, places, motions


def test_diagonal_drag_in_from_the_corner_covers_the_first_fragments():
    # The window comes in from above and left of the frame, so its secret first shows
    # as a corner fragment.
    images, masks, places, motions = drag_scene(30, 4, 3, (-110, -70))
    first_visible = next(i for i, m in enumerate(masks) if m.any())
    whole = next(i for i, m in enumerate(masks) if m.sum() == SECRET.sum())
    assert first_visible < whole
    frames = RetainedFrames()
    for index, image in enumerate(images):
        found = [secret_box(*places[index], index)] if index >= whole else []
        rec = record(image, index, found, motions[index])
        frames.add(rec)
        frames.trace(rec)
    for (rec, output), mask in zip(finish_all(frames), masks):
        if rec.frame.frame_id <= whole:
            assert shown_glyphs(rec.frame.image, output, mask) == 0, rec.frame.frame_id


def two_windows(frames):
    """Window A drags right, window B drags down, each by its own amount."""
    win_a, at_a = window(seed=5)
    win_b, at_b = window(seed=9)
    shape = (300, 420, 3)
    images, masks_b, places_b, moves_a, moves_b = [], [], [], [], []
    for index in range(frames):
        ax, ay = 10 + 4 * index, 20
        bx, by = 240, 60 + 3 * index
        image = textured_desktop(*shape[:2])
        scratch = np.zeros(shape[:2], bool)
        paste(image, scratch, win_a, at_a, ax, ay)
        mask = np.zeros(shape[:2], bool)
        places_b.append(paste(image, mask, win_b, at_b, bx, by))
        images.append(image)
        masks_b.append(mask)
        moves_a.append(window_area(ax, ay, 4, 0, 160, 80, shape))
        moves_b.append(window_area(bx, by, 0, 3, 160, 80, shape))
    return images, masks_b, places_b, moves_a, moves_b


@pytest.mark.parametrize("hints", ["both", "only_a", "global_a"])
def test_independent_windows_never_borrow_each_others_motion(hints):
    images, masks_b, places_b, moves_a, moves_b = two_windows(12)
    frames = RetainedFrames()
    for index, image in enumerate(images):
        if index == 0:
            motions = []
        elif hints == "both":
            motions = [moves_a[index], moves_b[index]]
        elif hints == "only_a":
            motions = [moves_a[index]]  # the detector follows one move a frame
        else:
            motions = [(0, 0, image.shape[1], image.shape[0], 4, 0)]  # one global shift
        found = [secret_box(*places_b[index], index)] if index == 11 else []
        rec = record(image, index, found, motions)
        frames.add(rec)
        frames.trace(rec)
    results = finish_all(frames)
    for rec, _ in results:
        for cover in rec.added:
            # Every traced cover sits on B's secret in that frame, never elsewhere.
            gx, gy = places_b[rec.frame.frame_id]
            assert gx - 2 <= cover.x and cover.x + cover.w <= gx + GLYPH_W + 2
            assert gy - 2 <= cover.y and cover.y + cover.h <= gy + GLYPH_H + 2
    if hints == "both":
        for (rec, output), mask in zip(results, masks_b):
            assert shown_glyphs(rec.frame.image, output, mask) == 0, rec.frame.frame_id
    else:
        # B's own move wasn't given: nothing is traced rather than something misplaced.
        assert not any(rec.added for rec, _ in results)


# --- 4. A secret shown for one frame, then gone ----------------------------------------

def test_a_one_frame_secret_marked_unread_stays_covered_after_later_clean_frames():
    """Frame 1's scan couldn't read the rows (an 'unread' cover). Frames 2 and 3 no
    longer show the secret, and read clean. Nothing about them settles frame 1."""
    base = filler_page(120, 300)
    flash = base.copy()
    flash[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    mask = np.zeros(base.shape[:2], bool)
    mask[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W] = SECRET
    frames = RetainedFrames()
    unread = Detection(90, 40, 80, 32, "unread", 1.0, 0, 1)
    recs = [
        record(base.copy(), 0, []),
        record(flash, 1, [unread]),
        record(base.copy(), 2, []),
        record(base.copy(), 3, []),
    ]
    for rec in recs:
        frames.add(rec)
        frames.refresh(lambda _frame: None)  # nothing known to settle it
        frames.trace(rec)
    for rec, output in finish_all(frames):
        if rec.frame.frame_id == 1:
            assert shown_glyphs(rec.frame.image, output, mask) == 0


def test_incoming_rows_that_changed_before_a_clean_read_stay_unread():
    """The detector's own look-ahead: rows that showed a secret and then changed are
    lost, never cleared by a later frame's clean read of different pixels."""
    incoming = IncomingRows(10)
    size = (200, 300)
    strip = (0, 180, 300, 20, "unread")
    incoming.record(0, [strip], [(strip, 200, -1, [strip[:4]])], None, None, size)
    changed = np.zeros(size, bool)
    changed[185:195, 100:160] = True  # the secret vanished: unexplained change
    incoming.record(1, [], [], changed, None, size)
    incoming.record(2, [], [], np.zeros(size, bool), None, size)  # read clean
    settled = incoming.settled(0, forget=False)
    kept = [box for box in settled if box[4] in ("unread", "held")]  # never shown raw
    assert any(y <= 185 and y + h >= 195 for _, y, _, h, _ in kept)


def test_a_scan_that_fails_on_a_one_frame_secret_never_lets_that_frame_out_raw():
    """OCR failing (or timing out) on the frame that shows the secret: that frame goes
    out as the last safe picture, whatever later frames read."""
    base = filler_page(120, 300)
    flash = base.copy()
    flash[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    mask = np.zeros(base.shape[:2], bool)
    mask[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W] = SECRET
    frames = RetainedFrames()
    for rec in [record(base.copy(), 0, []), record(flash, 1, None), record(base.copy(), 2, [])]:
        frames.add(rec)
        frames.trace(rec)
    for rec, output in finish_all(frames):
        if rec.frame.frame_id == 1:
            assert shown_glyphs(rec.frame.image, output, mask) == 0
            assert (output == base).all()  # the last safe picture stood in for it


# --- 5. Held areas and confirmed covers -----------------------------------------------

def test_a_confirmed_cover_wins_over_a_held_area_on_top_of_it():
    compositor = Compositor(padding=0)
    compositor.render(Frame(np.full((80, 160, 3), 90, np.uint8), 0.0, "screen", 0), lambda _: [])
    now = Frame(np.full((80, 160, 3), 220, np.uint8), 0.1, "screen", 1)
    secret = Detection(40, 20, 40, 16, "api_key", 1.0, 0, 1)
    held = Detection(30, 15, 70, 30, "held", 1.0, 0, 1)
    out = compositor.render(now, lambda _: [secret, held]).image
    assert not out[20:36, 40:80].any()  # black, not the older picture's 90
    assert (out[15:20, 30:100] == 90).all()  # the rest of the held area is the old picture


def test_a_traced_cover_wins_over_a_held_pane():
    image = filler_page(120, 300)
    image[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    mask = np.zeros(image.shape[:2], bool)
    mask[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W] = SECRET
    frames = RetainedFrames()
    held = Detection(0, 0, 300, 120, "held", 1.0, 0, 1)
    first = record(image.copy(), 0, [])
    second = record(image.copy(), 1, [held])
    third = record(image.copy(), 2, [secret_box(100, 50, 2)])
    for rec in (first, second, third):
        frames.add(rec)
        frames.trace(rec)
    compositor = Compositor(padding=0)
    out = finish_all(frames, compositor)
    for rec, output in out:
        assert shown_glyphs(rec.frame.image, output, mask) == 0


# --- 6. Empty foreground, missing metadata, stale results, flush ------------------------

def faint_text(delta=6):
    image = np.full((100, 500, 3), 28, np.uint8)
    cv2.putText(image, "password=hunter2", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (28 + delta,) * 3, 1, cv2.LINE_8)
    return image


def test_faint_incoming_text_is_not_blank_on_an_exact_screen():
    image = faint_text()
    changed = np.any(image != 28, axis=2)
    assert changed.any()
    boxes = _ink_boxes(image, 30, 70, 0, 500, 1, floor=0)
    covered = np.zeros(changed.shape, bool)
    for x, y, w, h in boxes:
        covered[y : y + h, x : x + w] = True
    assert not (changed[30:70] & ~covered[30:70]).any()
    # The webcam's noise floor still ignores sensor noise, as before.
    assert _ink_boxes(image, 30, 70, 0, 500, 1, floor=8) == []


def test_a_screen_detector_holds_faint_incoming_rows():
    detector = TextDetector(reader=lambda image: ([], None))
    assert detector._detector._noise_floor == 0  # the floor its strips are inked with


def test_missing_metadata_falls_back_to_what_the_scan_reported():
    image = filler_page(120, 300)
    unread = Detection(0, 100, 300, 20, "unread", 1.0, 0, 0)
    rec = record(image, 0, [unread], motions=None)
    frames = RetainedFrames()
    frames.add(rec)
    frames.refresh(lambda _frame: None)
    assert rec.boxes() == [unread]
    frames.refresh(lambda _frame: [])  # settled clean, by the detector's own checks
    frames.refresh(lambda _frame: None)  # then forgotten: the last answer stays
    assert rec.boxes() == []


def test_a_result_for_another_frame_is_never_drawn_on_this_one():
    image = filler_page(120, 300)
    image[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    mask = np.zeros(image.shape[:2], bool)
    mask[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W] = SECRET
    compositor = Compositor(padding=0)
    rec = record(image, 5, [secret_box(100, 50, 5)])
    rec.settled = [secret_box(100, 50, 4)]  # a stale answer about frame 4
    output = draw(compositor, rec)
    assert shown_glyphs(image, output, mask) == 0  # held: black, as nothing is safe yet


def test_compare_matches_the_text_pixels_exactly():
    later = np.full((40, 80, 3), WHITE, np.uint8)
    later[10:22, 10:70][SECRET] = 20
    box, clip = (10, 10, GLYPH_W, GLYPH_H), (0, 0, 80, 40)
    assert compare(later, later.copy(), box, clip, 0, 0)[0] == SHOWN
    # Not there yet: different pixels, so never traced onto, whatever else is there.
    assert compare(later, np.full_like(later, WHITE), box, clip, 0, 0)[0] == UNEXPLAINED
    other = np.full_like(later, WHITE)
    other[10:22, 10:70][glyphs(seed=99)] = 20  # other text in the same place
    assert compare(later, other, box, clip, 0, 0)[0] == UNEXPLAINED
    # A caret's worth of difference still matches.
    caret = later.copy()
    caret[10:22, 69] = 0
    assert compare(later, caret, box, clip, 0, 0)[0] == SHOWN
    # Only background inside the area it came from: none of the text showed yet.
    assert compare(later, later.copy(), box, (0, 30, 80, 10), 0, 0)[0] == HIDDEN


def test_draining_on_stop_never_draws_anything():
    frames = RetainedFrames()
    frames.add(record(filler_page(40, 40), 0, []))
    drained = frames.drain()
    assert len(drained) == 1 and drained[0].finished
    assert frames.take(float("inf")) is None
    late = record(filler_page(40, 40), 1, [secret_box(5, 5, 1)])
    frames.add(late)
    frames.trace(late)
    assert not drained[0].added  # a finished frame never changes


def test_the_last_frames_of_a_stream_wait_for_their_deadline_with_their_covers():
    sink = Sink()
    pipeline = DelayedPipeline(
        SecretSource(), sink, LateFinder(9), Compositor(padding=0), 0.2, 100, 10,
        look_ahead=0.05,
    )
    stats = pipeline.run()
    assert not any((image == 123).any() for image in sink.images)
    assert stats.final_black_sent
    assert min(stats.latencies) >= 150  # not flushed out early


def test_a_stricter_policy_never_lets_an_earlier_generation_out():
    class Policy:
        strict = False

        def __call__(self, frame):
            return [Detection(20, 20, 40, 10, "api_key", 1.0, 500, frame.frame_id)] if self.strict else []

        def settled(self, frame, forget=True):
            return None

    policy = Policy()
    sink = Sink()
    pipeline = DelayedPipeline(
        SecretSource(), sink, policy, Compositor(padding=0), 0.3, 100, 40, look_ahead=0.05,
    )

    def tighten():
        policy.strict = True

    import threading

    timer = threading.Timer(0.12, lambda: pipeline.change_policy(tighten, stricter=True))
    timer.start()
    stats = pipeline.run()
    timer.join()
    after = [i for i, image in enumerate(sink.images) if not image.any()]
    first_black = after[0] if after else len(sink.images)
    # Once the change is made, only black or frames scanned under it go out.
    assert not any((image == 123).any() for image in sink.images[first_black:])
    assert stats.invalidated + stats.black_sends > 0


def test_detector_reports_the_moves_it_followed():
    """The hint the tracing uses comes from the detector's own follows."""
    win, _ = window(120, 260)
    desktop = textured_desktop(300, 500)
    detector = TextDetector(reader=lambda image: ([], None))
    for index, (x, y) in enumerate([(40, 40), (52, 49)]):
        image = desktop.copy()
        image[y : y + 120, x : x + 260] = win
        detector(Frame(image, index / 60, "screen", index))
    motions = detector.motions()
    assert motions and motions[0][4:] == (12, 9)


# --- 7. Where the text starts --------------------------------------------------------

def field_frames(letters_at, total, caret=False):
    """A bordered input field. The secret is typed one 6-column letter per frame from
    frame letters_at; with caret, a caret blinks beside it every other frame."""
    images, masks = [], []
    for index in range(total):
        image = filler_page(120, 300)
        image[40:72, 90:180] = WHITE
        image[40, 90:180] = image[71, 90:180] = 60  # the field's border
        image[40:72, 90] = image[40:72, 179] = 60
        typed = max(0, min(GLYPH_W, (index - letters_at + 1) * 6)) if letters_at >= 0 else GLYPH_W
        mask = np.zeros(image.shape[:2], bool)
        mask[50 : 50 + GLYPH_H, 100 : 100 + typed] = SECRET[:, :typed]
        image[mask] = 20
        if caret and index % 2:
            image[48:64, 100 + typed + 1] = 0
        images.append(image)
        masks.append(mask)
    return images, masks


def test_typed_text_is_traced_only_onto_frames_showing_it_whole_never_the_empty_field():
    """Letters typed one by one differ from the finished text, so they rely on their own
    frames' scans: a trace covers only the frames showing exactly what was found. The
    empty field before is never covered: nothing changes without a secret."""
    images, masks = field_frames(letters_at=5, total=20)
    whole = next(i for i, m in enumerate(masks) if m.sum() == SECRET.sum())
    frames = RetainedFrames()
    for index, image in enumerate(images):
        found = [Detection(88, 38, 94, 36, "api_key", 1.0, 500, index)] if index == 19 else []
        rec = record(image, index, found)
        frames.add(rec)
        frames.trace(rec)
    results = finish_all(frames)
    for (rec, output), mask in zip(results, masks):
        if rec.frame.frame_id >= whole:
            assert shown_glyphs(rec.frame.image, output, mask) == 0, rec.frame.frame_id
        else:
            assert not rec.added, rec.frame.frame_id


def test_a_still_secret_with_a_blinking_caret_is_covered_all_the_way_back():
    images, masks = field_frames(letters_at=-1, total=20, caret=True)
    frames = RetainedFrames()
    for index, image in enumerate(images):
        found = [secret_box(100, 50, index)] if index == 19 else []
        rec = record(image, index, found)
        frames.add(rec)
        frames.trace(rec)
    for (rec, output), mask in zip(finish_all(frames), masks):
        assert shown_glyphs(rec.frame.image, output, mask) == 0, rec.frame.frame_id


# --- 8. The eval scores what really went out ------------------------------------------

def test_scoring_does_not_trust_a_held_tag_on_raw_output():
    from eval.motion_flicker import _scored

    raw = np.full((20, 30, 3), 200, np.uint8)
    raw[5:15, 10:20] = 40  # the secret, new in this frame
    previous = np.full_like(raw, 200)
    held = np.zeros((20, 30), bool)
    held[:, :] = True
    image, scored, foreign = _scored(raw, raw.copy(), held, previous)
    truth = np.zeros((20, 30), bool)
    truth[5:15, 10:20] = True
    assert (truth & (scored == image).all(axis=2)).sum() == truth.sum()  # still counted
    assert foreign == truth.sum()
    # Held pixels that really are the picture sent before are left out, as before.
    _, scored, foreign = _scored(raw, previous.copy(), held, previous)
    assert not (truth & (scored == image).all(axis=2)).any() and foreign == 0


def test_negative_control_raw_output_tagged_held_fails_the_eval(monkeypatch):
    import eval.motion_flicker as flicker

    def draw(index):
        image = np.full((60, 90, 3), 200, np.uint8)
        mask = np.zeros((GLYPH_H, GLYPH_W), bool)
        if index >= 2:
            mask = SECRET.copy()
            image[20 : 20 + GLYPH_H, 10 : 10 + GLYPH_W][mask] = 20
        return image, [("api_key", 10, 20, mask, ("k", 0, 0))]

    scene = flicker.Scene("negative_control", 5, draw, shape=(60, 90))

    def raw_outputs(scene, look_ahead, scan_ms, retain=0):
        for index in range(scene.frames):
            image, secrets = scene.frame(index)
            scan_ms.append(0.0)
            yield image, secrets, image.copy(), np.ones(image.shape[:2], bool)

    monkeypatch.setattr(flicker, "_outputs", raw_outputs)
    result = flicker.run(scene)
    assert result["exposed_glyph_px"] > 0  # must fail: raw pixels went out
    assert result["first_exposed_frame"] == 2
    assert result["held_not_earlier_px"] > 0


def test_a_cover_carried_without_a_read_is_not_traced_back():
    """A cover the detector only carried (nothing under it was read this frame) says
    nothing new about earlier frames, so it isn't spread back onto them."""
    image = filler_page(120, 300)
    image[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    frames = RetainedFrames()
    for index in range(4):
        found = [secret_box(100, 50, index)] if index == 3 else []
        rec = record(image.copy(), index, found)
        rec.reads = [(0, 0, 50, 50)] if index == 3 else []  # read elsewhere only
        frames.add(rec)
        frames.trace(rec)
    frames.flush()
    assert frames.traced == 0
    frames = RetainedFrames()
    for index in range(4):
        found = [secret_box(100, 50, index)] if index == 3 else []
        rec = record(image.copy(), index, found)
        rec.reads = [(90, 40, 100, 40)] if index == 3 else []  # read where it is: found
        frames.add(rec)
        frames.trace(rec)
    frames.flush()
    assert frames.traced == 3  # frames 0-2


# --- 9. Nothing changes unless there is something to redact ---------------------------

def test_rows_unread_at_the_deadline_repeat_the_last_safe_frame_never_a_black_strip():
    base = filler_page(120, 300)
    frames = RetainedFrames()
    unread = Detection(0, 100, 300, 20, "unread", 1.0, 0, 1)
    first = record(base.copy(), 0, [])
    second = record(base.copy(), 1, [unread])
    second.frame.image[100:120] = 77  # what its unread rows show
    for rec in (first, second):
        frames.add(rec)
    assert first.drawing() == [] and second.drawing() is None
    compositor = Compositor(padding=0)
    out0 = draw(compositor, frames.take(float("inf")))
    held = compositor.hold(frames.take(float("inf")).frame)
    assert held.state == "held" and (held.image == out0).all()  # frame 0 again
    assert compositor.unchecked_frames == 1


class NeverSettles:
    """Reports incoming rows as unread on frame 5 only, and never settles them."""

    def __call__(self, frame):
        if frame.frame_id == 5:
            return [Detection(10, 40, 60, 10, "unread", 1.0, 0, frame.frame_id)]
        return []

    def settled(self, frame, forget=True):
        return None


def test_whole_pipeline_never_draws_unchecked_rows_black():
    sink = Sink()
    pipeline = DelayedPipeline(
        SecretSource(), sink, NeverSettles(), Compositor(padding=0), 0.2, 100, 12,
        look_ahead=0.05,
    )
    stats = pipeline.run()
    sent = sink.images[:-1]  # the last is the final black
    ids = [int(image[0, 0, 0]) * 256 + int(image[0, 0, 1]) for image in sent]
    assert 5 not in ids and 4 in ids  # frame 4 went out again in frame 5's place
    assert not any((image[40:50, 10:70] == 0).all() for image in sent)  # no black strip
    assert stats.unchecked_at_deadline == 1


def test_a_box_of_one_colour_is_still_traced_whole():
    image = np.full((60, 120, 3), 200, np.uint8)
    image[20:32, 20:100] = 123  # a found box filled by one colour, no background in it
    frames = RetainedFrames()
    for index in range(4):
        found = [Detection(20, 20, 80, 12, "api_key", 1.0, 500, index)] if index == 3 else []
        rec = record(image.copy(), index, found)
        frames.add(rec)
        frames.trace(rec)
    frames.flush()
    assert frames.traced == 3



def test_a_cover_found_on_one_frame_only_is_not_traced_back():
    """A cover the next frame doesn't confirm is most likely a misread. Traced back, it
    would put black on frames that showed no secret, so it covers only its own frame."""
    image = filler_page(120, 300)
    frames = RetainedFrames()
    for index in range(8):
        found = [secret_box(100, 52, index)] if index == 5 else []  # one frame, then gone
        rec = record(image.copy(), index, found)
        frames.add(rec)
        frames.trace(rec)
    assert frames.traced == 0
    confirmed = RetainedFrames()
    for index in range(8):
        found = [secret_box(100, 52, index)] if index >= 5 else []
        rec = record(image.copy(), index, found)
        confirmed.add(rec)
        confirmed.trace(rec)
    confirmed.flush()  # the stream ends before six frames have followed
    assert confirmed.traced == 5  # frames 0-4



def run_reread(frames_count, found_on, sighted_on, reread=True):
    """A still secret. found_on: frames whose scan reports its cover. sighted_on: frames
    whose reads found it again. With reread, every frame read its area again."""
    image = filler_page(120, 300)
    image[50 : 50 + GLYPH_H, 100 : 100 + GLYPH_W][SECRET] = 20
    frames = RetainedFrames()
    box = secret_box(100, 50, 0)
    for index in range(frames_count):
        found = [secret_box(100, 50, index)] if index in found_on else []
        rec = record(image.copy(), index, found)
        rec.reads = [(0, 0, 300, 120)] if reread or index == min(found_on) else []
        rec.sightings = [("api_key", 100, 50, GLYPH_W, GLYPH_H)] if index in sighted_on else []
        frames.add(rec)
        frames.trace(rec)
    frames.flush()
    return frames


def test_a_line_read_as_secret_on_some_frames_only_is_not_traced_back():
    """Re-read every frame, found on frame 10 and then only now and then: a misread. Its
    covers stay on the frames that found it, as on main, not spread over a second."""
    frames = run_reread(20, found_on=set(range(10, 20)), sighted_on={10, 13, 17})
    assert frames.traced == 0


def test_a_late_find_that_later_reads_keep_finding_is_traced_back():
    frames = run_reread(20, found_on=set(range(10, 20)), sighted_on=set(range(10, 20)))
    assert frames.traced == 10  # frames 0-9


def test_a_still_late_find_not_read_again_is_traced_back_while_its_cover_stays():
    frames = run_reread(20, found_on=set(range(10, 20)), sighted_on={10}, reread=False)
    assert frames.traced == 10


def test_detector_reports_what_its_reads_found_this_frame():
    image = np.full((80, 400, 3), 255, np.uint8)

    def reader(crop):
        return [([[10, 20], [300, 20], [300, 50], [10, 50]], "contact: alex.fake@example.com", 1.0)], None

    detector = TextDetector(reader=reader)
    detector(Frame(image, 0.0, "screen", 0))
    found = detector.sightings()
    assert found and found[0][0] == "email"
    detector(Frame(image.copy(), 1 / 60, "screen", 1))  # nothing changed: nothing read
    assert detector.sightings() == []
