"""Look-ahead for incoming rows: a strip goes out clear only where later frames read it."""

import numpy as np

from eclipse.text_detect.incoming import IncomingRows, ScrollStep

HEIGHT, WIDTH = 200, 300
SIZE = HEIGHT, WIDTH
VIEW = (0, 200)  # the pane is the whole picture
LINE = 10  # rows a line needs to be whole


def nothing_changed():
    return np.zeros((HEIGHT, WIDTH), bool)


def scroll(shift, unexplained=None):
    """Content moved up by -shift rows, every pixel checked from row 0 down."""
    rows = HEIGHT - abs(shift)
    if unexplained is None:
        unexplained = np.zeros((rows, WIDTH), bool)
    return ScrollStep(float(shift), 0, WIDTH, 0, unexplained, VIEW)


def strip(top, bottom):
    """A strip at the bottom edge, as rows scroll up into view."""
    return ((0, top, WIDTH, bottom - top, 'unread'), VIEW[1], -1)


def first_frame(incoming, top=180):
    box, edge, inward = strip(top, HEIGHT)
    incoming.record(0, [box, (50, 20, 40, 10, 'email')], [(box, edge, inward, [box[:4]])], None, None, SIZE)


def black_rows(boxes):
    rows = set()
    for x, y, w, h, category in boxes:
        if category == 'unread':
            rows.update(range(y, y + h))
    return rows


def test_rows_read_by_a_later_frame_go_out_with_its_covers_and_no_black():
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    # 40 rows up: the strip's rows are now at 140 to 160, read, with an email at 150.
    later = [(10, 150, 60, 8, 'email')]
    incoming.record(1, later, [], np.ones((HEIGHT, WIDTH), bool), scroll(-40), SIZE)
    settled = incoming.settled(0)
    assert black_rows(settled) == set()
    assert (10, 190, 60, 8, 'email') in settled  # back where the email was in frame 0
    assert (50, 20, 40, 10, 'email') in settled  # what frame 0 found itself stays


def test_rows_still_unread_later_stay_black():
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    box, edge, inward = strip(140, HEIGHT)
    incoming.record(1, [box], [(box, edge, inward, [box[:4]])], np.ones((HEIGHT, WIDTH), bool), scroll(-20), SIZE)
    assert black_rows(incoming.settled(0)) == set(range(180, 200))


def test_rows_whose_pixels_the_scroll_does_not_explain_stay_black():
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    unexplained = np.zeros((HEIGHT - 40, WIDTH), bool)
    unexplained[145, 7] = True  # one pixel of row 185 of frame 0, now at 145
    step = scroll(-40, unexplained)
    incoming.record(1, [], [], np.ones((HEIGHT, WIDTH), bool), step, SIZE)
    assert black_rows(incoming.settled(0)) == {185}


def test_a_line_still_cut_by_the_edge_when_read_stays_black():
    """Reading after the scroll stops takes in the line cut by the edge. It isn't whole."""
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    incoming.record(1, [], [], np.ones((HEIGHT, WIDTH), bool), scroll(-4), SIZE)
    # Up 4, rows 180-193 are now 23 to 10 rows in from the edge at 200, and 194-199 fewer.
    assert black_rows(incoming.settled(0)) == set(range(194, 200))


def test_a_change_that_is_no_scroll_keeps_the_strip_black():
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    changed = nothing_changed()
    changed[190, 100] = True  # something redrawn in place, as a window moving over it
    incoming.record(1, [], [], changed, None, SIZE)
    assert 190 in black_rows(incoming.settled(0))


def test_rows_that_leave_the_view_unread_stay_black():
    incoming = IncomingRows(LINE)
    first_frame(incoming, top=100)
    incoming.record(1, [], [], np.ones((HEIGHT, WIDTH), bool), scroll(-150), SIZE)
    black = black_rows(incoming.settled(0))
    assert set(range(100, 150)) <= black  # above the top of the view now


def test_a_scroll_in_another_pane_does_not_move_the_strip():
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    other = ScrollStep(-40.0, 400, 500, 0, np.zeros((160, 100), bool), VIEW)
    changed = nothing_changed()
    changed[185, 10] = True
    incoming.record(1, [], [], changed, other, SIZE)
    assert 185 in black_rows(incoming.settled(0))


def test_settled_forgets_the_frame_and_those_before_it():
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    incoming.record(1, [], [], None, None, SIZE)
    assert incoming.settled(1) is not None
    assert incoming.settled(0) is None
    assert incoming.settled(1) is None


# The detector and compositor together, on the scripted scrolls the cover tests use.

from eclipse.compositor.compositor import PADDING, Compositor  # noqa: E402
from eclipse.text_detect import TextDetector  # noqa: E402
from eclipse.types import Frame  # noqa: E402
from eval.scroll_scene import BACKGROUND, SceneReader, ScrollScene  # noqa: E402

SCRIPTS = {
    "down": list(range(6, 240, 6)),
    "rapid": list(range(40, 800, 40)),
    "slow": list(range(2, 120, 2)),
    "stopping": [*range(6, 96, 6), *[90] * 8, *range(96, 150, 6), *[144] * 8],
    "reversal": [*range(6, 132, 6), *range(120, 0, -6), *range(6, 72, 6)],
}


def drawn(offsets, look_ahead):
    """Each frame as it goes out, drawn look_ahead frames after it was scanned, with the
    rows it kept black for being unread."""
    scene = ScrollScene()
    detect = TextDetector(reader=SceneReader(scene))
    if look_ahead:
        detect.look_ahead()
    compositor = Compositor()
    waiting, out = [], []
    for index, offset in enumerate([0, *offsets]):
        frame = Frame(scene.image(offset), index / 60, "screen", index)
        waiting.append((frame, offset, detect(frame)))
        while len(waiting) > look_ahead or (index == len(offsets) and waiting):
            frame, offset, found = waiting.pop(0)
            settled = detect.settled(frame) if look_ahead else None
            boxes = found if settled is None else settled
            output = compositor.process(frame, lambda _: boxes)
            unread = sum(d.w * d.h for d in boxes if d.category == "unread")
            out.append((scene, offset, output, unread))
    return out


def shown_secret_pixels(scene, offset, output):
    raw = scene.image(offset)
    total = 0
    for left, top, right, bottom in scene.secret_boxes(offset).values():
        text = (raw[top:bottom, left:right] != BACKGROUND).any(axis=2)
        same = (output[top:bottom, left:right] == raw[top:bottom, left:right]).all(axis=2)
        total += int((text & same).sum())
    return total


def test_looking_ahead_never_shows_a_secret_and_leaves_far_fewer_rows_black():
    """A slow scroll gains least: rows wait to be read until enough have come in, which
    takes longer than the look-ahead at 2 rows a frame."""
    for name, offsets in SCRIPTS.items():
        before = drawn(offsets, 0)
        after = drawn(offsets, 18)
        for scene, offset, output, _ in after:
            assert shown_secret_pixels(scene, offset, output) == 0, (name, offset)
        unread_before = sum(unread for *_, unread in before)
        unread_after = sum(unread for *_, unread in after)
        share = 0.6 if name == "slow" else 0.5
        assert unread_after < share * unread_before, (name, unread_after, unread_before)


def test_rows_with_uncovered_text_that_looks_like_a_secret_stay_black():
    """A read can get a secret wrong, so look-ahead won't clear rows on secret-looking
    text it found no secret in. Earlier frames keep them black, as before look-ahead."""
    incoming = IncomingRows(LINE)
    first_frame(incoming)
    doubtful = [(10, 150, 120, 8)]  # rows 190-198 of frame 0, now 40 rows up
    incoming.record(1, [], [], np.ones((HEIGHT, WIDTH), bool), scroll(-40), SIZE, doubtful)
    assert black_rows(incoming.settled(0)) == set(range(190, 198))


def test_black_rows_cover_only_what_the_strip_showed():
    incoming = IncomingRows(LINE)
    box, edge, inward = strip(180, HEIGHT)
    ink = [(30, 184, 100, 10), (200, 190, 50, 6)]  # two words; the rest was background
    incoming.record(0, [box], [(box, edge, inward, ink)], None, None, SIZE)
    incoming.record(1, [box], [(box, edge, inward, ink)], None, None, SIZE)  # still unread
    covers = incoming.settled(0)
    assert sorted(covers) == sorted([(*rect, 'unread') for rect in ink])


def test_rows_no_later_frame_clears_hold_the_pane_instead_of_going_black():
    incoming = IncomingRows(LINE)
    box, edge, inward = strip(180, HEIGHT)
    pane = (0, 0, WIDTH, HEIGHT)
    incoming.record(0, [box], [(box, edge, inward, [box[:4]])], None, None, SIZE, pane=pane)
    # Up 20, but still unread: the next frame's strip runs from row 160.
    later, edge1, inward1 = strip(160, HEIGHT)
    strips = [(later, edge1, inward1, [later[:4]])]
    everything = np.ones((HEIGHT, WIDTH), bool)
    incoming.record(1, [later], strips, everything, scroll(-20), SIZE, pane=pane)
    settled = incoming.settled(0)
    assert (0, 0, WIDTH, HEIGHT, 'held') in settled
    assert not [b for b in settled if b[4] == 'unread']
