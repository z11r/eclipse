"""Find sensitive text in images and return boxes to cover.

Images must be uint8 BGR arrays. Boxes are (x, y, width, height, category).
If detection fails, the pipeline must black out the frame."""

import bisect
import functools
import math
import re
import threading
import time
from typing import Iterable

import cv2
import numpy as np

from eclipse.text_detect.covers import MOVE, REREAD, STAY, CoverTracks, Sighting
from eclipse.text_detect.incoming import IncomingRows, ScrollStep
from eclipse.text_detect.labels import garbled_label
from eclipse.text_detect.rules import (
    UNREAD_MARK,
    ends_in_label,
    find_careful_spans,
    find_spans,
    names_a_secret,
    keyword_pattern,
    luhn_valid,
    standard_characters,
)

Box = tuple[int, int, int, int, str]
CATEGORIES = frozenset({'email', 'phone', '2fa', 'api_key', 'card', 'keyword'})

_BAND_MARGIN = 24  # rows re-read above and below a change
_SIDE_MARGIN = 60  # columns re-read either side of a change
_FULL_SCAN_SHARE = 0.6  # past this share of the screen, read the whole frame
_CHAR_MARGIN = 1.5  # characters of extra cover on each side of a secret
_SCROLL_MIN_HEIGHT = 120  # a changed area smaller than this isn't worth tracking
_SCROLL_MIN_WIDTH = 200
_SCROLL_MAX_SHIFT = 240  # most rows one frame can scroll
_SCROLL_TOLERANCE = 40  # blurred pixels this close count as the same text
_SCROLL_MAX_MISMATCH = 0.03  # past this share, it isn't a clean scroll
_SCROLL_EDGE = 16  # OCR boxes can sit this far outside the text they cover
_SCROLL_BLOCK = 8  # square of pixels judged together when finding which part scrolled
_SCROLL_SPLIT = 40  # change, in grey levels summed over a block's width, too small to judge
_SCROLL_JOIN_DOWN = 6  # blocks of blank lines that still join one pane's text
_SCROLL_JOIN_ACROSS = 2  # blocks of blank columns that still join; wider gaps part windows
_SCROLL_EXACT = 0.01  # share of a group's change a scroll may leave unexplained to join it
_SCROLL_LOOSE = 0.25  # past this share left unexplained, even the best group didn't scroll
_SCROLL_JOIN_BLOCKS = 16  # smaller groups can match a scroll by chance, so never join
_SCROLL_PART_WIDTH = 100  # narrowest pane, once it is told apart from what changed beside it
_NOISE_FLOOR = 8  # a pixel this close to where it came from hasn't changed
_CHANGE_BLOCK = 8  # with a sensor noise floor, compare blocks this size, not single pixels
_WHOLE_PIXEL = 0.05  # a shift this close to a whole number lands text on the same pixels
_SCROLL_RECHECK = 30  # frames of fractional scrolling between reads of the whole view
_UNREAD_READ_AT = 96  # rows of incoming text to collect before reading them
_UNREAD_KEEP = 24  # rows left black after a read, as the last line may be cut off
_CUT_REACH = 3  # a line cut by a pane edge is looked for this many ordinary lines deep
_RAIL_SHARE = 0.95  # share of rows a column must be inked in to be a border, not text
_LINE_GAP = 1  # blank rows, beyond this many, part one line from the next
_FOLLOW_BUDGET = 0.004  # seconds a frame may spend trying areas past its largest
_FOLLOW_STRICT = 0.004  # seconds before a frame stops trying motion last frame didn't have
_FOLLOW_RETRY = 4  # and every this many frames, everything is tried in full again
_SCROLL_RECHECK_REACH = 64  # rows past twice the first estimate the pane's own one looks
_SCROLL_SHRINK = 2  # finding the scrolling pane compares frames this many times smaller
_MOVE_REPEAT_REACH = 6  # pixels either way a carried-on drag's expected shift may be off
_MOVE_GAP_MAX = 8  # most dropped captures a drag's expected shift is stretched across
_MOVE_HALVE_FIRST = 1_000_000  # pixels past which a move guess halves before floats
_UNREAD_MIN_WAIT = 2  # frames between reads of rows that came in whole
_UNREAD_MIN_NEW = 8  # rows past the kept ones that a read after waiting needs
_INK_ROW_GAP = 4  # blank rows that still join one line's ink into one cover
_INK_WORD_GAP = 24  # blank columns that still join words; wider gaps part them
_INK_BACKGROUND = 0.5  # share of a strip one colour must fill to be its background
_INK_SAMPLES = 4096  # pixels looked at to find that colour
_STAYED_PUT_PIXELS = 40  # text pixels in uncovered rows needed to judge if they moved
_STAYED_PUT_SHARE = 0.98  # of them unchanged, for a drag over a still background
# With OCR workers reading every changed row ahead anyway, incoming rows are read on every
# frame past the kept ones, so the black at the incoming edge stays one steady height
# instead of growing and snapping back between reads.
_UNREAD_READ_AT_AHEAD = _UNREAD_KEEP + 1
_MOVE_MIN_HEIGHT = 60  # a changed area smaller than this isn't checked for a moved window
_MOVE_MIN_WIDTH = 100
_MOVE_COARSE = 4  # the shift is first estimated on frames this many times smaller
_MOVE_SAMPLES = 4000  # textured pixels that vote on the exact shift
_MOVE_SCREEN_SAMPLES = 500  # of those, how many first narrow the candidate shifts
_MOVE_FINALISTS = 4  # candidate shifts that then get the full vote
_MOVE_MIN_SHARE = 0.25  # share of changed textured pixels the shift must explain
_MOVE_CLIMB = 8  # most steps from the best candidate to a better neighbour
# Changes this close, sideways or up and down, count as one area. Lines of scrolling text
# sit closer than the up-and-down gap, so a pane mostly stays one area; windows side by
# side, like a scrolling note beside OBS's live preview, sit farther apart than the
# sideways one. Areas only steer the tries: a pane that falls apart is still followed.
_AREA_JOIN_X = 8
_AREA_JOIN_Y = 24
_AREA_BLOCK = 4  # areas are found on blocks this big, as they only steer where to look
# Small pieces this close to an area are tried with it: a dragged window's edges and
# shadow sit apart from its text, past blank margins, and must move with it.
_AREA_NEAR = 40
# Share of an area's changed pixels a follow must explain for the area to count as
# followed. Some pixels of a live preview always match by chance.
_AREA_EXPLAINED = 0.5
_AREA_FAILS = 3  # frames in a row an area's changes went unexplained before it's left out
_AREA_RETRY = 10  # frames between tries of an area left out, in case it starts to scroll
_AREA_FORGET = 120  # frames an area left out can go unseen before it's forgotten
_AREA_INSIDE = 0.8  # share of an area that must lie in a failing one to count as it
# and how big it must be beside that failing one: a window dragged off a live preview is
# a small part of the box the two filled together, and must be tried on its own again.
_AREA_SIMILAR = 0.5
_TEXTURE = 40  # a pixel this far from its neighbours' range is part of text or an edge
# How words on a line meet, in text heights. Closer than _TIGHT_GAP, OCR split one token,
# as Vision splits long hex into pieces. Up to _JOIN_GAP they are words of one line. Past
# that they part, unless a label ends the first and the pixels between show a mark OCR
# didn't read, up to _UNREAD_REACH. _MARK_EDGE is left out beside each word for its edges.
_TIGHT_GAP = 0.3
_JOIN_GAP = 2
_UNREAD_REACH = 4
_MARK_EDGE = 0.15
# Past this gap, in text heights, a line that names a secret is checked for punctuation
# OCR dropped, like the '", "' between JSON fields, so a value ends there.
_MARK_GAP = 0.6
# A line ending in a label, as "Password:" or "Commit SHA:", gives its meaning to the line
# right below: no more than this many text heights below, starting under or just past it.
_BELOW_GAP = 1.5  # Vision's boxes for small text are tight, so gaps look wide
_LABEL_WORDS = 3  # most words in a label line whose colon OCR misread
TIGHT, SPACE, UNREAD, NEWLINE = 'tight', 'space', 'unread', 'newline'
# Careful rereads of lines the fast read left in doubt, when a rescue reader is given:
# crops reread in one frame, seconds of rereading allowed per second, and the most saved
# up. A crop reaches this many text heights past its line, for a value OCR didn't see.
_RESCUE_PER_FRAME = 2
_RESCUE_BUDGET = 0.15
_RESCUE_BURST = 0.3
_RESCUE_REACH = 24
_RESCUE_CACHE = 256
_RESCUE_WHILE_MOVING = False  # careful rereads take 25-55 ms in the scan: while text moves they stall frames
_WAITING_MARGIN = 2  # standard pixels of cover around a value waiting for its careful reread
# A word that ends in a separator, or one OCR read with a stray mark after it: "abc:", "x=.".
_ENDS_IN_SEPARATOR = re.compile(r'[:=][.,;"\x27]{0,2}$')
_SEPARATORS = {TIGHT: ' ', SPACE: ' ', UNREAD: f' {UNREAD_MARK} ', NEWLINE: '\n'}
_TIGHT_SEPARATORS = {TIGHT: '', SPACE: ' ', UNREAD: f' {UNREAD_MARK} ', NEWLINE: '\n'}


class DetectionError(RuntimeError):
    """Detection failed. The pipeline must black out this frame."""


# The rules themselves live in rules.py. These names stay here for callers and tests.
_keyword_pattern = keyword_pattern
_standard_characters = standard_characters


def _match_text(text: str, keywords: Iterable[str] = ()) -> list[tuple[str, int, int]]:
    """Return (category, start, end) for each sensitive value in the text."""
    return [span[:3] for span in find_spans(text, tuple(keywords))]


def find_matches(text: str, keywords: Iterable[str] = ()) -> list[tuple[str, int, int]]:
    """Return (category, start, end) for each sensitive value in the text.

    Answers are remembered: see find_spans. Each span is the value, not its label.
    """
    return [span[:3] for span in find_spans(text, tuple(keywords))]


def _row_means(gray) -> np.ndarray:
    """Each row's mean, exactly as gray.mean(axis=1) gives it, in a tenth of the time."""
    # The sums are whole numbers well under 2**53, so float64 holds them exactly, and
    # dividing exact sums gives the same values numpy's mean does.
    return cv2.reduce(gray, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F)[:, 0] / gray.shape[1]


def _best_shift(differences: np.ndarray, limit: int) -> int:
    """The nonzero shift with the smallest difference, the most negative of any tie."""
    scores = np.array(differences, dtype=float)
    scores[limit] = np.inf  # no move at all isn't a shift
    return int(np.argmin(scores)) - limit


def _shift_differences(now_rows, before_rows, limit: int, block: int = 128) -> np.ndarray:
    """Mean absolute difference of two row profiles at every shift from -limit to limit.

    Entry i is shift i - limit, where now[j] lines up with before[j - shift]. One pass
    over all shifts in place of a Python call per shift: 1.2 ms rather than 1.9 ms.
    """
    height = len(now_rows)
    padded = np.zeros(height + 2 * limit)
    padded[limit : limit + height] = before_rows
    # Row r of these windows lines before up with now at shift limit - r.
    windows = np.lib.stride_tricks.sliding_window_view(padded, height)[::-1]
    totals = np.empty(2 * limit + 1)
    for first in range(0, 2 * limit + 1, block):
        totals[first : first + block] = np.abs(windows[first : first + block] - now_rows).sum(
            axis=1
        )
    # Padding compared now against 0, adding |now| for rows with nothing to compare.
    # Take those back out.
    ahead = np.concatenate(([0.0], np.cumsum(np.abs(now_rows))))
    shifts = np.arange(-limit, limit + 1)
    outside = np.where(
        shifts > 0,
        ahead[np.clip(shifts, 0, height)],
        ahead[height] - ahead[np.clip(height + shifts, 0, height)],
    )
    return (totals - outside) / np.maximum(height - np.abs(shifts), 1)


def classify_text(text: str, keywords: Iterable[str] = ()) -> set[str]:
    """Return the types of sensitive text found, without the secret text."""
    return {category for category, _, _ in find_matches(text, keywords)}


def _merge_bands(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []

    for top, bottom in sorted(bands):
        if merged and top <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], bottom))
        else:
            merged.append((top, bottom))

    return merged


def _overlaps(area, other) -> bool:
    """Whether two (top, bottom, left, right) areas overlap."""
    return (
        area[0] < other[1] and area[1] > other[0]
        and area[2] < other[3] and area[3] > other[2]
    )


def _join(area, other):
    return (
        min(area[0], other[0]), max(area[1], other[1]),
        min(area[2], other[2]), max(area[3], other[3]),
    )


def _inside(rect) -> tuple[slice, slice]:
    """The rows and columns of a (left, top, right, bottom, ...) box, for indexing a mask."""
    x1, y1, x2, y2 = rect[:4]
    return (
        slice(max(int(y1), 0), max(int(y2) + 1, 0)),
        slice(max(int(x1), 0), max(int(x2) + 1, 0)),
    )


def _part_of_a_pixel(now_rows, before_rows) -> float:
    """How much further than a whole pixel some aligned rows moved.

    Downscaled frames scroll by fractions of a pixel. Rounding every frame's shift
    would add up and slide covers off their text during a long scroll.
    """
    slope = np.gradient(before_rows)
    steepness = float((slope * slope).sum())

    if steepness == 0:
        return 0.0

    return float(np.clip((slope * (before_rows - now_rows)).sum() / steepness, -0.5, 0.5))


def _strongest(difference: np.ndarray) -> np.ndarray:
    """The largest change in any colour, so text that differs only in hue still counts."""
    blue, green, red = cv2.split(difference)
    return cv2.max(cv2.max(blue, green), red)


def _unexplained(
    now: np.ndarray, previous: np.ndarray, first_row: int, left: int, right: int, fraction: float
) -> np.ndarray:
    """Pixels the scroll doesn't account for.

    now is the new image's overlap with the old one, which starts at first_row of
    previous, between columns left and right.

    This is the check that clears pixels, so nothing here is softened or thinned. A
    scroll by whole pixels must reproduce each pixel, so any real difference counts.
    After a fractional scroll each pixel is a mix of neighbouring rows, so it only has
    to fall between them: new text still stands out, unless it is drawn over something
    of the same brightness, which the scroll recheck is there to catch.
    """
    height = now.shape[0]
    if abs(fraction) < _WHOLE_PIXEL:
        before = previous[first_row:first_row + height, left:right]
        return _strongest(cv2.absdiff(now, before)) > _NOISE_FLOOR
    # One row more on each side, as the rows at the edge mix with rows past it.
    start = max(first_row - 1, 0)
    around = previous[start:first_row + height + 1, left:right]
    rows = np.ones((3, 1), np.uint8)
    skip = first_row - start
    darkest = cv2.erode(around, rows)[skip:skip + height]
    brightest = cv2.dilate(around, rows)[skip:skip + height]
    too_dark = cv2.subtract(darkest, now)
    too_bright = cv2.subtract(now, brightest)
    return cv2.max(_strongest(too_dark), _strongest(too_bright)) > _NOISE_FLOOR


@functools.lru_cache(maxsize=64)
def _hanning(size) -> np.ndarray:
    """Phase correlation's window for a picture size, made once. Never written to."""
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    window.flags.writeable = False
    return window


def _changed_box(changed, box):
    """The smallest (top, bottom, left, right) holding every changed pixel, all of which
    lie within box, or None if nothing changed. Only box is looked at."""
    top, bottom, left, right = box
    inside = changed[top:bottom, left:right].view(np.uint8)
    rows = np.flatnonzero(cv2.reduce(inside, 1, cv2.REDUCE_MAX))
    if not rows.size:
        return None
    columns = np.flatnonzero(cv2.reduce(inside, 0, cv2.REDUCE_MAX))
    return (
        top + int(rows[0]), top + int(rows[-1]) + 1,
        left + int(columns[0]), left + int(columns[-1]) + 1,
    )


def _phase_guess(merged, size) -> tuple[int, int]:
    """The shift phase correlation finds between two blanked pictures, at full scale.

    merged holds the changed mask (0 to 255) and both pictures blanked by it, as floats.
    Shrinking averages, so each picture's mean over changed pixels comes from the small
    pictures, not another pass over the big ones. Floats matter: rounding to whole levels
    adds noise that phase correlation, which evens out every frequency, turns into
    another guess.
    """
    coverage, *blanked = cv2.split(cv2.resize(merged, size, interpolation=cv2.INTER_AREA))
    coverage /= 255
    total = float(coverage.sum())
    small = [
        shrunk - np.float32(float(shrunk.sum()) / total if total else 0.0) * coverage
        for shrunk in blanked
    ]
    (sx, sy), _ = cv2.phaseCorrelate(small[0], small[1], _hanning(size))
    return round(sx * _MOVE_COARSE), round(sy * _MOVE_COARSE)


def _texture(gray) -> np.ndarray:
    """Pixels on text or an edge: blank areas match any shift, so they can't vote."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel) > _TEXTURE


def _shifted_match(image, before, dx: int, dy: int, area) -> np.ndarray:
    """Within area (top, bottom, left, right), the pixels that show exactly what the
    previous frame showed dx, dy away. Pixels whose source is off the frame never match."""
    height, width = image.shape[:2]
    top, bottom, left, right = area
    match = np.zeros((bottom - top, right - left), bool)
    t, b = max(top, dy, 0), min(bottom, height + dy, height)
    l, r = max(left, dx, 0), min(right, width + dx, width)
    if b > t and r > l:
        difference = cv2.absdiff(image[t:b, l:r], before[t - dy : b - dy, l - dx : r - dx])
        # Every channel within the floor, as one OpenCV pass: numpy's max over the
        # channel axis took several milliseconds on a Retina screen.
        floor = (_NOISE_FLOOR,) * 3
        np.greater(
            cv2.inRange(difference, (0, 0, 0), floor), 0,
            out=match[t - top : b - top, l - left : r - left],
        )
    return match


def _ink(area: np.ndarray) -> np.ndarray | None:
    """Which pixels of an area aren't its background colour, or None if no one colour
    fills most of it, so there is no telling background from text."""
    if area.size == 0:
        return np.zeros(area.shape[:2], bool)
    # The background is most of the area, so a few thousand evenly spread pixels find it.
    step = max(int((area.shape[0] * area.shape[1] / _INK_SAMPLES) ** 0.5), 1)
    sample = area[::step, ::step].reshape(-1, 3).astype(np.int32)
    packed = (sample[:, 0] << 16) | (sample[:, 1] << 8) | sample[:, 2]
    colours, counts = np.unique(packed, return_counts=True)
    common = int(counts.argmax())
    if counts[common] < _INK_BACKGROUND * packed.size:
        return None
    value = int(colours[common])
    background = np.empty_like(area)
    background[:] = (value >> 16, (value >> 8) & 255, value & 255)
    return _strongest(cv2.absdiff(area, background)) > _NOISE_FLOOR


def _cut_line(
    image, edge: int, inward: int, left: int, right: int, reach: int, scale: int, fallback: int
) -> int:
    """How many rows in from a pane's edge the line it cuts through reaches, or 0 if
    the edge falls in a gap between lines. A line the edge cuts can't be read whole.

    Only blank rows prove the edge cuts nothing. Up to reach rows are looked at, and a
    line running past them counts as reach. If the background can't be told from text,
    the answer is fallback, the rows a line of ordinary text takes.
    """
    if inward > 0:
        area = image[edge:edge + reach, left:right]
    else:
        area = image[max(edge - reach, 0):edge, left:right][::-1]  # nearest the edge first
    ink = _ink(area)
    if ink is None:
        return fallback
    # Columns inked in nearly every row are rails, not text: a pane's border, the pane
    # beside it, a scroll bar. Text leaves gaps between lines. Counted as ink, a rail left
    # no blank row anywhere, so three lines at the edge stayed black, each line in turn.
    rails = ink.mean(axis=0) > _RAIL_SHARE
    if rails.any() and not rails.all():
        ink = ink[:, ~rails]
    rows = ink.any(axis=1)
    gap = _LINE_GAP * scale
    if not rows[:gap + 1].any():
        return 0
    inked = np.flatnonzero(rows)
    # The line touching the edge runs on to the first blank rows between lines.
    band = _runs(inked, gap)[0]
    return reach if band[1] >= len(rows) else min(band[1] + gap, reach)


def _ink_boxes(image, top, bottom, left, right, scale) -> list[tuple[int, int, int, int]]:
    """Boxes over everything in an area that isn't its background colour, as x, y, w, h.

    Pixels of the background colour can't show anything, so only the rest needs to be
    covered. If no one colour fills most of the area, there is no telling background
    from text, and the whole area comes back as one box.
    """
    area = image[top:bottom, left:right]
    if area.size == 0:
        return []
    ink = _ink(area)
    if ink is None:
        return [(left, top, right - left, bottom - top)]
    boxes = []
    for first, last in _runs(np.flatnonzero(ink.any(axis=1)), _INK_ROW_GAP * scale):
        columns = np.flatnonzero(ink[first:last].any(axis=0))
        for start, stop in _runs(columns, _INK_WORD_GAP * scale):
            boxes.append((left + start, top + first, stop - start, last - first))
    return boxes


def _runs(indices: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Sorted indices as (first, last + 1) runs, joined across gaps of up to gap."""
    if not indices.size:
        return []
    breaks = np.flatnonzero(np.diff(indices) > gap + 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]])) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


@functools.lru_cache(maxsize=4096)
def _looks_secret(text: str) -> bool:
    """Text a misread secret could hide in: an @, five or more digits, or a long
    token mixing digits or cases. Look-ahead won't clear rows on such text unless a
    cover is on it, as the read may have got a real secret wrong."""
    if '@' in text or sum(character.isdigit() for character in text) >= 5:
        return True
    for token in text.split():
        if len(token) >= 16 and (
            sum(character.isdigit() for character in token) >= 2
            or (any(c.islower() for c in token) and any(c.isupper() for c in token)
                and not token.isalpha())
        ):
            return True
    return False


def _shift_shares(image, before, ys, xs, now, shifts) -> np.ndarray:
    """For each dx, dy in shifts, the share of the sampled pixels ys, xs that show exactly
    what the previous frame showed dx, dy away. now holds image[ys, xs] as int16. Pixels
    whose source is off the frame don't match."""
    height, width = image.shape[:2]
    shifts = np.asarray(shifts, dtype=np.intp).reshape(-1, 2)
    from_y = ys[None, :] - shifts[:, 1:2]
    from_x = xs[None, :] - shifts[:, 0:1]
    inside = (from_y >= 0) & (from_y < height) & (from_x >= 0) & (from_x < width)
    source = before[np.clip(from_y, 0, height - 1), np.clip(from_x, 0, width - 1)]
    same = (np.abs(source.astype(np.int16) - now[None]).max(axis=2) <= _NOISE_FLOOR) & inside
    return same.sum(axis=1) / ys.size


def _outside(result, part, changed):
    """What still needs reading after following part of changed: what following left,
    plus everything changed outside part. A follow that found nothing hands part back
    untouched, and then that's all of changed, without whole-frame arithmetic."""
    if result is part:
        return changed
    if part is changed:
        return result
    return result | (changed & ~part)


def _marked(image, left, right, height) -> bool:
    """Whether the pixels between two words on a line show a mark OCR didn't read.

    The edges next to each word are left out, as glyphs' soft edges spill past their boxes.
    With no telling background from text, it may hold a mark, so the answer is yes.
    """
    if image is None:
        return False
    edge = _MARK_EDGE * height
    x1, x2 = int(math.ceil(left[2] + edge)), int(math.floor(right[0] - edge))
    y1, y2 = int(max(min(left[1], right[1]), 0)), int(math.ceil(max(left[3], right[3])))
    if x2 <= x1 or y2 <= y1:
        return False
    ink = _ink(image[y1:y2, x1:x2])
    return True if ink is None else bool(ink.any())


def _in_doubt(text: str, spans) -> bool:
    """Whether a line's fast read leaves it unsettled: a value whose check failed, or a
    secret named with no value found for it, which a misread may have hidden."""
    if any(span.uncertain for span in spans):
        return True
    return names_a_secret(text) and not any(
        span.category in ('api_key', 'card') for span in spans
    )


def _labelled_from_above(lines) -> dict[int, int]:
    """For each line right under a line that ends in a label, that label line's index."""
    starts = sorted((parts[0][1], index) for index, (parts, _) in enumerate(lines))
    tops = [top for top, _ in starts]
    found = {}
    for index, (parts, _) in enumerate(lines):
        last = parts[-1][4].rstrip()
        # A short line ending in a colon, which OCR can read as a quote or semicolon.
        short = len(parts) <= _LABEL_WORDS and not any('"' in p[4] for p in parts[:-1])
        if not (
            last.endswith((':', '..')) or ends_in_label(last)
            or short and (
                last[-1:] in ('"', ';') and last.count('"') == 1
                or last.endswith(('".', "'."))
            )
        ):
            continue
        top, bottom = min(p[1] for p in parts), max(p[3] for p in parts)
        left, right = parts[0][0], parts[-1][2]
        tall = bottom - top
        first = bisect.bisect_left(tops, top + 0.5 * tall)
        last_row = bisect.bisect_right(tops, bottom + _BELOW_GAP * tall)
        # The nearest line below that starts under the label or just past its end.
        for _, other in starts[first:last_row]:
            start = lines[other][0][0][0]
            if other != index and other not in found and left - tall <= start <= right + tall:
                found[other] = index
                break
    return found


_has_word = functools.lru_cache(maxsize=16384)(lambda text: any(c.isalnum() for c in text))


def _alternatives(one, other) -> bool:
    """Whether two text boxes on a row are mostly the same place, so two readings of it."""
    overlap = min(one[2], other[2]) - max(one[0], other[0])
    return overlap > 0.5 * min(one[2] - one[0], other[2] - other[0])


def _text_hash(text: str) -> int:
    """Tells text apart across reads, so tracks never have to keep the text itself."""
    return hash(''.join(text.split()).lower())


class Detector:
    """Read text from images and find sensitive information.

    Use one detector per screen or camera. Only rows that changed are read
    again, and text read earlier is kept until its pixels change. A secret's cover
    lives in CoverTracks, so one read missing it doesn't uncover it. Each result is
    everything to cover in that frame. Changing settings clears saved results."""

    def __init__(
        self, keywords=(), enabled=None, padding=0, reader=None, pixel_scale=1, noise_floor=0,
        ocr_threads=6, rescue=None, rescue_budget=_RESCUE_BUDGET,
    ):
        self.stats = {
            'frames': 0, 'full_scans': 0, 'partial_scans': 0, 'scan_seconds': 0.0,
            'scanned_share': 0.0, 'diff_seconds': 0.0, 'find_seconds': 0.0,
            'det_seconds': 0.0, 'rec_seconds': 0.0, 'reads': 0, 'lines': 0, 'scrolls': 0,
            'checks': 0, 'copy_seconds': 0.0, 'track_seconds': 0.0, 'cached_frames': 0,
            'moves': 0, 'areas_left_out': 0, 'areas_seconds': 0.0, 'scroll_seconds': 0.0,
            'narrow_seconds': 0.0, 'move_seconds': 0.0, 'ahead_seconds': 0.0, 'follow_cut': 0,
            'estimate_seconds': 0.0, 'columns_seconds': 0.0, 'pixels_seconds': 0.0,
            'grow_seconds': 0.0, 'held_lost': 0, 'held_unread': 0, 'held_cut': 0,
            'held_doubtful': 0, 'follow_skipped': 0, 'lost_view': 0, 'lost_unexplained': 0,
            'lost_edge': 0, 'lost_still': 0, 'moves_repeated': 0,
            'repeat_tries': 0, 'repeat_crowded': 0, 'repeat_upright': 0, 'repeat_missed': 0,
            'miss_inside': 0, 'miss_near': 0, 'miss_far': 0, 'miss_lost': 0, 'drag_searched': 0,
            'repeat_seconds': 0.0,
            'uncertain': 0, 'doubtful_lines': 0, 'rescues': 0, 'rescue_cached': 0,
            'rescue_skipped': 0, 'rescue_failed': 0, 'rescue_seconds': 0.0,
            'rescue_deferred': 0,
        }
        if not isinstance(padding, int) or padding < 0:
            raise ValueError('padding must be a nonnegative integer')

        if not isinstance(noise_floor, int) or noise_floor < 0:
            raise ValueError('noise_floor must be a nonnegative integer')

        self.padding = padding
        self._reader = reader
        # A slower, more careful reader for lines the fast read left in doubt, or None.
        self._rescue = rescue
        self._rescue_budget = rescue_budget
        self._rescue_allowance = _RESCUE_BURST
        self._rescue_clock = None
        self._rescued = {}  # crop fingerprint -> words read there, newest last
        self._careful_ids = set()  # ids of words a careful reread gave, still on screen
        self._careful_areas = set()
        self._rescue_waiting = set()  # ids of doubtful words whose reread had no room yet
        self._was_full = False  # whether the last scan read the whole screen
        self._ocr_threads = ocr_threads  # for the RapidOCR reader made when none is given
        # Cameras redraw every pixel with sensor noise, so any-pixel-changed would read
        # the whole frame every time. Above 0, blocks whose average moved less than
        # this count as unchanged. Screens are exact, so they keep 0.
        self._noise_floor = noise_floor
        # Distances below are for standard frames. Retina frames have twice the pixels.
        self._band_margin = _BAND_MARGIN * pixel_scale
        self._side_margin = _SIDE_MARGIN * pixel_scale
        self._scroll_min_height = _SCROLL_MIN_HEIGHT * pixel_scale
        self._scroll_min_width = _SCROLL_MIN_WIDTH * pixel_scale
        self._scroll_max_shift = _SCROLL_MAX_SHIFT * pixel_scale
        self._scroll_edge = _SCROLL_EDGE * pixel_scale
        reads_ahead = hasattr(reader, 'prepare')  # see read_ahead.py
        self._unread_read_at = (
            _UNREAD_READ_AT_AHEAD if reads_ahead else _UNREAD_READ_AT
        ) * pixel_scale
        self._unread_keep = _UNREAD_KEEP * pixel_scale
        self._pixel_scale = pixel_scale
        self._scroll_block = _SCROLL_BLOCK * pixel_scale
        self._scroll_part_width = _SCROLL_PART_WIDTH * pixel_scale
        self._move_min_height = _MOVE_MIN_HEIGHT * pixel_scale
        self._move_min_width = _MOVE_MIN_WIDTH * pixel_scale
        self._area_join = (_AREA_JOIN_X * pixel_scale, _AREA_JOIN_Y * pixel_scale)
        self._area_block = _AREA_BLOCK * pixel_scale
        self._area_near = _AREA_NEAR * pixel_scale
        self._tracks = CoverTracks(pixel_scale)
        self._incoming = IncomingRows(_UNREAD_KEEP * pixel_scale)
        self._looking_ahead = False  # following strips costs time, so only when asked
        self._shape = None
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self.reused_read = False  # whether the last frame needed no OCR
        self._keywords = ()
        self._enabled = frozenset()
        self.configure(keywords, enabled)

    def configure(self, keywords=(), enabled=None):
        """Change what counts as sensitive, all at once. A bad setting changes nothing.

        Covers in categories that stay on are kept. Dropping them would leave a known
        secret relying on one new read to be found again.
        """
        words = tuple(keywords)
        categories = CATEGORIES if enabled is None else frozenset(enabled)

        if not all(isinstance(word, str) for word in words):
            raise ValueError('keywords must contain strings')

        if not categories <= CATEGORIES:
            raise ValueError('unknown detection category')

        for word in words:
            _keyword_pattern(word)

        with self._lock:
            still_on = categories & self._enabled
            if not set(self._keywords) <= set(words):
                still_on -= {'keyword'}  # a cover can't say which keyword it was for
            self._keywords = words
            self._enabled = categories
            self._forget_scan()
            self._tracks.keep_only(still_on)

    def cancel(self):
        """Stop for good at the next chance, for shutdown. Later scans fail, so stay safe.

        A read already inside the OCR engine can't be stopped. Takes no lock, because
        detect() holds it for the whole scan.
        """
        self._cancelled.set()

    def reset(self):
        """Forget everything, covers included, because old positions no longer apply.

        For a new source or new settings. The pipeline must clear its buffer too."""
        with self._lock:
            self._forget_scan()
            self._tracks.clear()

    def _forget_scan(self):
        """Forget what was read, so the next frame is read whole. Covers stay."""
        self._previous = None
        self._regions = []
        self._boxes = ()
        self._unread = {}  # one strip for each way rows can come in: 1 is down, -1 is up
        self._unread_waited = {}  # frames each strip has waited since its last read
        self._strip_ink = {}  # each strip's covers, over its text only
        self._uncovered = None  # rows a scroll uncovered this frame, to tell a drag apart
        self._probe = None  # the rows the last frame's scroll uncovered
        self._pane = None  # the last scrolling pane's edges, which cut lines off where it stops
        self._incoming.clear()  # strips can only be followed through frames read in order
        self._last_changed = None  # this frame's changed pixels, while strips are followed
        self._step = None  # this frame's checked scroll, while strips are followed
        self._scrolled = False
        self._moved = False  # whether known text moved with a window this frame
        self._last_move = None  # the last window move, which the next one likely repeats
        self._drag_before = None  # the move the frame before that, while a drag goes on
        # Captures between the frame compared against and this one: more than one when
        # frames were dropped. And the captures the last two moves each spanned.
        self._previous_key = None
        self._gap = self._move_gap = self._before_gap = 1
        self._drag_area = None  # top, bottom, left, right of what the last move carried
        self._view = None  # top and bottom of the area scrolling now
        self._fractional_frames = 0  # frames of this scroll that landed between pixels
        # Areas whose changes keep going unexplained, like a live video or OBS's preview:
        # [top, bottom, left, right, fails in a row, frame last tried, frame last seen].
        self._failing = []
        self._follow_frame = 0
        self._last_followed = None  # 'scroll' or 'move', what last frame followed
        self._follow_deadline = None  # past this, a frame skips its costly follow tries

    def _scroll_action(self, rect, image, changed, top, bottom, left, right) -> str:
        """What a scroll of the given area does to a piece of text or a cover."""
        x1, y1, x2, y2 = rect[:4]

        if not (x1 < right and x2 > left and y1 < bottom and y2 > top):
            return STAY

        if not changed[_inside(rect)].any():
            pixels = image[_inside(rect)]
            if pixels.size and int(pixels.max()) - int(pixels.min()) > _SCROLL_TOLERANCE:
                # Text that didn't change hasn't moved: it's still text beside the scroll
                # that the area's bounds took in. Blank pixels prove nothing either way.
                return STAY

        if x1 >= left - self._scroll_edge and x2 <= right + self._scroll_edge:
            return MOVE

        # Text cut by the edge of the scrolling area can't be trusted.
        return REREAD

    def _stayed_put(self, image, uncovered, shift) -> bool:
        """Whether rows a scroll uncovered last frame are unchanged now, while the area
        moved again, rather than moved along with it as scrolling text does. Only rows
        showing something can tell: blank ones look the same either way, so they don't."""
        top, bottom, left, right, before = uncovered
        grey = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        textured = _texture(grey)
        if textured.sum() < _STAYED_PUT_PIXELS * self._pixel_scale:
            return False
        still = _strongest(cv2.absdiff(image[top:bottom, left:right], before)) <= _NOISE_FLOOR
        if still[textured].mean() < _STAYED_PUT_SHARE:
            return False
        height = image.shape[0]
        a, b = max(top + shift, 0), min(bottom + shift, height)
        if b - a < bottom - top:
            return True  # it would have moved off the frame: it stayed
        moved = _strongest(cv2.absdiff(image[a:b, left:right], before)) <= _NOISE_FLOOR
        # Scrolled text matches its shifted self almost exactly; a busy background can
        # match a shifted copy of itself often, by chance, but not that often.
        return moved[textured].mean() < _STAYED_PUT_SHARE - 0.08

    def _scroll_shift(self, image, top, bottom, left, right, greys=None, reach=None):
        """The whole-row shift that best explains how the area changed, with the grey
        pictures it was found on, or None when no shift explains it better than none.
        greys, if given, are those pictures already made for this area. reach, if given,
        is the farthest shift to try."""
        if greys is None:
            now = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            before = cv2.cvtColor(self._previous[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        else:
            now, before = greys
        now_rows = _row_means(now)
        before_rows = _row_means(before)
        limit = min(self._scroll_max_shift, (bottom - top) // 2)
        if reach is not None:
            limit = max(min(limit, reach), 1)
        differences = _shift_differences(now_rows, before_rows, limit)
        shift = _best_shift(differences, limit)
        if differences[shift + limit] >= differences[limit]:
            return None
        return shift, now, before, now_rows, before_rows

    def _scrolling_part(self, changed, estimate, top, bottom, left, right):
        """The part of a changed area that scrolled: (top, bottom, left, right), or None.

        A block of pixels scrolled when the shift explains its change far better than no
        move does, or exactly. The pane is the group of such blocks that the shift explains
        most, joined across blank lines but not across a wide gap to the side, where other
        windows sit. Only rows on screen in both frames are compared, so the rows coming
        in are added back at the pane's incoming edge.
        """
        shift, now, before = estimate[:3]
        length = (bottom - top) - abs(shift)
        now_start, before_start = max(shift, 0), max(-shift, 0)  # within the area
        block = self._scroll_block
        # Compared shrunk: each frame is shrunk after it's lined up by the shift, so a
        # whole-row scroll still matches, and averaging pixels softens the redrawn text
        # edges of a scroll that lands between pixels. This only finds the pane; every
        # pixel that moves is still checked at full size by the scroll that follows.
        factor = _SCROLL_SHRINK * self._pixel_scale
        length -= length % factor
        width = (right - left) // factor * factor
        small = width // factor, length // factor
        down, across = -(-length // block), -(-width // block)

        def shrink(picture):
            return cv2.resize(picture, small, interpolation=cv2.INTER_AREA)

        current = shrink(now[now_start:now_start + length, :width])
        rows = slice(top + now_start, top + now_start + length)
        mask = shrink(changed[rows, left:left + width].view(np.uint8) * 255)

        def unexplained(source_row):
            """Per block, the summed grey difference where pixels changed."""
            source = shrink(before[source_row:source_row + length, :width])
            difference = cv2.absdiff(current, source)
            difference = cv2.bitwise_and(difference, difference, mask=mask)
            padded = cv2.copyMakeBorder(
                difference, 0, down * block // factor - small[1], 0,
                across * block // factor - small[0],
                cv2.BORDER_CONSTANT, value=0,
            ).astype(np.float32)
            means = cv2.resize(padded, (across, down), interpolation=cv2.INTER_AREA)
            return np.rint(means * (block * block))

        missed = unexplained(before_start)  # what the shift leaves unexplained
        static = unexplained(now_start)  # what no move at all leaves unexplained
        scrolled = (static > 0) & (
            (missed == 0) | ((static > _SCROLL_SPLIT * block) & (missed * 2 < static))
        )
        if not scrolled.any():
            return None
        joined = cv2.dilate(
            scrolled.astype(np.uint8),
            np.ones((2 * _SCROLL_JOIN_DOWN + 1, 2 * _SCROLL_JOIN_ACROSS + 1), np.uint8),
        )
        count, labels = cv2.connectedComponents(joined, connectivity=8)
        groups = []
        for label in range(1, count):
            part = scrolled & (labels == label)
            spanned = np.flatnonzero(part.any(axis=1))
            groups.append((
                float((static - missed)[part].sum()), float(missed[part].sum()),
                float(static[part].sum()), int(spanned[0]), int(spanned[-1]), part,
            ))
        best = max(groups, key=lambda group: group[0])
        if best[1] > _SCROLL_LOOSE * best[2]:
            return None  # the best match is a poor one, like a mirror of the screen
        # Other columns of the same page, like a sidebar or a table's, move exactly with
        # it. Things that only look a little like the scroll, like a preview of the screen
        # or a meter, are left to be read as they are.
        pane = np.zeros_like(scrolled)
        for _, left_over, change, first_block, last_block, part in groups:
            if part is best[5] or (
                left_over <= _SCROLL_EXACT * change
                and part.sum() >= _SCROLL_JOIN_BLOCKS
                and first_block <= best[4] and last_block >= best[3]
            ):
                pane |= part
        hits = np.argwhere(pane)
        first = left + int(hits[:, 1].min()) * block
        last = min(left + (int(hits[:, 1].max()) + 1) * block, right)
        # The pane's rows: every changed row in its columns. A line coming in after a wide
        # blank gap is still the pane's, and must be held as incoming, not read while cut.
        # What changes beside the pane is in other columns, so it stays out.
        first_row, last_row = top, bottom
        inside = changed[first_row:last_row, first:last]
        found_rows = np.flatnonzero(inside.any(axis=1))
        found_columns = np.flatnonzero(inside.any(axis=0))
        return (
            first_row + int(found_rows[0]), first_row + int(found_rows[-1]) + 1,
            first + int(found_columns[0]), first + int(found_columns[-1]) + 1,
        )

    def _find_scroll(self, image, changed, area):
        """The area that scrolled and its shift estimate, or None if nothing did.

        Other things change in the same frame: a video, a meter, a screen preview that
        mirrors the scroll. Their pixels must not stretch the area that scrolls, or the
        rows held black as they come in land on them instead of on the pane's edge. When
        they drown the scroll out altogether, _follow_areas and _follow_scroll try the
        separate areas and the columns that scrolled alone, so this tries only the one.
        """
        top, bottom, left, right = area
        if bottom - top < self._scroll_min_height or right - left < self._scroll_part_width:
            return None
        estimate = self._scroll_shift(image, top, bottom, left, right)
        if estimate is None:
            return None
        narrow_started = time.monotonic()
        try:
            return self._narrowed(image, changed, estimate, area)
        finally:
            self.stats['narrow_seconds'] += time.monotonic() - narrow_started

    def _narrowed(self, image, changed, estimate, area):
        """The part of area that scrolled, with the shift found on it alone, or None."""
        top, bottom, left, right = area
        part = self._scrolling_part(changed, estimate, top, bottom, left, right)
        if part is None:
            return None
        if part == area:
            return part, estimate
        if (
            part[1] - part[0] < self._scroll_min_height
            or part[3] - part[2] < self._scroll_part_width
        ):
            return None
        # The shift again, on the pane alone: what changed beside it can pull the first
        # estimate off, and covers moved by a wrong shift would leave their text.
        inside = slice(part[0] - top, part[1] - top), slice(part[2] - left, part[3] - left)
        # Far enough to catch a first estimate thrown well off, as two windows can: the
        # pane's true shift is rarely more than the wrong one plus a few lines.
        estimate = self._scroll_shift(
            image, *part, greys=(estimate[1][inside], estimate[2][inside]),
            reach=2 * abs(estimate[0]) + _SCROLL_RECHECK_REACH * self._pixel_scale,
        )
        return None if estimate is None else (part, estimate)

    def _follow_scroll(self, image, changed):
        """Follow a scroll, even with other things changing elsewhere on the screen.

        The scroll is first looked for across everything that changed. Other changes, such
        as a live OBS preview beside the scrolling window, make that whole area fail the
        clean-scroll test, so then the columns that did scroll are found and tried alone.
        Everything outside them still counts as changed.
        """
        remaining = self._follow_scroll_in(image, changed)
        if self._scrolled:
            return remaining
        columns = self._scroll_columns(image, changed)
        if columns is None:
            return changed
        left, right = columns
        part = np.zeros_like(changed)
        part[:, left:right] = changed[:, left:right]
        remaining = self._follow_scroll_in(image, part)
        if not self._scrolled:
            return changed
        remaining[:, :left] |= changed[:, :left]
        remaining[:, right:] |= changed[:, right:]
        return remaining

    def _in_time(self, kind: str) -> bool:
        """Whether this frame may still try to follow this kind of motion, scroll or move:
        always the kind last frame followed, the other only within the budget."""
        if (
            self._follow_deadline is None
            or kind == self._last_followed
            or time.monotonic() <= self._follow_deadline
        ):
            return True
        self.stats['follow_skipped'] += 1
        return False

    def _scroll_columns(self, image, changed):
        if not self._in_time('scroll'):
            return None
        started = time.monotonic()
        try:
            return self._scroll_columns_timed(image, changed)
        finally:
            self.stats['columns_seconds'] += time.monotonic() - started

    def _scroll_columns_timed(self, image, changed):
        """Left and right of the widest run of columns that scrolled vertically, or None.

        Proposes only: the scroll found there must still pass every check of a clean
        scroll before anything moves.
        """
        rows = np.flatnonzero(changed.any(axis=1))
        columns = np.flatnonzero(changed.any(axis=0))
        if not rows.size:
            return None
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(columns[0]), int(columns[-1]) + 1
        if bottom - top < self._scroll_min_height or right - left < self._scroll_min_width:
            return None
        now = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        before = cv2.cvtColor(self._previous[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        mask = changed[top:bottom, left:right]
        # Each stripe of columns proposes its own shift from its row profiles, as a whole
        # scroll is found, so a pane that scrolled can win its stripes even while something
        # beside it moves another way.
        guesses = {}
        stripe = self._scroll_min_width * 2
        height = bottom - top
        limit = min(self._scroll_max_shift, height // 2)
        for start in range(0, right - left, stripe):
            part = slice(start, min(start + stripe, right - left))
            if part.stop - part.start < self._scroll_min_width // 2 or not mask[:, part].any():
                continue
            differences = _shift_differences(
                _row_means(now[:, part]), _row_means(before[:, part]), limit
            )
            shift = _best_shift(differences, limit)
            if differences[shift + limit] < differences[limit]:
                guesses[shift] = guesses.get(shift, 0) + part.stop - part.start
        if not guesses:
            return None  # nothing moved vertically

        textured = _texture(now) & mask
        now_soft, before_soft = cv2.blur(now, (3, 3)), cv2.blur(before, (3, 3))
        best = None
        guess = max(guesses, key=guesses.get)
        for shift in range(guess - 1, guess + 2):  # the profile shift can be off by a fraction
            if shift == 0 or abs(shift) >= height:
                continue
            start, stop = max(shift, 0), height + min(shift, 0)
            moved = textured[start:stop]
            same = cv2.absdiff(now_soft[start:stop], before_soft[start - shift:stop - shift])
            same = (same <= _SCROLL_TOLERANCE) & moved
            score = int(same.sum())
            if best is None or score > best[0]:
                best = (score, moved.sum(axis=0), same.sum(axis=0))
        if best is None or not best[0]:
            return None
        _, moved, same = best

        block = self._scroll_edge * 2
        good = []  # per block: True scrolled, False didn't, None nothing to tell by
        for start in range(0, right - left, block):
            have = int(moved[start:start + block].sum())
            if have < block // 2:
                good.append(None)
            else:
                good.append(bool(same[start:start + block].sum() >= (1 - _SCROLL_MAX_MISMATCH * 3) * have))
        runs, run = [], None
        for index, value in enumerate(good + [False]):
            if value is False:
                if run is not None:
                    runs.append(run)
                run = None
            elif value is True:
                run = (run[0] if run is not None else index, index + 1)
        if not runs:
            return None
        first, last = max(runs, key=lambda r: r[1] - r[0])
        run_left, run_right = left + first * block, min(left + last * block, right)
        if run_right - run_left < self._scroll_min_width or (run_left, run_right) == (left, right):
            return None
        return run_left, run_right

    def _follow_scroll_in(self, image, changed):
        started = time.monotonic()
        try:
            return self._follow_scroll_in_timed(image, changed)
        finally:
            self.stats['scroll_seconds'] += time.monotonic() - started

    def _follow_scroll_in_timed(self, image, changed):
        """If the change is a clean vertical scroll, move known text along with it.

        The tolerant comparisons here only propose a shift, and never clear a pixel.
        Returns what still needs reading: the rows newly on screen, plus every pixel
        the shift doesn't explain, however thin or faint. If this isn't a clean
        scroll, returns the change untouched, so a wrong guess only costs time.
        """
        rows = np.flatnonzero(changed.any(axis=1))
        columns = np.flatnonzero(changed.any(axis=0))

        if not rows.size:
            return changed

        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(columns[0]), int(columns[-1]) + 1
        height = bottom - top

        if height < self._scroll_min_height or right - left < self._scroll_min_width:
            return changed

        # Taken once a frame, as this can be tried on several areas of one frame.
        uncovered = self._probe
        found = self._find_scroll(image, changed, (top, bottom, left, right))
        if found is None:
            return changed
        (top, bottom, left, right), estimate = found
        height = bottom - top
        shift, now, before, now_rows, before_rows = estimate
        if uncovered is not None and self._stayed_put(image, uncovered, shift):
            # What last frame's move uncovered didn't move with this one: a window dragged
            # up or down over a still background, not a pane scrolling text in at its edge.
            # Nothing scrolls in, so nothing is held black; the window move follows it,
            # and what it uncovers is read where it stays.
            if shift > 0:
                rows = top, min(top + shift, bottom)
            else:
                rows = max(bottom + shift, top), bottom
            self._uncovered = (*rows, left, right, image[rows[0]:rows[1], left:right].copy())
            return changed

        length = height - abs(shift)
        now_start, before_start = max(shift, 0), max(-shift, 0)
        # Half-pixel moves redraw text edges, so compare softened images.
        mismatch = cv2.absdiff(  # |a - b| of uint8 fits uint8, so this is exact
            cv2.blur(now[now_start:now_start + length], (3, 3)),
            cv2.blur(before[before_start:before_start + length], (3, 3)),
        ) > _SCROLL_TOLERANCE

        if mismatch.mean() > _SCROLL_MAX_MISMATCH:
            return changed

        fraction = _part_of_a_pixel(
            now_rows[now_start:now_start + length], before_rows[before_start:before_start + length]
        )
        first_row, old_row = top + now_start, top + before_start
        if abs(fraction) >= _WHOLE_PIXEL:
            # The row at the incoming edge is partly made of a row not seen yet, so it
            # counts as incoming and stays covered until it can be read.
            length -= 1
            if shift > 0:
                first_row, old_row = first_row + 1, old_row + 1
        # Changes beside the scrolling area are read as they are.
        remaining = changed.copy()
        remaining[top:bottom, left:right] = False
        # A pixel that didn't change at all stays cleared, even if it doesn't match the
        # shift: that is what still content beside a moving pane looks like.
        remaining[first_row:first_row + length, left:right] = changed[
            first_row:first_row + length, left:right
        ] & _unexplained(
            image[first_row:first_row + length, left:right],
            self._previous,
            old_row,
            left,
            right,
            fraction,
        )
        moved = []
        # Rows that look the same frame to frame don't count as changed, so this frame's
        # changed rows can fall short of the area's real edges. Remember the widest yet.
        if self._view is not None:
            top_of_view, bottom_of_view = min(top, self._view[0]), max(bottom, self._view[1])
        else:
            top_of_view, bottom_of_view = top, bottom
        self._view = top_of_view, bottom_of_view
        self._pane = top_of_view, bottom_of_view, left, right
        exact_shift = shift + fraction
        if self._incoming.following:
            self._step = self._checked_step(image, shift, fraction, left, right)

        if abs(fraction) >= _WHOLE_PIXEL:
            self._fractional_frames += 1
            if self._fractional_frames % _SCROLL_RECHECK == 0:
                remaining[top_of_view:bottom_of_view, left:right] = True

        def action(rect) -> str:
            return self._scroll_action(
                rect, image, changed, top_of_view, bottom_of_view, left, right
            )

        for region in self._regions:
            todo = action(region)
            if todo == STAY:
                moved.append(region)
            elif todo == REREAD:
                remaining[_inside(region)] = True
            else:
                x1, y1, x2, y2, text = region
                new_top = max(y1 + exact_shift, top_of_view)
                new_bottom = min(y2 + exact_shift, bottom_of_view)
                if new_bottom > new_top:
                    moved.append(self._carry(region, (x1, new_top, x2, new_bottom, text)))

        for place in self._tracks.follow_scroll(
            action, exact_shift, top_of_view, bottom_of_view, image
        ):
            remaining[_inside(place)] = True

        self._regions = moved
        self.stats['scrolls'] += 1
        self._scrolled = True
        self._collect_unread(image, shift, left, right, first_row, length, remaining)
        incoming = (top_of_view, first_row) if shift > 0 else (
            first_row + length, bottom_of_view
        )
        if incoming[1] > incoming[0]:
            self._uncovered = (
                *incoming, left, right, image[incoming[0]:incoming[1], left:right].copy()
            )
        return remaining

    def _move_shift(self, image, changed, area):
        """The dx, dy that a window moving whole explains best, or None.

        Unchanged pixels are blanked before comparing, so still content around the move
        can't vote for no move at all. The estimate only proposes: the shift has to make
        nearly all sampled text pixels match exactly before anything is moved by it.
        """
        top, bottom, left, right = area
        now = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        mask = changed[top:bottom, left:right]
        mask8 = mask.view(np.uint8)
        size = (max((right - left) // _MOVE_COARSE, 8), max((bottom - top) // _MOVE_COARSE, 8))
        guess = self._coarse_guess(now, mask8, area, size)
        return self._checked_move(image, now, mask8, area, guess)

    def _coarse_guess(self, now, mask8, area, size) -> tuple[int, int]:
        """The shift phase correlation proposes for an area, at full detail."""
        top, bottom, left, right = area
        before = cv2.cvtColor(self._previous[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        # Blank unchanged pixels, so still content around the move can't vote for no move
        # at all. Both pictures and the mask then shrink in one three-channel call, about
        # three times quicker than one call each.
        layers = [mask8 * np.uint8(255)]
        for gray in (before, now):
            layers.append(cv2.bitwise_and(gray, gray, mask=mask8))
        merged = cv2.merge(layers)
        # Shrinking by a whole factor takes OpenCV's quick path, several times quicker
        # than an uneven one. Dropping the last few rows and columns past a multiple of the
        # factor makes it whole; they're under four pixels at the area's edge, and every
        # candidate the guess leads to is still checked exactly.
        exact = (size[0] * _MOVE_COARSE, size[1] * _MOVE_COARSE)
        if exact[0] <= merged.shape[1] and exact[1] <= merged.shape[0]:
            merged = merged[: exact[1], : exact[0]]
        if (bottom - top) * (right - left) >= _MOVE_HALVE_FIRST:
            # A big area is halved first while still whole levels, which turns most of the
            # float work into byte work: halving rounds by at most half a level, far below
            # what the shift guess notices, and every candidate is still checked exactly.
            half = (max(merged.shape[1] // 2, size[0]), max(merged.shape[0] // 2, size[1]))
            merged = cv2.resize(merged, half, interpolation=cv2.INTER_AREA)
        return _phase_guess(merged.astype(np.float32), size)

    def _drag_region(self, changed):
        """Where a dragged window can be this frame, as a box (top, bottom, left, right),
        and the changed pixels in it: around where the last move landed, widened by as
        far as the drag may have gone since, frames dropped in between included."""
        height, width = changed.shape[:2]
        top, bottom, left, right = self._drag_area
        dx, dy = self._last_move
        scale = self._gap / self._move_gap
        reach = _MOVE_REPEAT_REACH + 3 * (max(self._gap, self._move_gap) - 1)
        grow_x = int(abs(dx) * scale) + 2 * reach + self._area_near
        grow_y = int(abs(dy) * scale) + 2 * reach + self._area_near
        box = (
            max(top - grow_y, 0), min(bottom + grow_y, height),
            max(left - grow_x, 0), min(right + grow_x, width),
        )
        region = np.zeros_like(changed)
        region[box[0]:box[1], box[2]:box[3]] = changed[box[0]:box[1], box[2]:box[3]]
        return box, region

    def _repeat_move(self, image, changed, box):
        """The shift near last frame's move that changed pixels now match, or None.
        Changed pixels lie within box (top, bottom, left, right)."""
        self._repeat_tried = ([], 0)  # the shifts looked around, and how far, for the log
        found = _changed_box(changed, box)
        if found is None:
            return None
        top, bottom, left, right = found
        if bottom - top < self._move_min_height or right - left < self._move_min_width:
            return None
        now = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        mask8 = np.ascontiguousarray(changed[top:bottom, left:right]).view(np.uint8)
        # Frames dropped while the scanner was busy stretch a drag's shift: two captures
        # apart, it moved twice as far. So the shift expected is the last one per capture
        # times the captures since, and where the drag is heading adds its last change in
        # speed, for a drag that speeds up, slows down or turns.
        last, gap = self._last_move, self._gap
        speed = (last[0] / self._move_gap, last[1] / self._move_gap)
        centres = [(round(speed[0] * gap), round(speed[1] * gap))]
        before = self._drag_before
        if before is not None:
            # Speeds are averages over each move, so they belong to its middle. The change
            # in speed per capture, carried from the last move's middle across this gap.
            earlier = (before[0] / self._before_gap, before[1] / self._before_gap)
            span = (self._move_gap + self._before_gap) / 2
            reach = gap * (gap + self._move_gap) / 2 / span
            centres.append((
                round(speed[0] * gap + (speed[0] - earlier[0]) * reach),
                round(speed[1] * gap + (speed[1] - earlier[1]) * reach),
            ))
        extra = []
        for centre in centres:
            if all(max(abs(centre[0] - c[0]), abs(centre[1] - c[1])) > 2 for c in [last, *extra]):
                extra.append(centre)
        # The farther back the frame compared against, the farther the drag can stray
        # from where it was heading: the window widens by two pixels a dropped capture.
        # A narrow window only: checking it costs about a millisecond and a half. A drag
        # that strayed farther, as a hand does across dropped frames, is found by the one
        # search over its region that follows a miss, which costs less than a wide window.
        reach = _MOVE_REPEAT_REACH
        self._repeat_tried = ([last, *extra], reach)
        return self._checked_move(
            image, now, mask8, (top, bottom, left, right), last, extra, reach
        )

    def _checked_move(self, image, now, mask8, area, guess, also=(), reach=_MOVE_COARSE):
        """The candidate shift within reach of guess, or of any shift in also, that
        nearly all sampled text pixels match."""
        top, bottom, left, right = area

        textured = cv2.bitwise_and(_texture(now).view(np.uint8), mask8)
        # In the same row-by-row order as np.nonzero, several times faster.
        found = cv2.findNonZero(textured)
        if found is None or len(found) < _MOVE_SAMPLES // 8:
            return None
        found = found.reshape(-1, 2).astype(np.intp)  # x, y; some OpenCV builds add an axis
        xs, ys = found[:, 0], found[:, 1]
        if ys.size > _MOVE_SAMPLES:
            pick = np.linspace(0, ys.size - 1, _MOVE_SAMPLES).astype(int)
            ys, xs = ys[pick], xs[pick]
        ys, xs = ys + top, xs + left
        now_pixels = image[ys, xs].astype(np.int16)
        candidates = {
            (centre[0] + ex, centre[1] + ey)
            for centre in [guess, *also]
            for ex in range(-reach, reach + 1)
            for ey in range(-reach, reach + 1)
        }
        if self._last_move is not None:
            candidates.add(self._last_move)
        candidates.discard((0, 0))
        candidates = sorted(candidates)
        # Every candidate first votes on a few evenly spread samples. The right shift
        # matches nearly all of them, so only the leaders need the full vote.
        step = max(ys.size // _MOVE_SCREEN_SAMPLES, 1)
        if reach > _MOVE_COARSE:
            # A wide window first votes on a quarter of the samples, so it costs about
            # what a narrow one does; its leaders then get the usual screening.
            wide = step * 4
            first = _shift_shares(
                image, self._previous, ys[::wide], xs[::wide], now_pixels[::wide], candidates
            )
            keep = np.argsort(-first, kind='stable')[:_MOVE_FINALISTS * 4]
            candidates = [candidates[i] for i in sorted(keep) if first[i] > 0]
            if not candidates:
                return None
        screen = _shift_shares(
            image, self._previous, ys[::step], xs[::step], now_pixels[::step], candidates
        )
        leaders = np.argsort(-screen, kind='stable')[:_MOVE_FINALISTS]
        finalists = [candidates[i] for i in leaders if screen[i] > 0]
        if not finalists:
            return None
        shares = _shift_shares(image, self._previous, ys, xs, now_pixels, finalists)
        best = int(np.argmax(shares))
        shift, share = finalists[best], float(shares[best])
        # The best candidate may sit at the edge of the shifts tried, a pixel short of the
        # real one, which lies just outside: text a pixel off still matches a fair share of
        # samples. So step to any better neighbour, tried or not, until none is better. The
        # real shift matches nearly every sample and its neighbours far fewer, so this
        # settles on it.
        seen = {shift}
        for _ in range(_MOVE_CLIMB):
            around = [
                (shift[0] + ex, shift[1] + ey)
                for ex in (-1, 0, 1) for ey in (-1, 0, 1)
                if (shift[0] + ex, shift[1] + ey) not in seen
                and (shift[0] + ex, shift[1] + ey) != (0, 0)
            ]
            if not around:
                break
            seen.update(around)
            scores = _shift_shares(image, self._previous, ys, xs, now_pixels, around)
            better = int(np.argmax(scores))
            if scores[better] <= share:
                break
            shift, share = around[better], float(scores[better])
        if share < _MOVE_MIN_SHARE:
            return None
        return shift

    def _estimate_move(self, image, changed, box=None):
        started = time.monotonic()
        try:
            return self._estimate_move_timed(image, changed, box)
        finally:
            self.stats['estimate_seconds'] += time.monotonic() - started

    def _estimate_move_timed(self, image, changed, box=None):
        """The dx, dy a window moving whole within changed explains best, or None. If
        box is given, every changed pixel lies within it."""
        found = _changed_box(changed, box or (0, changed.shape[0], 0, changed.shape[1]))
        if found is None:
            return None
        top, bottom, left, right = found
        if bottom - top < self._move_min_height or right - left < self._move_min_width:
            return None
        return self._move_shift(image, changed, (top, bottom, left, right))

    def _follow_move(self, image, changed, shift=None):
        started = time.monotonic()
        try:
            return self._follow_move_timed(image, changed, shift)
        finally:
            self.stats['move_seconds'] += time.monotonic() - started

    def _follow_move_timed(self, image, changed, shift=None):
        """If a window moved whole, move known text and covers along with it.

        Something moved whole when its pixels now show exactly what the previous frame
        showed one fixed dx, dy away. Known text moves only if every pixel of its new
        place matches; anything else is read again, as is every pixel the move doesn't
        explain, so a wrong shift only costs time. The shift is estimated from changed,
        unless one is given. Returns what still needs reading.
        """
        rows = np.flatnonzero(changed.any(axis=1))
        if not rows.size:
            return changed
        columns = np.flatnonzero(changed.any(axis=0))
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(columns[0]), int(columns[-1]) + 1
        if bottom - top < self._move_min_height or right - left < self._move_min_width:
            return changed

        if shift is None:
            shift = self._move_shift(image, changed, (top, bottom, left, right))
        if shift is None:
            self._last_move = None
            return changed
        dx, dy = shift
        height, width = image.shape[:2]
        # Where moved content can be now: the changed area and its own copy dx, dy away.
        area = (
            max(min(top, top + dy), 0), min(max(bottom, bottom + dy), height),
            max(min(left, left + dx), 0), min(max(right, right + dx), width),
        )
        inside = np.s_[area[0]:area[1], area[2]:area[3]]
        match = np.zeros_like(changed)
        match[inside] = _shifted_match(image, self._previous, dx, dy, area)
        # Nothing matches outside area, so only area needs combining with changed.
        if not (changed[inside] & match[inside]).any():
            return changed
        self._last_move = shift
        self._moved = True
        self._moved_area = area  # top, bottom, left, right of what moved, and where to
        self.stats['moves'] += 1
        remaining = changed.copy()
        remaining[inside] &= ~match[inside]
        # Where the moved content landed: changed pixels the shift explains. A drag that
        # carries on is looked for around here next frame, not across the whole screen.
        moved = (changed[inside] & match[inside]).view(np.uint8)
        rows = np.flatnonzero(cv2.reduce(moved, 1, cv2.REDUCE_MAX))
        columns = np.flatnonzero(cv2.reduce(moved, 0, cv2.REDUCE_MAX))
        self._drag_area = (
            area[0] + int(rows[0]), area[0] + int(rows[-1]) + 1,
            area[2] + int(columns[0]), area[2] + int(columns[-1]) + 1,
        )

        def action(rect) -> str:
            x1, y1, x2, y2 = rect[:4]
            old = (
                max(math.floor(y1), 0), min(math.ceil(y2), height),
                max(math.floor(x1), 0), min(math.ceil(x2), width),
            )
            if old[1] <= old[0] or old[3] <= old[2] or not changed[
                old[0]:old[1], old[2]:old[3]
            ].any():
                return STAY  # nothing changed where it was: still text beside the move
            new = (
                max(math.floor(y1 + dy), 0), min(math.ceil(y2 + dy), height),
                max(math.floor(x1 + dx), 0), min(math.ceil(x2 + dx), width),
            )
            if new[1] <= new[0] or new[3] <= new[2]:
                return REREAD  # it would leave the frame, so nothing is known about it
            if match[new[0]:new[1], new[2]:new[3]].all():
                return MOVE  # what's left of it on the frame shows exactly what it had
            return REREAD

        moved = []
        for region in self._regions:
            todo = action(region)
            x1, y1, x2, y2, text = region
            if todo == STAY:
                moved.append(region)
            elif todo == MOVE:
                moved.append(self._carry(region, (x1 + dx, y1 + dy, x2 + dx, y2 + dy, text)))
            else:
                # Read the line whole wherever it may be now, never just the part that changed.
                for place in ((x1, y1, x2, y2), (x1 + dx, y1 + dy, x2 + dx, y2 + dy)):
                    remaining[_inside(place)] = True
        for place in self._tracks.follow_move(action, dx, dy):
            remaining[_inside(place)] = True
            remaining[_inside((place[0] + dx, place[1] + dy, place[2] + dx, place[3] + dy))] = True
        self._regions = moved
        return remaining

    def _checked_step(self, image, shift, fraction, left, right) -> ScrollStep:
        """This scroll, for following black incoming rows into later frames: every
        pixel of the pane's view that the shift doesn't reproduce, changed or not."""
        top_of_view, bottom_of_view = self._view
        length = bottom_of_view - top_of_view - abs(shift)
        first, old = top_of_view + max(shift, 0), top_of_view + max(-shift, 0)
        if abs(fraction) >= _WHOLE_PIXEL:
            length -= 1  # the row at the incoming edge is partly one not seen yet
            if shift > 0:
                first, old = first + 1, old + 1
        length = max(length, 0)
        # Only the rows the followed strips land on need checking: the rest of the pane
        # is the scroll's business, checked as ever. Rows outside count as unchecked.
        rows = self._incoming.rows_after(shift + fraction)
        if rows is not None:
            start, stop = max(rows[0], first), min(rows[1], first + length)
            if stop <= start:
                start = stop = first
            old += start - first
            first, length = start, stop - start
        unexplained = _unexplained(
            image[first:first + length, left:right], self._previous, old, left, right, fraction
        )
        return ScrollStep(shift + fraction, left, right, first, unexplained, self._view)

    def _cut(self, image, edge, inward, left, right) -> int:
        """Rows in from a pane edge that the line it cuts through takes, if any."""
        return _cut_line(
            image, edge, inward, left, min(right, image.shape[1]),
            _CUT_REACH * self._unread_keep, self._pixel_scale, self._unread_keep,
        )

    def _collect_unread(self, image, shift, left, right, first, length, remaining):
        """Track rows new to the screen. They stay black until they can be read.

        A line coming onto the screen can't be read until enough of it shows, so
        rows wait here, covered, until there are enough to hold whole lines. These
        are per-frame covers, not secrets: each frame reports only what is still
        unread, and a read replaces them with whatever it actually found.
        """
        top_of_view, bottom_of_view = self._view
        rows = (top_of_view, first) if shift > 0 else (first + length, bottom_of_view)
        incoming = (rows[0], rows[1], max(left - self._scroll_edge, 0), right + self._scroll_edge)
        direction = 1 if shift > 0 else -1
        strips = {}

        for way, strip in self._unread.items():
            if strip[2] >= right or strip[3] <= left:
                # Its pane stopped scrolling, so what it covers can be read now.
                remaining[strip[0]:strip[1], strip[2]:strip[3]] = True
                continue
            moved = (
                max(strip[0] + shift, top_of_view), min(strip[1] + shift, bottom_of_view),
                strip[2], strip[3],
            )
            if moved[1] <= moved[0]:
                continue
            if way == direction:
                incoming = _join(incoming, moved)
            else:
                strips[way] = moved  # a reversal left these rows at the other edge

        waited = self._unread_waited.get(direction, 0) + 1
        rows_in = incoming[1] - incoming[0]
        # What can't be read yet: the line the edge cuts through, if it cuts one.
        edge = top_of_view if shift > 0 else bottom_of_view
        keep = self._cut(image, edge, direction, incoming[2], incoming[3])
        if rows_in >= self._unread_read_at or (
            # Read what came in once it holds more than the line the edge cuts through:
            # a whole line came in. A slow scroll gets there long before 96 rows do.
            waited >= _UNREAD_MIN_WAIT
            and rows_in >= keep + _UNREAD_MIN_NEW * self._pixel_scale
        ):
            waited = 0
            remaining[incoming[0]:incoming[1], incoming[2]:incoming[3]] = True
            if shift > 0:
                incoming = (incoming[0], incoming[0] + keep, *incoming[2:])
            else:
                incoming = (incoming[1] - keep, incoming[1], *incoming[2:])

        if incoming[1] > incoming[0]:
            strips[direction] = incoming
        self._unread = strips
        self._unread_waited = {direction: waited}

    def _read_unread(self, changed) -> None:
        """Scrolling has stopped, so read whatever is still covered."""
        if not self._scrolled:
            for top, bottom, left, right in self._unread.values():
                changed[top:bottom, left:right] = True
            self._unread = {}
            self._unread_waited = {}
            self._view = None
            self._fractional_frames = 0

    def _follow(self, image, changed):
        """Follow a scroll, or else a moved window, across everything in changed."""
        changed = self._follow_scroll(image, changed)
        if not self._scrolled:
            changed = self._follow_move(image, changed)
        return changed

    def _follow_part(self, image, changed, part):
        """Follow within part of changed only. Returns what still needs reading: what
        following part left, plus everything changed outside it."""
        return _outside(self._follow(image, part), part, changed)

    def _areas(self, changed):
        started = time.monotonic()
        try:
            return self._areas_timed(changed)
        finally:
            self.stats['areas_seconds'] += time.monotonic() - started

    def _areas_timed(self, changed):
        """The separate areas of change: (areas big enough to follow, largest first;
        smaller pieces). Each is top, bottom, left, right. They only steer where to look:
        every follow is still checked pixel by pixel."""
        block = self._area_block
        height, width = changed.shape
        rows, columns = height // block, width // block
        if not rows or not columns:
            return [], []
        # Every other pixel, and whole blocks only so the shrink takes OpenCV's fast path:
        # a third of the time. What this misses, a one-pixel line or the last few rows
        # and columns, just isn't part of any area, so it's followed with the rest.
        blocks = cv2.resize(
            changed[:rows * block:2, :columns * block:2].view(np.uint8) * np.uint8(255),
            (columns, rows), interpolation=cv2.INTER_AREA,
        ) > 0
        join_x, join_y = (-(-join // (2 * block)) for join in self._area_join)
        joined = cv2.dilate(
            blocks.view(np.uint8), np.ones((2 * join_y + 1, 2 * join_x + 1), np.uint8)
        )
        count, labels, boxes, _ = cv2.connectedComponentsWithStats(joined, connectivity=4)
        areas, pieces = [], []
        for label in range(1, count):
            x, y, w, h = (int(v) for v in boxes[label, :4])
            loose = (y * block, (y + h) * block, x * block, (x + w) * block)
            # Joining grew each area by up to its margins; too small even so, it's a piece.
            if h * block < self._move_min_height or w * block < self._move_min_width:
                pieces.append(loose)
                continue
            inside = (labels[y:y + h, x:x + w] == label) & blocks[y:y + h, x:x + w]
            ys, xs = np.nonzero(inside)
            top, bottom = (int(ys.min()) + y) * block, (int(ys.max()) + y + 1) * block
            left, right = (int(xs.min()) + x) * block, (int(xs.max()) + x + 1) * block
            if bottom - top >= self._move_min_height and right - left >= self._move_min_width:
                areas.append((top, bottom, left, right))
            else:
                pieces.append(loose)
        areas.sort(key=lambda a: (a[1] - a[0]) * (a[3] - a[2]), reverse=True)
        return areas, pieces

    @staticmethod
    def _overlap(box, entry) -> int:
        return (
            max(min(box[1], entry[1]) - max(box[0], entry[0]), 0)
            * max(min(box[3], entry[3]) - max(box[2], entry[2]), 0)
        )

    def _failing_entry(self, area):
        """The remembered failing area this one is: mostly inside it, and not much
        smaller. None if there isn't one."""
        size = (area[1] - area[0]) * (area[3] - area[2])
        for entry in self._failing:
            entry_size = (entry[1] - entry[0]) * (entry[3] - entry[2])
            if (
                self._overlap(area, entry) >= _AREA_INSIDE * size
                and size >= _AREA_SIMILAR * entry_size
            ):
                return entry
        return None

    def _in_failing(self, piece) -> bool:
        """Whether a small piece of change lies in an area left out for failing, like
        the cursor's copy in OBS's preview."""
        size = (piece[1] - piece[0]) * (piece[3] - piece[2])
        return any(
            entry[4] >= _AREA_FAILS and self._overlap(piece, entry) >= _AREA_INSIDE * size
            for entry in self._failing
        )

    def _follow_areas(self, image, changed):
        """Follow scrolls and moved windows, even beside changes that can't be followed.

        Changes fall in separate areas. In this order, until something is followed:
        1. The quick scroll check across all of them together, as one pane's lines can
           fall apart. When every change is small, like a cursor and its copy in the
           preview, that's all: windows that move change a big area, their frame.
        2. Each area big enough to follow, alone and largest first, with every check, so
           a note still scrolls, or a window still moves, beside OBS's live preview,
           whose shrunken copy of the move would otherwise compete with it.
        3. A move across the big areas together, for a window whose change fell apart.
        An area whose changes went unexplained several frames running is left out of the
        tries, with the small pieces inside it, and only tried again now and then. Left
        out never means skipped: its changes, like every change no follow explains, are
        all read again.
        """
        self._follow_frame += 1
        frame = self._follow_frame
        out_of_time = time.monotonic() + _FOLLOW_BUDGET
        # Motion carries on from frame to frame. The costly tries, the window-move and
        # column searches, run in full for the kind of motion last frame followed; the
        # other kind stops once a frame has spent its budget, so a big change nothing
        # explains, like a page swapped whole, doesn't cost several frames' time. Every
        # few frames everything is tried in full, so a drag or scroll that starts is
        # still found. What isn't followed is read, as always.
        full = frame % _FOLLOW_RETRY == 0
        self._follow_deadline = None if full else time.monotonic() + _FOLLOW_STRICT
        areas, pieces = self._areas(changed)
        left_out, due = [], []
        for area in areas:
            entry = self._failing_entry(area)
            if entry is not None and entry[4] >= _AREA_FAILS:
                entry[6] = frame
                (due if frame - entry[5] >= _AREA_RETRY else left_out).append(area)
        tried = [area for area in areas if area not in left_out and area not in due]
        hidden = [piece for piece in pieces if self._in_failing(piece)]
        visible = [piece for piece in pieces if piece not in hidden]
        others = bool(visible)

        part = changed
        if left_out or due or hidden:
            self.stats['areas_left_out'] += 1
            part = changed.copy()
            for top, bottom, left, right in left_out + due + hidden:
                part[top:bottom, left:right] = False
        last_move = self._last_move
        crowded = len(tried) > 1 or bool(left_out or due or hidden)
        remaining = changed
        # An area is judged by a try that tells: alone, or any try that followed something.
        # A failed try of several at once proves nothing about each.
        judged = []
        drag = (
            self._last_followed == 'move' and last_move is not None
            and self._drag_area is not None
        )
        drag_missed = False
        if drag:
            # A drag carries on about where it was heading. It's looked for only where the
            # dragged window can be now, so changes elsewhere, like a reply streaming in
            # another window, neither cost time nor outvote its pixels:
            # 1. the shifts around the one expected, checked pixel by pixel as always;
            # 2. if none fits, one search for the shift, over that region alone.
            # Anything else big changing is checked for a scroll first, without the drag:
            # a pane scrolling in a second window must be followed to stay covered. A
            # straight up or down shift isn't taken here, as it may be a scroll. When the
            # drag stops, nothing fits, and everything is tried as usual.
            box, region = self._drag_region(part)
            elsewhere = [area for area in tried if not _overlaps(area, box)] + due
            if elsewhere:
                rest = part.copy()
                rest[box[0]:box[1], box[2]:box[3]] = False
                scrolled = self._follow_scroll_in(image, rest)
                if self._scrolled:
                    self.stats['repeat_crowded'] += 1
                    remaining = _outside(scrolled, rest, changed)
            if not self._scrolled:
                self.stats['repeat_tries'] += 1
                started = time.monotonic()
                shift = self._repeat_move(image, region, box)
                self.stats['repeat_seconds'] += time.monotonic() - started
                if shift is None:
                    self._last_move = last_move
                    shift = self._estimate_move(image, region, box)
                    self._last_move = last_move
                    if shift is not None:
                        self.stats['drag_searched'] += 1
                if shift is not None and shift[0] == 0:
                    self.stats['repeat_upright'] += 1
                    shift = None
                if shift is not None:
                    self.stats['moves_repeated'] += 1
                    remaining = _outside(self._follow_move(image, part, shift), part, changed)
                else:
                    self.stats['repeat_missed'] += 1
                    drag_missed = True
        if (len(tried) > 1 or others) and not (self._moved or self._scrolled or drag_missed):
            # Only the quick check: each big area gets the full one, alone, next.
            scrolled = self._follow_scroll_in(image, part)
            remaining = _outside(scrolled, part, changed)

        for area in tried + due:
            if self._scrolled or self._moved:
                break
            if judged and time.monotonic() > out_of_time:
                # Busy frames can hold many areas no scroll or move explains, and trying
                # each in full costs a frame's time several times over. The rest are read
                # as they are, which is always safe, and tried again next frame.
                self.stats['follow_cut'] += 1
                break
            near, block = self._area_near, self._area_block
            # The area a block wider, for pixels at its edge that finding it skipped.
            grown = (
                max(area[0] - block, 0), area[1] + block, max(area[2] - block, 0),
                area[3] + block,
            )
            # Wide: everything but the other big areas, so the rest of a pane or window
            # comes along, however far from this area. Close: the area and the pieces
            # right by it, as farther changes, like the cursor, can spoil a check.
            wide = part.copy()
            for top, bottom, left, right in tried:
                if (top, bottom, left, right) != area:
                    wide[top:bottom, left:right] = False
            wide[grown[0]:grown[1], grown[2]:grown[3]] = changed[
                grown[0]:grown[1], grown[2]:grown[3]
            ]
            close = np.zeros_like(changed)
            for top, bottom, left, right in [grown] + [
                piece for piece in visible
                if piece[0] < area[1] + near and piece[1] > area[0] - near
                and piece[2] < area[3] + near and piece[3] > area[2] - near
            ]:
                close[top:bottom, left:right] = changed[top:bottom, left:right]

            # A scroll is checked close. For the few frames before a busy area beside it
            # is left out, the rest of the pane may be read again instead of followed.
            # Even right after a drag: a pane can scroll inside a window being dragged,
            # and only following that scroll keeps it covered.
            remaining = _outside(self._follow_scroll(image, close), close, changed)
            if not self._scrolled:
                # A move is applied wide, each pixel still checked against the shift. With
                # nothing else big changing, the shift is estimated wide too, as before.
                # Beside other changes, like OBS's preview showing a shrunken copy of the
                # move, it's estimated close first, where those can't spoil it.
                self._last_move = last_move  # a failed try forgets it; this one may need it
                shift = (
                    self._estimate_move(image, close)
                    if crowded and self._in_time('move') and not drag_missed else None
                )
                # The largest area always gets its full move try, so a drag that starts
                # is followed from its first frame; the budget only stops the rest.
                if shift is None and (not judged or self._in_time('move')):
                    self._last_move = last_move
                    shift = self._estimate_move(image, wide)
                if shift is None:
                    self._last_move = None
                else:
                    remaining = _outside(self._follow_move(image, wide, shift), wide, changed)
            judged.append(area)

        if drag_missed:
            # How far the drag's real shift was from the shifts the quick check tried, so
            # a live run's log tells irregular motion from a check that couldn't decide.
            if not self._moved:
                self.stats['miss_lost'] += 1
            else:
                found, (centres, reach) = self._last_move, self._repeat_tried
                off = min(
                    (max(abs(found[0] - c[0]), abs(found[1] - c[1])) for c in centres),
                    default=math.inf,
                )
                bucket = 'miss_inside' if off <= reach else (
                    'miss_near' if off <= 3 * reach else 'miss_far'
                )
                self.stats[bucket] += 1
        if len(tried) > 1 and not (self._scrolled or self._moved) and (
            time.monotonic() <= out_of_time and self._in_time('move')
        ):
            together = np.zeros_like(changed)
            for top, bottom, left, right in tried:
                together[top:bottom, left:right] = changed[top:bottom, left:right]
            self._last_move = last_move
            remaining = _outside(self._follow_move(image, together), together, changed)

        # The drag's shift a frame before this one's, for where it's heading next, and the
        # captures each spanned.
        carries_on = self._moved and self._last_followed == 'move'
        self._drag_before = last_move if carries_on else None
        self._before_gap = self._move_gap if carries_on else 1
        if self._moved:
            self._move_gap = self._gap
        self._last_followed = 'scroll' if self._scrolled else 'move' if self._moved else None
        if self._scrolled or self._moved:
            # Only one scroll or move is followed a frame, so any other area left
            # unexplained has failed for this frame, whether or not it was tried alone.
            judged = list(dict.fromkeys(judged + tried))
        for area in judged:
            top, bottom, left, right = area
            entry = self._failing_entry(area)
            inside = changed[top:bottom, left:right]
            explained = inside & ~remaining[top:bottom, left:right]
            share = cv2.countNonZero(explained.view(np.uint8)) / max(
                cv2.countNonZero(inside.view(np.uint8)), 1
            )
            if share >= _AREA_EXPLAINED:
                if entry is not None:
                    self._failing.remove(entry)
            elif entry is not None:
                entry[:4], entry[4], entry[5], entry[6] = area, entry[4] + 1, frame, frame
            else:
                self._failing.append([*area, 1, frame, frame])
        self._failing = [e for e in self._failing if frame - e[6] <= _AREA_FORGET]
        return remaining

    def _changed_areas(self, image) -> list[tuple[int, int, int, int]]:
        """Areas to read again, each grown to hold whole pieces of known text."""
        height, width = image.shape[:2]

        if self._previous is None or self._previous.shape != image.shape:
            return [(0, height, 0, width)]

        pixels_started = time.monotonic()
        if self._noise_floor:
            changed = self._changed_blocks(image)
        else:
            difference = cv2.split(cv2.absdiff(image, self._previous))
            changed = cv2.bitwise_or(
                cv2.bitwise_or(difference[0], difference[1]), difference[2]
            ) > 0
        self.stats['pixels_seconds'] += time.monotonic() - pixels_started
        if self._incoming.following:
            self._last_changed = changed.copy()  # the steps below mark it up
        if self._noise_floor:
            # A camera's noise never matches exactly, and its pictures aren't windows.
            changed = self._follow_scroll(image, changed)
        else:
            changed = self._follow_areas(image, changed)
        self._read_unread(changed)

        for place in self._tracks.due():
            changed[_inside(place)] = True
            self.stats['checks'] += 1

        rows = np.flatnonzero(changed.any(axis=1))
        bands = _merge_bands([
            (max(int(row) - self._band_margin, 0), min(int(row) + self._band_margin + 1, height))
            for row in rows
        ])
        areas = []

        for top, bottom in bands:
            columns = np.flatnonzero(changed[top:bottom].any(axis=0))
            areas.append((
                top, bottom,
                max(int(columns[0]) - self._side_margin, 0),
                min(int(columns[-1]) + self._side_margin + 1, width),
            ))

        known = [
            (
                max(math.floor(y1), 0), min(math.ceil(y2), height),
                max(math.floor(x1), 0), min(math.ceil(x2), width),
            )
            for x1, y1, x2, y2 in [*(r[:4] for r in self._regions), *self._tracks.places()]
        ]

        # Grow until no area cuts through known text or overlaps another area.
        grow_started = time.monotonic()
        grew = True
        while grew:
            grew = False
            for index, area in enumerate(areas):
                for other in known + areas[index + 1:]:
                    if _overlaps(area, other) and _join(area, other) != area:
                        areas[index] = area = _join(area, other)
                        grew = True
            kept = []
            for area in areas:
                if not any(_overlaps(area, other) for other in kept):
                    kept.append(area)
                else:
                    index = next(i for i, other in enumerate(kept) if _overlaps(area, other))
                    kept[index] = _join(kept[index], area)
                    grew = True
            areas = kept

        self.stats['grow_seconds'] += time.monotonic() - grow_started
        scanned = sum((bottom - top) * (right - left) for top, bottom, left, right in areas)
        if scanned > _FULL_SCAN_SHARE * height * width:
            return [(0, height, 0, width)]

        return areas

    def _changed_blocks(self, image) -> np.ndarray:
        """Pixels in blocks whose average colour moved past the noise floor.

        Averaging a block cancels sensor noise but not new text: a character changes
        many pixels of its block by far more than noise does.
        """
        height, width = image.shape[:2]
        size = _CHANGE_BLOCK
        rows, columns = -(-height // size), -(-width // size)  # round up: edges count too

        def small(picture):
            padded = cv2.copyMakeBorder(
                picture, 0, rows * size - height, 0, columns * size - width,
                cv2.BORDER_REPLICATE,
            )
            return cv2.resize(padded, (columns, rows), interpolation=cv2.INTER_AREA)

        moved = _strongest(cv2.absdiff(small(image), small(self._previous))) > self._noise_floor
        changed = np.repeat(np.repeat(moved, size, axis=0), size, axis=1)
        return np.ascontiguousarray(changed[:height, :width])

    def _remember(self, snapshot, areas) -> None:
        """Keep this frame as the one later frames are compared with.

        With a noise floor, only the areas just read take the new pixels. Elsewhere the
        old pixels stay, so a slow change can't creep past the floor one frame at a time.
        """
        if not self._noise_floor or self._previous is None or self._scrolled:
            self._previous = snapshot
            return
        previous = self._previous.copy()
        for top, bottom, left, right in areas:
            previous[top:bottom, left:right] = snapshot[top:bottom, left:right]
        self._previous = previous

    def _read(self, image, top: int, left: int) -> list[tuple]:
        """Read one area and return text regions in full-frame coordinates."""
        read_area = getattr(self._reader, 'read_area', None)
        results, timings = read_area(image, top, left) if read_area else self._reader(image)
        self.stats['reads'] += 1
        self.stats['lines'] += len(results or [])
        if isinstance(timings, (list, tuple)) and len(timings) == 3:
            self.stats['det_seconds'] += timings[0] or 0
            self.stats['rec_seconds'] += timings[2] or 0
        return self._parse(results, top, left)

    @staticmethod
    def _parse(results, top: int, left: int) -> list[tuple]:
        """OCR results as text regions in full-frame coordinates. Raises if malformed."""
        regions = []

        for polygon, text, _ in results or []:
            points = np.asarray(polygon, dtype=float)

            if (
                points.shape != (4, 2)
                or not np.isfinite(points).all()
                or not isinstance(text, str)
            ):
                raise ValueError('invalid OCR result')

            regions.append((
                float(points[:, 0].min()) + left,
                float(points[:, 1].min()) + top,
                float(points[:, 0].max()) + left,
                float(points[:, 1].max()) + top,
                text,
            ))

        return regions

    def _lines(self, regions, image=None) -> list[tuple[list[tuple], list[str]]]:
        """Neighbouring text on the same row, as lines read left to right, each with how
        its pieces meet: TIGHT, SPACE or UNREAD, one for each gap. Lines give each value
        its label and join secrets OCR split into pieces.

        Gaps are measured against the row's usual text height, not either piece's own
        box: OCR gives "=" or "." a box a few pixels tall, which would otherwise part a
        password from its label. Text far apart on a row, as in two windows side by side,
        stays apart, unless a label ends the piece before and the pixels between show a
        mark OCR didn't read, as when Vision drops the "=" of "password = hunter2".
        Pieces that overlap are two readings of the same text, so they go in separate
        lines instead of being joined into a token neither reading had.
        """
        rows = []
        # Text comes top first, so a row ending above this text can't take it or any
        # after it. Only rows still reaching down this far are compared: the same rows
        # match, first one first, without comparing every word with every row on screen.
        active = []

        for region in sorted(regions, key=lambda r: (r[1], r[0])):
            _, y1, _, y2, _ = region
            active = [row for row in active if row[1] >= y1]

            for row in active:
                overlap = min(row[1], y2) - max(row[0], y1)
                if overlap >= 0.5 * min(row[1] - row[0], y2 - y1):
                    row[0] = min(row[0], y1)
                    row[1] = max(row[1], y2)
                    row[2].append(region)
                    break
            else:
                row = [y1, y2, [region]]
                rows.append(row)
                active.append(row)

        lines = []

        for _, _, parts in rows:
            parts.sort(key=lambda r: r[0])
            heights = sorted(r[3] - r[1] for r in parts if _has_word(r[4]))
            heights = heights or sorted(r[3] - r[1] for r in parts)
            height = heights[len(heights) // 2]  # the row's usual height: its median
            open_lines: list[tuple[list[tuple], list[str]]] = []

            for part in parts:
                placed = False
                for line in list(open_lines):
                    last = line[0][-1]
                    gap = part[0] - last[2]
                    if gap > _UNREAD_REACH * height:
                        open_lines.remove(line)  # everything after this is farther still
                        lines.append(line)
                        continue
                    if placed or _alternatives(last, part):
                        continue
                    kind = TIGHT if gap < _TIGHT_GAP * height else SPACE
                    if gap > _TIGHT_GAP * height and (
                        ends_in_label(last[4])
                        or gap > _MARK_GAP * height
                        and any(names_a_secret(p[4]) for p in line[0])
                    ) and _marked(image, last, part, height):
                        kind = UNREAD
                    elif gap > _JOIN_GAP * height:
                        continue
                    line[0].append(part)
                    line[1].append(kind)
                    placed = True
                if not placed:
                    open_lines.append(([part], []))

            lines.extend(open_lines)

        return lines

    def _cover(self, part, start: int, end: int) -> tuple[float, float, float, float]:
        """Estimate where characters start to end sit inside a text region.

        OCR places whole words at best. A match that stops partway through a word
        has no known edge, so it takes the whole word. Apple's OCR gives one region
        per word, so there this is that word's own box.
        """
        x1, y1, x2, y2, text = part

        while start > 0 and not text[start - 1].isspace():
            start -= 1

        while end < len(text) and not text[end].isspace():
            end += 1

        width = (x2 - x1) / max(len(text), 1)
        # Characters vary in width, so allow extra rather than clip the secret.
        left = max(x1, x1 + (start - _CHAR_MARGIN) * width)
        right = min(x2, x1 + (end + _CHAR_MARGIN) * width)
        return left, y1, right, y2

    def _find_areas(self, regions, image=None) -> set[tuple]:
        """What to cover: (category, rect, text hash) for each piece of each secret.

        Each line is judged as a whole, so a value is found with its label ("password:")
        or turned down by it ("Order number:"). Each piece of text is also judged alone,
        for values that need no label, in case a neighbour joined onto them breaks their
        pattern. A rule gives the value's span, and each piece it touches is covered.
        """
        areas = set()
        lines = self._lines(regions, image)
        uncertain = 0
        self._doubtful = []  # (line's parts, ...) the fast read can't settle
        self._careful_areas = set()  # areas found only in a careful reread's words

        def add(span, part, first, last, text):
            area = (span.category, self._cover(part, first, last), _text_hash(text))
            areas.add(area)
            if id(part) in self._careful_ids:
                self._careful_areas.add(area)

        # A line under a label line is judged with it, so "Commit SHA:" over its hash, or
        # "Password:" over a form's field, gives the value its meaning. The label line is
        # judged alone too.
        alone = list(lines)
        for below, above in _labelled_from_above(alone).items():
            parts, kinds = alone[above]
            lines[below] = (parts + alone[below][0], kinds + [NEWLINE] + alone[below][1])

        for parts, kinds in lines:
            if len(parts) > 1:
                for region in parts:
                    for span in find_spans(region[4], self._keywords):
                        if span.standalone:
                            add(span, region, span.start, span.end, region[4][span.start:span.end])

            # As the words sit; joined where OCR split one token; and wholly joined, for
            # strong patterns only, as that loses the spaces a label's meaning needs.
            joins = [([_SEPARATORS[kind] for kind in kinds], False)]
            if TIGHT in kinds:
                joins.append(([_TIGHT_SEPARATORS[kind] for kind in kinds], False))
            if len(parts) > 1:
                joins.append(([''] * len(kinds), True))

            settled = False
            for separators, standalone_only in joins:
                joined = parts[0][4] + ''.join(
                    separator + part[4] for separator, part in zip(separators, parts[1:])
                )
                spans = find_spans(joined, self._keywords)
                if not standalone_only and self._careful_ids and any(
                    id(part) in self._careful_ids for part in parts
                ):
                    # A careful reread of a garbled line: a label two edits off counts.
                    spans = spans + find_careful_spans(joined)
                if not standalone_only and (
                    self._rescue is None or not _in_doubt(joined, spans)
                ):
                    settled = True  # doubts only matter when there's a reread to settle them

                for span in spans:
                    if standalone_only and not span.standalone:
                        continue
                    position = 0
                    touched = []

                    for index, part in enumerate(parts):
                        first = max(span.start - position, 0)
                        last = min(span.end - position, len(part[4]))
                        if last > first:
                            if len(part[4].split()) > 1:
                                # One box for several words can't place the value's
                                # first character, so the cover starts at its label.
                                first = min(first, max(span.anchor - position, 0))
                            touched.append((part, first, last))
                        position += len(part[4])
                        if index < len(separators):
                            position += len(separators[index])

                    if standalone_only and len(touched) < 2:
                        continue  # joining without spaces is only for split secrets

                    uncertain += span.uncertain
                    for part, first, last in touched:
                        add(span, part, first, last, joined[span.start:span.end])

            if not settled:
                self._doubtful.append(parts)

        self._uncertain = uncertain
        return areas

    def _carry(self, old: tuple, new: tuple) -> tuple:
        """new is old moved by a scroll or drag: it keeps old's marks, as a careful
        reread's word or a line waiting for one."""
        key = id(old)
        if key in self._careful_ids:
            self._careful_ids.add(id(new))
        if key in self._rescue_waiting:
            self._rescue_waiting.add(id(new))
        return new

    def _waiting_covers(self, width: int, height: int) -> set[tuple]:
        """Covers for values after a label the fast read garbled, until the careful reread
        settles them, so the value doesn't show for the half second that takes.

        Only after a word that names a secret or reads as junk and ends in a separator
        (RSIAI.wiod: abc): a word like "password" in a sentence covers nothing.
        """
        boxes = set()
        margin = _WAITING_MARGIN * self._pixel_scale
        for parts in self._doubtful:
            if not any(id(part) in self._rescue_waiting for part in parts):
                continue
            for index, part in enumerate(parts[:-1]):
                word = part[4].rstrip()
                if _ENDS_IN_SEPARATOR.search(word) and (
                    names_a_secret(word) or garbled_label(word)
                ):
                    for value in parts[index + 1 :]:
                        left = max(0, math.floor(value[0]) - margin)
                        top = max(0, math.floor(value[1]) - margin)
                        right = min(width, math.ceil(value[2]) + margin)
                        bottom = min(height, math.ceil(value[3]) + margin)
                        if right > left and bottom > top:
                            boxes.add((left, top, right - left, bottom - top, 'unread'))
                    break
        return boxes

    def _rescue_reads(self, image, fresh, moving: bool = False) -> list[tuple]:
        """Reread, carefully, lines the fast read left in doubt, and return what it saw.

        A line is in doubt when it names a secret ("password", "card number") but no value
        was found after it, or holds a value whose check failed. Only lines read this frame
        are tried, each crop with its whole line and room to the right for a value OCR
        missed. Neighbouring crops are read as one. The words come back as a second
        reading beside the first: never joined into it, and never clearing a cover.
        Answers are kept for the exact pixels they read. A time budget bounds the work.
        """
        now = time.monotonic()
        if self._rescue_clock is not None:
            self._rescue_allowance = min(
                _RESCUE_BURST,
                self._rescue_allowance + (now - self._rescue_clock) * self._rescue_budget,
            )
        self._rescue_clock = now
        height, width = image.shape[:2]
        crops = []
        for parts in self._doubtful:
            if not any(id(part) in fresh for part in parts) and not any(
                id(part) in self._rescue_waiting for part in parts
            ):
                continue
            self.stats['doubtful_lines'] += 1
            tall = max(float(np.median([p[3] - p[1] for p in parts])), 1.0)
            top = max(int(min(p[1] for p in parts) - 0.5 * tall), 0)
            bottom = min(int(math.ceil(max(p[3] for p in parts) + 0.5 * tall)), height)
            # The rows of this line, with room for a spell-check underline under it. Ink
            # outside them belongs to the lines above and below, and blanked out, as cut-off
            # neighbours made Vision misread a whole line (passwrod: abc as Rasswredac).
            own = (
                max(int(min(p[1] for p in parts) - 0.15 * tall), 0),
                min(int(math.ceil(max(p[3] for p in parts) + 0.4 * tall)), height),
            )
            if parts[-1][4].rstrip().endswith(':') or ends_in_label(parts[-1][4]):
                # A label with nothing after it: its value may be on the next line.
                bottom = min(int(math.ceil(bottom + 1.5 * tall)), height)
                own = (own[0], bottom)
            left = max(int(min(p[0] for p in parts) - tall), 0)
            right = min(int(math.ceil(max(p[2] for p in parts) + _RESCUE_REACH * tall)), width)
            for crop in crops:
                if top <= crop[1] + tall and bottom >= crop[0] - tall and (
                    left < crop[3] and right > crop[2]
                ):
                    crop[:4] = [min(crop[0], top), max(crop[1], bottom),
                                min(crop[2], left), max(crop[3], right)]
                    crop[4].append(own)
                    crop[5].extend(parts)
                    break
            else:
                crops.append([top, bottom, left, right, [own], list(parts)])
        # Lines left over for lack of time wait for a later frame, even on a still screen,
        # where nothing is read fresh again. They're dropped once read or gone.
        waiting = set()
        for crop in crops[_RESCUE_PER_FRAME:]:
            waiting.update(id(part) for part in crop[5])
        if moving:
            # A careful read takes 25-30 ms inside the scan; while text moves that holds up
            # frames and the covers jump. The lines wait, covered, until the screen settles.
            for crop in crops:
                waiting.update(id(part) for part in crop[5])
            self._rescue_waiting = waiting
            self.stats['rescue_deferred'] += len(crops)
            return []
        words = []
        for top, bottom, left, right, rows, members in crops[:_RESCUE_PER_FRAME]:
            pixels = np.array(image[top:bottom, left:right])
            keep = np.zeros(bottom - top, bool)
            for first, last in rows:
                keep[max(first - top, 0) : max(last - top, 0)] = True
            if not keep.all():
                edge = np.concatenate((pixels[:, 0], pixels[:, -1]))
                pixels[~keep] = np.median(edge, axis=0).astype(pixels.dtype)
            key = (top, left, pixels.shape, hash(pixels.tobytes()))
            if key in self._rescued:
                self.stats['rescue_cached'] += 1
                words.extend(self._rescued[key])
                continue
            if self._rescue_allowance <= 0:
                self.stats['rescue_skipped'] += 1
                waiting.update(id(part) for part in members)
                continue
            started = time.monotonic()
            try:
                found = self._parse(self._rescue(pixels)[0], top, left)
            except Exception:
                self.stats['rescue_failed'] += 1  # the fast read still stands
                continue
            finally:
                spent = time.monotonic() - started
                self._rescue_allowance -= spent
                self.stats['rescue_seconds'] += spent
            self.stats['rescues'] += 1
            self._rescued[key] = found
            while len(self._rescued) > _RESCUE_CACHE:
                self._rescued.pop(next(iter(self._rescued)))
            words.extend(found)
        self.stats['rescue_skipped'] += max(len(crops) - _RESCUE_PER_FRAME, 0)
        self._rescue_waiting = waiting
        return words

    def _look_ahead(self, key, image) -> None:
        if key is None or not self._looking_ahead:
            self._incoming.clear()  # frames without keys can't be followed
        else:
            strips = []
            if self._unread:
                height, width = self._shape[:2]
                top_of_view, bottom_of_view = self._view
                for way, (top, bottom, left, right) in self._unread.items():
                    box = (left, top, min(right, width) - left, bottom - top, 'unread')
                    edge = top_of_view if way > 0 else bottom_of_view
                    strips.append((box, edge, way, self._strip_ink.get(way, [])))
            covered = [box for box in self._boxes if box[4] != 'unread']
            doubtful = [
                (x1, y1, x2 - x1, y2 - y1)
                for x1, y1, x2, y2, text in self._regions
                if _looks_secret(text) and not any(
                    x < x2 and x + w > x1 and y < y2 and y + h > y1 for x, y, w, h, _ in covered
                )
            ] if self._incoming.following or strips else []
            cut = None  # where the pane's edges cut lines through; unknown without a pane
            if self._pane is not None and (strips or self._incoming.following):
                top, bottom, left, right = self._pane
                left, right = max(left - self._scroll_edge, 0), right + self._scroll_edge
                rows_top = self._cut(image, top, 1, left, right)
                rows_bottom = self._cut(image, bottom, -1, left, right)
                width = right - left
                cut = [
                    (left, top, width, rows_top),
                    (left, bottom - rows_bottom, width, rows_bottom),
                ]
                cut = [rect for rect in cut if rect[3] > 0]
            pane = None
            if strips and self._pane is not None:
                top, bottom, left, right = self._pane
                left = max(left - self._scroll_edge, 0)
                right = min(right + self._scroll_edge, self._shape[1])
                pane = (left, top, right - left, bottom - top)
            self._incoming.record(
                key, list(self._boxes), strips, self._last_changed, self._step,
                self._shape[:2], doubtful, cut, pane,
            )

    def look_ahead(self) -> None:
        """Follow black incoming rows into later frames from now on, for settled()."""
        with self._lock:
            self._looking_ahead = True

    def settled(self, key) -> list[Box] | None:
        """The boxes for an earlier frame, with its incoming rows cleared where frames
        since have read them, or None if that frame is no longer known. Forgets it and
        every frame before it. The rows still unread stay as 'unread' boxes."""
        with self._lock:
            boxes = self._incoming.settled(key)
            for reason, rows in self._incoming.take_held_rows().items():
                self.stats[f'held_{reason}'] += rows
            for reason, rows in self._incoming.take_lost_rows().items():
                self.stats[f'lost_{reason}'] += rows
            return boxes

    def detect(self, frame, key=None) -> list[Box]:
        """Boxes to cover in this frame. Give each frame a key, in order, to have
        settled() clear its incoming rows later, once following frames have read them."""
        try:
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.shape[0] == 0
                or frame.shape[1] == 0
            ):
                raise ValueError('invalid frame')

            with self._lock:
                if frame.shape != self._shape:
                    self.reset()  # nothing known about the old size fits this one
                    self._shape = frame.shape

                # Save our own copy of this image.
                copy_started = time.monotonic()
                snapshot = np.array(frame, copy=True, order='C')
                diff_started = time.monotonic()
                self.stats['copy_seconds'] += diff_started - copy_started
                self._scrolled = False
                self._moved = False
                self._last_changed = self._step = None
                self._probe, self._uncovered = self._uncovered, None
                gap = 1 if key is None or self._previous_key is None else key - self._previous_key
                self._gap = min(max(gap, 1), _MOVE_GAP_MAX)
                areas = self._changed_areas(snapshot)
                self.stats['frames'] += 1
                self.stats['diff_seconds'] += time.monotonic() - diff_started
                self.reused_read = not areas
                self.stats['cached_frames'] += self.reused_read

                # Doubtful lines still waiting for a careful reread keep a still screen
                # scanning, cheaply, until each has had one.
                waiting = self._rescue is not None and bool(self._rescue_waiting)
                if not areas and not self._scrolled and not self._moved and not waiting:
                    self._look_ahead(key, snapshot)
                    return list(self._boxes)

                if self._reader is None:
                    from rapidocr_onnxruntime import RapidOCR

                    # The defaults enlarge thin rows, check for upside-down text and
                    # take every core, which starves OBS. Six threads scan as fast.
                    self._reader = RapidOCR(
                        det_limit_type='max', det_limit_side_len=1536, use_cls=False,
                        intra_op_num_threads=self._ocr_threads, inter_op_num_threads=1,
                    )

                started = time.monotonic()
                regions = [
                    region for region in self._regions
                    if not any(
                        region[1] < bottom and region[3] > top
                        and region[0] < right and region[2] > left
                        for top, bottom, left, right in areas
                    )
                ]

                kept = len(regions)
                for top, bottom, left, right in areas:
                    if self._cancelled.is_set():
                        raise RuntimeError('detection cancelled')
                    regions.extend(self._read(snapshot[top:bottom, left:right], top, left))
                fresh = {id(region) for region in regions[kept:]}

                height, width = snapshot.shape[:2]
                full = areas == [(0, height, 0, width)]
                if self._pane is not None and (
                    (full and not self._scrolled)
                    or (self._moved and _overlaps(self._moved_area, self._pane))
                ):
                    self._pane = None  # its window moved, or everything changed: stale edges
                if areas:
                    self.stats['full_scans' if full else 'partial_scans'] += 1
                self.stats['scan_seconds'] += time.monotonic() - started
                self.stats['scanned_share'] += sum(
                    (bottom - top) * (right - left) for top, bottom, left, right in areas
                ) / (height * width)
                boxes = set()

                find_started = time.monotonic()
                found = self._find_areas(regions, snapshot)
                if self._rescue is not None and self._doubtful and (fresh or self._rescue_waiting):
                    rescued = self._rescue_reads(
                        snapshot, fresh,
                        # Whole new screens frame after frame, as while apps switch, stall
                        # the same way. A single new screen is read carefully at once.
                        moving=(self._scrolled or self._moved or (full and self._was_full))
                        and not _RESCUE_WHILE_MOVING,
                    )
                    if rescued:
                        regions.extend(rescued)
                        self._careful_ids.update(id(word) for word in rescued)
                        found = self._find_areas(regions, snapshot)
                self._was_full = full
                # Forget rereads no longer on screen, so an id is never reused by mistake.
                on_screen = {id(region) for region in regions}
                self._careful_ids &= on_screen
                self._rescue_waiting &= on_screen
                self.stats['uncertain'] += self._uncertain
                track_started = time.monotonic()
                self.stats['find_seconds'] += track_started - find_started

                self._tracks.update(snapshot, [
                    Sighting(
                        category, rect, text_hash,
                        any(_overlaps((rect[1], rect[3], rect[0], rect[2]), a) for a in areas),
                        (category, rect, text_hash) in self._careful_areas,
                    )
                    for category, rect, text_hash in found
                    if category in self._enabled
                ], areas, self._pane)

                for category, (x1, y1, x2, y2) in self._tracks.covers():
                    left = max(0, min(width, math.floor(x1) - self.padding))
                    top = max(0, min(height - 1, math.floor(y1) - self.padding))
                    right = max(0, min(width, math.ceil(x2) + self.padding))
                    # A cover still in use keeps a sliver on screen, even if its box has
                    # just left, because OCR boxes can stop short of the last of the text.
                    bottom = max(top + 1, min(height, math.ceil(y2) + self.padding))

                    if right > left:
                        boxes.add((left, top, right - left, bottom - top, category))

                # Unread rows are covered where they show something, not their whole width.
                self._strip_ink = {
                    way: _ink_boxes(
                        snapshot, top, bottom, left, min(right, width), self._pixel_scale
                    )
                    for way, (top, bottom, left, right) in self._unread.items()
                }
                for ink in self._strip_ink.values():
                    boxes.update((x, y, w, h, 'unread') for x, y, w, h in ink)
                if self._rescue is not None and self._rescue_waiting:
                    boxes.update(self._waiting_covers(width, height))

                self.stats['track_seconds'] += time.monotonic() - track_started
                # Save results only after the whole scan succeeds.
                self._boxes = tuple(sorted(boxes))
                self._regions = regions
                self._remember(snapshot, areas)
                self._previous_key = key
                ahead_started = time.monotonic()
                self._look_ahead(key, snapshot)
                self.stats['ahead_seconds'] += time.monotonic() - ahead_started

                return list(self._boxes)

        except Exception:
            # Known secrets are still on screen, so a failed scan keeps their covers.
            with self._lock:
                self._forget_scan()

            raise DetectionError('Text detection failed; black out this frame.') from None


def apply_covers(frame, boxes: Iterable[Box]):
    """Make a copy of the image with solid black boxes over sensitive areas."""
    output = frame.copy()
    height, width = output.shape[:2]

    for x, y, w, h, _ in boxes:
        if w <= 0 or h <= 0:
            continue
        output[max(0, y):min(height, y + h), max(0, x):min(width, x + w)] = 0

    return output


_default = Detector()


def detect(frame) -> list[Box]:
    return _default.detect(frame)


def configure(keywords=(), enabled=None):
    _default.configure(keywords, enabled)


def reset():
    _default.reset()
