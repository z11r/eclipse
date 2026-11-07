"""Find sensitive text in images and return boxes to cover.

Images must be uint8 BGR arrays. Boxes are (x, y, width, height, category).
If detection fails, the pipeline must black out the frame."""

import functools
import math
import re
import threading
import time
import unicodedata
from typing import Iterable

import cv2
import numpy as np

from eclipse.text_detect.covers import MOVE, REREAD, STAY, CoverTracks, Sighting

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
_NOISE_FLOOR = 8  # a pixel this close to where it came from hasn't changed
_WHOLE_PIXEL = 0.05  # a shift this close to a whole number lands text on the same pixels
_SCROLL_RECHECK = 30  # frames of fractional scrolling between reads of the whole view
_UNREAD_READ_AT = 96  # rows of incoming text to collect before reading them
_UNREAD_KEEP = 24  # rows left black after a read, as the last line may be cut off


class DetectionError(RuntimeError):
    """Detection failed. The pipeline must black out this frame."""


# OCR can read @ as a copyright sign, and the last dot as a comma or a space. Only
# common endings count without a dot, so "@alice hello" isn't an email.
_EMAIL = re.compile(
    r'[A-Z0-9._%+\-]+\s*[@©]\s*[A-Z0-9.\-]+'
    r'(?:\s*\.\s*[A-Z]{2,}|[\s,]+(?:com|org|net|edu|gov|io|ca|co|dev|ai)\b)',
    re.I,
)
_PHONE = re.compile(
    r'(?<!\w)(?:\+\d{1,3}[ .\-]?)?(?:\(\d{3}\)|\d{3})[ .\-]?\d{3}[ .\-]?\d{4}(?!\w)'
)
# A bare 6-8 digit number can be a code. 4-5 digits are too common (ports, years),
# so they only count next to a word like "code".
_CODE_ALONE = re.compile(r'(?<![\w.:/\-])(?:\d{6,8}|\d{3}[ \-]\d{3})(?![\w.:/\-])')
_CODE_HINT = re.compile(r'\b(?:code|otp|verification|verify|2fa|passcode|pin)\b', re.I)
_CODE = re.compile(r'(?<!\w)\d(?:[ \-]?\d){3,7}(?!\w)')
_CARD = re.compile(r'(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)')
_KEYS = [
    re.compile(r'\b(?:sk|rk)-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{16,}\b'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    re.compile(r'\b(?:AKIA|ASIA)\s?[A-Z0-9]{16}\b'),
    re.compile(r'\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b'),
    re.compile(r'\bxox[abeoprs]-[A-Za-z0-9\-]{10,}'),
    re.compile(r'\bAIza[A-Za-z0-9_\-]{30,}'),
    re.compile(r'\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b'),
    # Any name ending in one of these, so DB_PASSWORD and AWS_SECRET_ACCESS_KEY count.
    re.compile(
        r'(?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]*'
        r'(?:api[_ -]?key|access[_ -]?key|access[_ -]?token|token|secret(?:[_ -]?key)?'
        r'|private[_ -]?key|passw(?:or)?d|pwd|authorization)'
        r'\b\s*[=:]\s*[\"\x27]?(?:Bearer\s+)?[^\s\"\x27,;]{6,}',
        re.I,
    ),
    re.compile(r'\bBearer\s+[A-Za-z0-9._~+/=\-]{12,}', re.I),
    re.compile(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])'),
    # A password inside a link: postgres://user:password@host
    re.compile(r'\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s/@]+@', re.I),
    # A private key: its header, then each line of its body.
    re.compile(r'-{2,}\s*BEGIN [A-Z ]*PRIVATE KEY'),
    re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/])'),
]
# Characters OCR tends to swap, for matching keywords.
_LOOKALIKES = {
    '0': 'o', 'o': '0', '1': 'il|', 'i': '1l|', 'l': '1i|',
    '2': 'z', 'z': '2', '5': 's', 's': '5', '8': 'b', 'b': '8',
}
_SHORT_KEYWORD = 4  # below this many characters, match whole words only


def luhn_valid(value: str) -> bool:
    digits = [int(c) for c in value if c.isascii() and c.isdigit()]

    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False

    total = 0

    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def _keyword_pattern(word: str) -> re.Pattern:
    """Match a keyword through spacing changes and common OCR misreads.

    OCR can drop spaces or add them inside a word, so any gap is allowed anywhere.
    """
    letters = ''.join(unicodedata.normalize('NFKC', word).lower().split())
    pattern = r'\s*'.join(
        '[' + re.escape(char + _LOOKALIKES.get(char, '')) + ']' for char in letters
    )

    if len(letters) < _SHORT_KEYWORD:
        pattern = rf'(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])'

    return re.compile(pattern, re.I)


def _standard_characters(text: str) -> tuple[str, list[int] | None]:
    """Convert look-alike characters, such as full-width brackets, to standard ones.

    One character can become several, so this also returns where each one came from.
    Covers need positions in the text OCR gave us, not in the converted text.
    """
    if text.isascii():
        return text, None

    converted = []
    origins = []

    for index, char in enumerate(text):
        standard = unicodedata.normalize('NFKC', char)
        converted.append(standard)
        origins.extend([index] * len(standard))

    return ''.join(converted), origins


def _match_text(text: str, keywords: Iterable[str] = ()) -> list[tuple[str, int, int]]:
    """Return (category, start, end) for each sensitive part of the text."""
    text, origins = _standard_characters(text)
    matches = []

    def place(category, start, end):
        if origins is None:
            matches.append((category, start, end))
        elif end > start:
            matches.append((category, origins[start], origins[end - 1] + 1))

    def add(category, pattern):
        for match in pattern.finditer(text):
            place(category, match.start(), match.end())

    add('email', _EMAIL)
    add('phone', _PHONE)
    add('2fa', _CODE_ALONE)

    if _CODE_HINT.search(text):
        add('2fa', _CODE)

    for pattern in _KEYS:
        add('api_key', pattern)

    for match in _CARD.finditer(text):
        if luhn_valid(match.group()):
            place('card', match.start(), match.end())

    for word in keywords:
        if word.strip():
            add('keyword', _keyword_pattern(word))

    return matches


@functools.lru_cache(maxsize=16384)
def _remembered_matches(text: str, keywords: tuple[str, ...]) -> tuple:
    return tuple(_match_text(text, keywords))


def find_matches(text: str, keywords: Iterable[str] = ()) -> list[tuple[str, int, int]]:
    """Return (category, start, end) for each sensitive part of the text.

    Answers are remembered. A frame re-checks every word on screen, nearly all of them
    unchanged, and checking them afresh took 2 ms a frame. The answer depends only on
    the text and the keywords, so a remembered one is always the same answer.
    """
    return list(_remembered_matches(text, tuple(keywords)))


def _row_means(gray) -> np.ndarray:
    """Each row's mean, exactly as gray.mean(axis=1) gives it, in a tenth of the time."""
    # The sums are whole numbers well under 2**53, so float64 holds them exactly, and
    # dividing exact sums gives the same values numpy's mean does.
    return cv2.reduce(gray, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F)[:, 0] / gray.shape[1]


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


def _text_hash(text: str) -> int:
    """Tells text apart across reads, so tracks never have to keep the text itself."""
    return hash(''.join(text.split()).lower())


class Detector:
    """Read text from images and find sensitive information.

    Use one detector per screen or camera. Only rows that changed are read
    again, and text read earlier is kept until its pixels change. A secret's cover
    lives in CoverTracks, so one read missing it doesn't uncover it. Each result is
    everything to cover in that frame. Changing settings clears saved results."""

    def __init__(self, keywords=(), enabled=None, padding=0, reader=None, pixel_scale=1):
        self.stats = {
            'frames': 0, 'full_scans': 0, 'partial_scans': 0, 'scan_seconds': 0.0,
            'scanned_share': 0.0, 'diff_seconds': 0.0, 'find_seconds': 0.0,
            'det_seconds': 0.0, 'rec_seconds': 0.0, 'reads': 0, 'lines': 0, 'scrolls': 0,
            'checks': 0,
        }
        if not isinstance(padding, int) or padding < 0:
            raise ValueError('padding must be a nonnegative integer')

        self.padding = padding
        self._reader = reader
        # Distances below are for standard frames. Retina frames have twice the pixels.
        self._band_margin = _BAND_MARGIN * pixel_scale
        self._side_margin = _SIDE_MARGIN * pixel_scale
        self._scroll_min_height = _SCROLL_MIN_HEIGHT * pixel_scale
        self._scroll_min_width = _SCROLL_MIN_WIDTH * pixel_scale
        self._scroll_max_shift = _SCROLL_MAX_SHIFT * pixel_scale
        self._scroll_edge = _SCROLL_EDGE * pixel_scale
        self._unread_read_at = _UNREAD_READ_AT * pixel_scale
        self._unread_keep = _UNREAD_KEEP * pixel_scale
        self._tracks = CoverTracks(pixel_scale)
        self._shape = None
        self._lock = threading.RLock()
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
        self._scrolled = False
        self._view = None  # top and bottom of the area scrolling now
        self._fractional_frames = 0  # frames of this scroll that landed between pixels

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

    def _follow_scroll(self, image, changed):
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

        now = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        before = cv2.cvtColor(self._previous[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        now_rows = _row_means(now)
        before_rows = _row_means(before)
        limit = min(self._scroll_max_shift, height // 2)
        differences = _shift_differences(now_rows, before_rows, limit)

        def difference(shift: int) -> float:
            return float(differences[shift + limit])

        shift = min((s for s in range(-limit, limit + 1) if s), key=difference)

        if difference(shift) >= difference(0):
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
        remaining = np.zeros_like(changed)
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
        exact_shift = shift + fraction

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
                    moved.append((x1, new_top, x2, new_bottom, text))

        for place in self._tracks.follow_scroll(
            action, exact_shift, top_of_view, bottom_of_view, image
        ):
            remaining[_inside(place)] = True

        self._regions = moved
        self.stats['scrolls'] += 1
        self._scrolled = True
        self._collect_unread(shift, left, right, first_row, length, remaining)
        return remaining

    def _collect_unread(self, shift, left, right, first, length, remaining):
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

        if incoming[1] - incoming[0] >= self._unread_read_at:
            remaining[incoming[0]:incoming[1], incoming[2]:incoming[3]] = True
            if shift > 0:
                incoming = (incoming[0], incoming[0] + self._unread_keep, *incoming[2:])
            else:
                incoming = (incoming[1] - self._unread_keep, incoming[1], *incoming[2:])

        strips[direction] = incoming
        self._unread = strips

    def _read_unread(self, changed) -> None:
        """Scrolling has stopped, so read whatever is still covered."""
        if not self._scrolled:
            for top, bottom, left, right in self._unread.values():
                changed[top:bottom, left:right] = True
            self._unread = {}
            self._view = None
            self._fractional_frames = 0

    def _changed_areas(self, image) -> list[tuple[int, int, int, int]]:
        """Areas to read again, each grown to hold whole pieces of known text."""
        height, width = image.shape[:2]

        if self._previous is None or self._previous.shape != image.shape:
            return [(0, height, 0, width)]

        difference = cv2.split(cv2.absdiff(image, self._previous))
        changed = cv2.bitwise_or(cv2.bitwise_or(difference[0], difference[1]), difference[2]) > 0
        changed = self._follow_scroll(image, changed)
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

        scanned = sum((bottom - top) * (right - left) for top, bottom, left, right in areas)
        if scanned > _FULL_SCAN_SHARE * height * width:
            return [(0, height, 0, width)]

        return areas

    def _read(self, image, top: int, left: int) -> list[tuple]:
        """Read one area and return text regions in full-frame coordinates."""
        read_area = getattr(self._reader, 'read_area', None)
        results, timings = read_area(image, top, left) if read_area else self._reader(image)
        self.stats['reads'] += 1
        self.stats['lines'] += len(results or [])
        if isinstance(timings, (list, tuple)) and len(timings) == 3:
            self.stats['det_seconds'] += timings[0] or 0
            self.stats['rec_seconds'] += timings[2] or 0
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

    def _groups(self, regions) -> list[list[tuple]]:
        """Group neighbouring text on the same row, to catch split emails and keys."""
        rows = []

        for region in sorted(regions, key=lambda r: (r[1], r[0])):
            _, y1, _, y2, _ = region

            for row in rows:
                overlap = min(row[1], y2) - max(row[0], y1)
                if overlap >= 0.5 * min(row[1] - row[0], y2 - y1):
                    row[0] = min(row[0], y1)
                    row[1] = max(row[1], y2)
                    row[2].append(region)
                    break
            else:
                rows.append([y1, y2, [region]])

        groups = []

        for _, _, parts in rows:
            parts.sort(key=lambda r: r[0])
            group = [parts[0]]

            for part in parts[1:]:
                previous = group[-1]
                if part[0] - previous[2] <= 2 * (previous[3] - previous[1]):
                    group.append(part)
                else:
                    groups.append(group)
                    group = [part]

            groups.append(group)

        return [group for group in groups if len(group) > 1]

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

    def _find_areas(self, regions) -> set[tuple]:
        areas = set()

        for region in regions:
            for category, start, end in find_matches(region[4], self._keywords):
                areas.add((
                    category, self._cover(region, start, end), _text_hash(region[4][start:end]),
                ))

        # Text split across regions, tried with and without spaces.
        for group in self._groups(regions):
            for separator in (' ', ''):
                joined = separator.join(part[4] for part in group)

                for category, start, end in find_matches(joined, self._keywords):
                    position = 0
                    touched = []

                    for part in group:
                        first = max(start - position, 0)
                        last = min(end - position, len(part[4]))
                        if last > first:
                            touched.append((part, first, last))
                        position += len(part[4]) + len(separator)

                    if len(touched) > 1:
                        for part, first, last in touched:
                            areas.add((
                                category, self._cover(part, first, last),
                                _text_hash(joined[start:end]),
                            ))

        return areas

    def detect(self, frame) -> list[Box]:
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
                snapshot = np.array(frame, copy=True, order='C')
                diff_started = time.monotonic()
                self._scrolled = False
                areas = self._changed_areas(snapshot)
                self.stats['frames'] += 1
                self.stats['diff_seconds'] += time.monotonic() - diff_started
                self.reused_read = not areas

                if not areas and not self._scrolled:
                    return list(self._boxes)

                if self._reader is None:
                    from rapidocr_onnxruntime import RapidOCR

                    # The defaults enlarge thin rows, check for upside-down text and
                    # take every core, which starves OBS. Six threads scan as fast.
                    self._reader = RapidOCR(
                        det_limit_type='max', det_limit_side_len=1536, use_cls=False,
                        intra_op_num_threads=6, inter_op_num_threads=1,
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

                for top, bottom, left, right in areas:
                    regions.extend(self._read(snapshot[top:bottom, left:right], top, left))

                height, width = snapshot.shape[:2]
                full = areas == [(0, height, 0, width)]
                if areas:
                    self.stats['full_scans' if full else 'partial_scans'] += 1
                self.stats['scan_seconds'] += time.monotonic() - started
                self.stats['scanned_share'] += sum(
                    (bottom - top) * (right - left) for top, bottom, left, right in areas
                ) / (height * width)
                boxes = set()

                find_started = time.monotonic()
                found = self._find_areas(regions)
                self.stats['find_seconds'] += time.monotonic() - find_started

                self._tracks.update(snapshot, [
                    Sighting(
                        category, rect, text_hash,
                        any(_overlaps((rect[1], rect[3], rect[0], rect[2]), a) for a in areas),
                    )
                    for category, rect, text_hash in found
                    if category in self._enabled
                ], areas)

                for category, (x1, y1, x2, y2) in self._tracks.covers():
                    left = max(0, min(width, math.floor(x1) - self.padding))
                    top = max(0, min(height - 1, math.floor(y1) - self.padding))
                    right = max(0, min(width, math.ceil(x2) + self.padding))
                    # A cover still in use keeps a sliver on screen, even if its box has
                    # just left, because OCR boxes can stop short of the last of the text.
                    bottom = max(top + 1, min(height, math.ceil(y2) + self.padding))

                    if right > left:
                        boxes.add((left, top, right - left, bottom - top, category))

                for top, bottom, left, right in self._unread.values():
                    boxes.add((left, top, min(right, width) - left, bottom - top, 'unread'))

                # Save results only after the whole scan succeeds.
                self._boxes = tuple(sorted(boxes))
                self._regions = regions
                self._previous = snapshot

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
