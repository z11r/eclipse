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
from eclipse.text_detect.incoming import IncomingRows, ScrollStep

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
_SCROLL_TRIES = 3  # parts of a change tried alone when the whole of it isn't a scroll
_NOISE_FLOOR = 8  # a pixel this close to where it came from hasn't changed
_CHANGE_BLOCK = 8  # with a sensor noise floor, compare blocks this size, not single pixels
_WHOLE_PIXEL = 0.05  # a shift this close to a whole number lands text on the same pixels
_SCROLL_RECHECK = 30  # frames of fractional scrolling between reads of the whole view
_UNREAD_READ_AT = 96  # rows of incoming text to collect before reading them
_UNREAD_KEEP = 24  # rows left black after a read, as the last line may be cut off
_CUT_REACH = 3  # a line cut by a pane edge is looked for this many ordinary lines deep
_LINE_GAP = 1  # blank rows, beyond this many, part one line from the next
_UNREAD_MIN_WAIT = 2  # frames between reads of rows that came in whole
_UNREAD_MIN_NEW = 8  # rows past the kept ones that a read after waiting needs
_INK_ROW_GAP = 4  # blank rows that still join one line's ink into one cover
_INK_WORD_GAP = 24  # blank columns that still join words; wider gaps part them
_INK_BACKGROUND = 0.5  # share of a strip one colour must fill to be its background
_STAYED_PUT_PIXELS = 40  # text pixels in uncovered rows needed to judge if they moved
_STAYED_PUT_SHARE = 0.98  # of them unchanged, for a drag over a still background
_MOVE_MIN_HEIGHT = 60  # a changed area smaller than this isn't checked for a moved window
_MOVE_MIN_WIDTH = 100
_MOVE_COARSE = 4  # the shift is first estimated on frames this many times smaller
_MOVE_SAMPLES = 4000  # textured pixels that vote on the exact shift
_MOVE_MIN_SHARE = 0.25  # share of changed textured pixels the shift must explain
_TEXTURE = 40  # a pixel this far from its neighbours' range is part of text or an edge


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
# A labeled phone remains private even when OCR misreads some area-code digits.
_PHONE_LABEL = re.compile(r'\b(?:phone|telephone|mobile|tel)\s*[:=]\s*([^\r\n:;]{7,40})', re.I)
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
    for match in _PHONE_LABEL.finditer(text):
        if sum(char.isdecimal() for char in match.group(1)) >= 7:
            place('phone', match.start(), match.end())
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
        match[t - top : b - top, l - left : r - left] = difference.max(axis=2) <= _NOISE_FLOOR
    return match


def _ink(area: np.ndarray) -> np.ndarray | None:
    """Which pixels of an area aren't its background colour, or None if no one colour
    fills most of it, so there is no telling background from text."""
    if area.size == 0:
        return np.zeros(area.shape[:2], bool)
    sample = area[::2, ::2].reshape(-1, 3).astype(np.int32)
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
        ocr_threads=6,
    ):
        self.stats = {
            'frames': 0, 'full_scans': 0, 'partial_scans': 0, 'scan_seconds': 0.0,
            'scanned_share': 0.0, 'diff_seconds': 0.0, 'find_seconds': 0.0,
            'det_seconds': 0.0, 'rec_seconds': 0.0, 'reads': 0, 'lines': 0, 'scrolls': 0,
            'checks': 0, 'copy_seconds': 0.0, 'track_seconds': 0.0, 'cached_frames': 0,
            'moves': 0,
        }
        if not isinstance(padding, int) or padding < 0:
            raise ValueError('padding must be a nonnegative integer')

        if not isinstance(noise_floor, int) or noise_floor < 0:
            raise ValueError('noise_floor must be a nonnegative integer')

        self.padding = padding
        self._reader = reader
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
        self._unread_read_at = _UNREAD_READ_AT * pixel_scale
        self._unread_keep = _UNREAD_KEEP * pixel_scale
        self._pixel_scale = pixel_scale
        self._scroll_block = _SCROLL_BLOCK * pixel_scale
        self._scroll_part_width = _SCROLL_PART_WIDTH * pixel_scale
        self._move_min_height = _MOVE_MIN_HEIGHT * pixel_scale
        self._move_min_width = _MOVE_MIN_WIDTH * pixel_scale
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
        self._uncovered = None  # rows a scroll uncovered last frame, to tell a drag apart
        self._pane = None  # the last scrolling pane's edges, which cut lines off where it stops
        self._incoming.clear()  # strips can only be followed through frames read in order
        self._last_changed = None  # this frame's changed pixels, while strips are followed
        self._step = None  # this frame's checked scroll, while strips are followed
        self._scrolled = False
        self._moved = False  # whether known text moved with a window this frame
        self._last_move = None  # the last window move, which the next one likely repeats
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

    def _scroll_shift(self, image, top, bottom, left, right, greys=None):
        """The whole-row shift that best explains how the area changed, with the grey
        pictures it was found on, or None when no shift explains it better than none.
        greys, if given, are those pictures already made for this area."""
        if greys is None:
            now = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            before = cv2.cvtColor(self._previous[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        else:
            now, before = greys
        now_rows = _row_means(now)
        before_rows = _row_means(before)
        limit = min(self._scroll_max_shift, (bottom - top) // 2)
        differences = _shift_differences(now_rows, before_rows, limit)
        shift = min(
            (s for s in range(-limit, limit + 1) if s), key=lambda s: differences[s + limit]
        )
        if differences[shift + limit] >= differences[limit]:
            return None
        return shift, now, before, now_rows, before_rows

    def _separate_changes(self, changed, top, bottom, left, right) -> list[tuple]:
        """Parts of the change that columns which didn't change keep apart, as (top,
        bottom, left, right), the most changed first. Empty if nothing stands apart."""
        counts = changed[top:bottom, left:right].sum(axis=0)
        columns = np.flatnonzero(counts)
        gap = 2 * _SCROLL_JOIN_ACROSS * self._scroll_block
        breaks = np.flatnonzero(np.diff(columns) > gap)
        if not breaks.size:
            return []
        parts = []
        for run in sorted(
            np.split(columns, breaks + 1), key=lambda run: -int(counts[run[0]:run[-1] + 1].sum())
        ):
            first, last = left + int(run[0]), left + int(run[-1]) + 1
            rows = np.flatnonzero(changed[top:bottom, first:last].any(axis=1))
            parts.append((top + int(rows[0]), top + int(rows[-1]) + 1, first, last))
        return parts

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
        # Compared at half size: each frame is shrunk after it's lined up by the shift, so
        # a whole-row scroll still matches exactly, and averaging pixels in twos softens
        # the redrawn text edges of a scroll that lands between pixels.
        length -= length % 2
        width = (right - left) // 2 * 2
        small = width // 2, length // 2
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
                difference, 0, down * block // 2 - small[1], 0, across * block // 2 - small[0],
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
        rows held black as they come in land on them instead of on the pane's edge. They
        can also drown the scroll out, so then each part of the change that stands apart
        is tried alone.
        """
        tries = [area]
        for index in range(1 + _SCROLL_TRIES):
            if index == 1:
                tries += self._separate_changes(changed, *area)[:_SCROLL_TRIES]
            if index >= len(tries):
                return None
            top, bottom, left, right = tries[index]
            if bottom - top < self._scroll_min_height or right - left < self._scroll_part_width:
                continue
            estimate = self._scroll_shift(image, top, bottom, left, right)
            if estimate is None:
                continue
            part = self._scrolling_part(changed, estimate, top, bottom, left, right)
            if part is None:
                continue
            if part == tries[index]:
                return part, estimate
            if (
                part[1] - part[0] < self._scroll_min_height
                or part[3] - part[2] < self._scroll_part_width
            ):
                continue
            # The shift again, on the pane alone: what changed beside it can pull the first
            # estimate off, and covers moved by a wrong shift would leave their text.
            inside = slice(part[0] - top, part[1] - top), slice(part[2] - left, part[3] - left)
            estimate = self._scroll_shift(
                image, *part, greys=(estimate[1][inside], estimate[2][inside])
            )
            if estimate is not None:
                return part, estimate
        return None

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

        uncovered, self._uncovered = self._uncovered, None
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
                    moved.append((x1, new_top, x2, new_bottom, text))

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
        before = cv2.cvtColor(self._previous[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        mask = changed[top:bottom, left:right]
        size = (max((right - left) // _MOVE_COARSE, 8), max((bottom - top) // _MOVE_COARSE, 8))
        small = []
        for gray in (before, now):
            values = np.where(mask, gray.astype(np.float32) - float(gray[mask].mean()), 0)
            small.append(cv2.resize(values.astype(np.float32), size, interpolation=cv2.INTER_AREA))
        window = cv2.createHanningWindow(size, cv2.CV_32F)
        (sx, sy), _ = cv2.phaseCorrelate(small[0], small[1], window)
        guess = (round(sx * _MOVE_COARSE), round(sy * _MOVE_COARSE))

        textured = _texture(now) & mask
        ys, xs = np.nonzero(textured)
        if ys.size < _MOVE_SAMPLES // 8:
            return None
        if ys.size > _MOVE_SAMPLES:
            pick = np.linspace(0, ys.size - 1, _MOVE_SAMPLES).astype(int)
            ys, xs = ys[pick], xs[pick]
        ys, xs = ys + top, xs + left
        height, width = image.shape[:2]
        candidates = {
            (guess[0] + ex, guess[1] + ey)
            for ex in range(-_MOVE_COARSE, _MOVE_COARSE + 1)
            for ey in range(-_MOVE_COARSE, _MOVE_COARSE + 1)
        }
        if self._last_move is not None:
            candidates.add(self._last_move)
        best, best_share = None, 0.0
        for dx, dy in candidates:
            if (dx, dy) == (0, 0):
                continue
            sy_, sx_ = ys - dy, xs - dx
            inside = (sy_ >= 0) & (sy_ < height) & (sx_ >= 0) & (sx_ < width)
            if not inside.any():
                continue
            difference = np.abs(
                image[ys[inside], xs[inside]].astype(np.int16)
                - self._previous[sy_[inside], sx_[inside]].astype(np.int16)
            ).max(axis=1)
            share = float((difference <= _NOISE_FLOOR).sum()) / ys.size
            if share > best_share:
                best, best_share = (dx, dy), share
        if best is None or best_share < _MOVE_MIN_SHARE:
            return None
        return best

    def _follow_move(self, image, changed):
        """If a window moved whole, move known text and covers along with it.

        Something moved whole when its pixels now show exactly what the previous frame
        showed one fixed dx, dy away. Known text moves only if every pixel of its new
        place matches; anything else is read again, as is every pixel the move doesn't
        explain, so a wrong shift only costs time. Returns what still needs reading.
        """
        rows = np.flatnonzero(changed.any(axis=1))
        if not rows.size:
            return changed
        columns = np.flatnonzero(changed.any(axis=0))
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(columns[0]), int(columns[-1]) + 1
        if bottom - top < self._move_min_height or right - left < self._move_min_width:
            return changed

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
        match = np.zeros_like(changed)
        match[area[0]:area[1], area[2]:area[3]] = _shifted_match(
            image, self._previous, dx, dy, area
        )
        explained = changed & match
        if not explained.any():
            return changed
        self._last_move = shift
        self._moved = True
        self._moved_area = area  # top, bottom, left, right of what moved, and where to
        self.stats['moves'] += 1
        remaining = changed & ~match

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
                moved.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, text))
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

    def _changed_areas(self, image) -> list[tuple[int, int, int, int]]:
        """Areas to read again, each grown to hold whole pieces of known text."""
        height, width = image.shape[:2]

        if self._previous is None or self._previous.shape != image.shape:
            return [(0, height, 0, width)]

        if self._noise_floor:
            changed = self._changed_blocks(image)
        else:
            difference = cv2.split(cv2.absdiff(image, self._previous))
            changed = cv2.bitwise_or(
                cv2.bitwise_or(difference[0], difference[1]), difference[2]
            ) > 0
        if self._incoming.following:
            self._last_changed = changed.copy()  # the steps below mark it up
        changed = self._follow_scroll(image, changed)
        if not self._scrolled and not self._noise_floor:
            # A camera's noise never matches exactly, and its pictures aren't windows.
            changed = self._follow_move(image, changed)
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
            if self._pane is not None:
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
            return self._incoming.settled(key)

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
                areas = self._changed_areas(snapshot)
                self.stats['frames'] += 1
                self.stats['diff_seconds'] += time.monotonic() - diff_started
                self.reused_read = not areas
                self.stats['cached_frames'] += self.reused_read

                if not areas and not self._scrolled and not self._moved:
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

                for top, bottom, left, right in areas:
                    if self._cancelled.is_set():
                        raise RuntimeError('detection cancelled')
                    regions.extend(self._read(snapshot[top:bottom, left:right], top, left))

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
                found = self._find_areas(regions)
                track_started = time.monotonic()
                self.stats['find_seconds'] += track_started - find_started

                self._tracks.update(snapshot, [
                    Sighting(
                        category, rect, text_hash,
                        any(_overlaps((rect[1], rect[3], rect[0], rect[2]), a) for a in areas),
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

                self.stats['track_seconds'] += time.monotonic() - track_started
                # Save results only after the whole scan succeeds.
                self._boxes = tuple(sorted(boxes))
                self._regions = regions
                self._remember(snapshot, areas)
                self._look_ahead(key, snapshot)

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
