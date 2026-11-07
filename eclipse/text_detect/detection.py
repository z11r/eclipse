"""Find sensitive text in images and return boxes to cover.

Images must be uint8 BGR arrays. Boxes are (x, y, width, height, category).
If detection fails, the pipeline must black out the frame."""

import math
import re
import threading
import time
import unicodedata
from typing import Iterable

import cv2
import numpy as np

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


def find_matches(text: str, keywords: Iterable[str] = ()) -> list[tuple[str, int, int]]:
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


class Detector:
    """Read text from images and find sensitive information.

    Use one detector per screen or camera. Only rows that changed are read
    again; text read earlier keeps being reported until its pixels change, so a
    secret that sits still stays covered. Changing settings clears saved results."""

    def __init__(self, keywords=(), enabled=None, padding=0, reader=None, pixel_scale=1):
        self.stats = {
            'frames': 0, 'full_scans': 0, 'partial_scans': 0, 'scan_seconds': 0.0,
            'scanned_share': 0.0, 'diff_seconds': 0.0, 'find_seconds': 0.0,
            'det_seconds': 0.0, 'rec_seconds': 0.0, 'reads': 0, 'lines': 0, 'scrolls': 0,
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
        self._lock = threading.RLock()
        self.configure(keywords, enabled)

    def configure(self, keywords=(), enabled=None):
        words = tuple(keywords)
        categories = CATEGORIES if enabled is None else frozenset(enabled)

        if not all(isinstance(word, str) for word in words):
            raise ValueError('keywords must contain strings')

        if not categories <= CATEGORIES:
            raise ValueError('unknown detection category')

        with self._lock:
            self._keywords = words
            self._enabled = categories
            self.reset()

    def reset(self):
        """Forget the last image and boxes. The pipeline must clear its buffer too."""
        with self._lock:
            self._previous = None
            self._regions = []
            self._boxes = ()
            self._unread = None
            self._scrolled = False

    def _follow_scroll(self, image, changed):
        """If the change is a clean vertical scroll, move known text along with it.

        Returns what still needs reading: the rows newly on screen, plus anything
        that doesn't match once the shift is taken into account. If this isn't a
        clean scroll, returns the change untouched, so a wrong guess only costs time.
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
        now_rows = now.mean(axis=1)
        before_rows = before.mean(axis=1)
        limit = min(self._scroll_max_shift, height // 2)

        def difference(shift: int) -> float:
            start = max(shift, 0), max(-shift, 0)
            length = height - abs(shift)
            return float(np.abs(
                now_rows[start[0]:start[0] + length] - before_rows[start[1]:start[1] + length]
            ).mean())

        shift = min((s for s in range(-limit, limit + 1) if s), key=difference)

        if difference(shift) >= difference(0):
            return changed

        length = height - abs(shift)
        now_start, before_start = max(shift, 0), max(-shift, 0)
        # Half-pixel moves redraw text edges, so compare softened images.
        mismatch = np.abs(
            cv2.blur(now[now_start:now_start + length], (3, 3)).astype(np.int16)
            - cv2.blur(before[before_start:before_start + length], (3, 3)).astype(np.int16)
        ) > _SCROLL_TOLERANCE

        if mismatch.mean() > _SCROLL_MAX_MISMATCH:
            return changed

        remaining = np.zeros_like(changed)
        # Half-pixel moves leave thin specks on text edges; real changes are thicker.
        solid = cv2.erode(mismatch.astype(np.uint8), np.ones((3, 3), np.uint8))
        solid = cv2.dilate(solid, np.ones((5, 5), np.uint8)) > 0
        remaining[top + now_start:top + now_start + length, left:right] = solid
        moved = []

        for x1, y1, x2, y2, text in self._regions:
            inside = (
                x1 >= left - self._scroll_edge and x2 <= right + self._scroll_edge
                and y1 < bottom and y2 > top
            )
            if not inside:
                # Text cut by the edge of the scrolling area can't be trusted.
                if x1 < right and x2 > left and y1 < bottom and y2 > top:
                    remaining[max(int(y1), 0):int(y2) + 1, max(int(x1), 0):int(x2) + 1] = True
                    continue
                moved.append((x1, y1, x2, y2, text))
                continue
            new_top, new_bottom = max(y1 + shift, top), min(y2 + shift, bottom)
            if new_bottom > new_top:
                moved.append((x1, new_top, x2, new_bottom, text))

        self._regions = moved
        self.stats['scrolls'] += 1
        self._scrolled = True
        self._collect_unread(shift, top, bottom, left, right, now_start, length, remaining)
        return remaining

    def _collect_unread(self, shift, top, bottom, left, right, now_start, length, remaining):
        """Track rows new to the screen. They stay black until they can be read.

        A line coming onto the screen can't be read until enough of it shows, so
        rows wait here, covered, until there are enough to hold whole lines.
        """
        first = top + now_start
        strip = (top, first) if shift > 0 else (first + length, bottom)
        unread = (strip[0], strip[1], max(left - self._scroll_edge, 0), right + self._scroll_edge)

        if self._unread is not None:
            old_top, old_bottom, old_left, old_right = self._unread
            old_top, old_bottom = max(old_top + shift, top), min(old_bottom + shift, bottom)
            if old_bottom > old_top:
                unread = _join(unread, (old_top, old_bottom, old_left, old_right))

        self._unread = unread

        if unread[1] - unread[0] >= self._unread_read_at:
            remaining[unread[0]:unread[1], unread[2]:unread[3]] = True
            if shift > 0:
                self._unread = (unread[0], unread[0] + self._unread_keep, unread[2], unread[3])
            else:
                self._unread = (unread[1] - self._unread_keep, unread[1], unread[2], unread[3])

    def _read_unread(self, changed) -> None:
        """Scrolling has stopped, so read whatever is still covered."""
        if self._unread is not None and not self._scrolled:
            top, bottom, left, right = self._unread
            changed[top:bottom, left:right] = True
            self._unread = None

    def _changed_areas(self, image) -> list[tuple[int, int, int, int]]:
        """Areas to read again, each grown to hold whole pieces of known text."""
        height, width = image.shape[:2]

        if self._previous is None or self._previous.shape != image.shape:
            return [(0, height, 0, width)]

        difference = cv2.split(cv2.absdiff(image, self._previous))
        changed = cv2.bitwise_or(cv2.bitwise_or(difference[0], difference[1]), difference[2]) > 0
        changed = self._follow_scroll(image, changed)
        self._read_unread(changed)
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
            for x1, y1, x2, y2, _ in self._regions
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
        results, timings = self._reader(image)
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
        """Estimate where characters start to end sit inside a text region."""
        x1, y1, x2, y2, text = part
        width = (x2 - x1) / max(len(text), 1)
        # Characters vary in width, so allow extra rather than clip the secret.
        left = max(x1, x1 + (start - _CHAR_MARGIN) * width)
        right = min(x2, x1 + (end + _CHAR_MARGIN) * width)
        return left, y1, right, y2

    def _find_areas(self, regions) -> set[tuple]:
        areas = set()

        for region in regions:
            for category, start, end in find_matches(region[4], self._keywords):
                areas.add((category, self._cover(region, start, end)))

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
                            areas.add((category, self._cover(part, first, last)))

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
                # Save our own copy of this image.
                snapshot = np.array(frame, copy=True, order='C')
                diff_started = time.monotonic()
                self._scrolled = False
                areas = self._changed_areas(snapshot)
                self.stats['frames'] += 1
                self.stats['diff_seconds'] += time.monotonic() - diff_started

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

                for category, (x1, y1, x2, y2) in found:
                    if category not in self._enabled:
                        continue

                    left = max(0, min(width, math.floor(x1) - self.padding))
                    top = max(0, min(height, math.floor(y1) - self.padding))
                    right = max(0, min(width, math.ceil(x2) + self.padding))
                    bottom = max(0, min(height, math.ceil(y2) + self.padding))

                    if right > left and bottom > top:
                        boxes.add((left, top, right - left, bottom - top, category))

                if self._unread is not None:
                    top, bottom, left, right = self._unread
                    boxes.add((left, top, min(right, width) - left, bottom - top, 'unread'))

                # Save results only after the whole scan succeeds.
                self._boxes = tuple(sorted(boxes))
                self._regions = regions
                self._previous = snapshot

                return list(self._boxes)

        except Exception:
            self.reset()

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
