"""Find sensitive text in images and return boxes to cover.

Images must be uint8 BGR arrays. Boxes are (x, y, width, height, category).
If detection fails, the pipeline must black out the frame."""

import math
import re
import threading
import unicodedata
from typing import Iterable

Box = tuple[int, int, int, int, str]
CATEGORIES = frozenset({'email', 'phone', '2fa', 'api_key', 'card', 'keyword'})


class DetectionError(RuntimeError):
    """Detection failed. The pipeline must black out this frame."""


_EMAIL = re.compile(r'[A-Z0-9._%+\-]+\s*@\s*[A-Z0-9.\-]+\s*\.\s*[A-Z]{2,}', re.I)
_PHONE = re.compile(
    r'(?<!\w)(?:\+\d{1,3}[ .\-]?)?(?:\(\d{3}\)|\d{3})[ .\-]?\d{3}[ .\-]?\d{4}(?!\w)'
)
# Broad on purpose: standalone 4-8 digit numbers can be authentication codes.
_CODE = re.compile(r'(?<!\w)\d(?:[ \-]?\d){3,7}(?!\w)')
_CARD = re.compile(r'(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)')
_KEYS = [
    re.compile(r'\b(?:sk|rk)-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{16,}\b'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b'),
    re.compile(
        r'\b(?:api[_ -]?key|access[_ -]?token|secret|password|authorization)\b\s*[=:]\s*[\"\x27]?(?:Bearer\s+)?[^\s\"\x27,;]{6,}', re.I
    ),
    re.compile(r'\bBearer\s+[A-Za-z0-9._~+/=\-]{12,}', re.I),
    re.compile(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])'),
]


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


def classify_text(text: str, keywords: Iterable[str] = ()) -> set[str]:
    """Return the types of sensitive text found, without the secret text."""
    # Convert alternate characters, such as full-width parentheses, to standard ones.
    text = unicodedata.normalize('NFKC', text)
    found = set()

    if _EMAIL.search(text):
        found.add('email')

    if _PHONE.search(text):
        found.add('phone')

    if _CODE.search(text):
        found.add('2fa')

    if any(pattern.search(text) for pattern in _KEYS):
        found.add('api_key')

    if any(luhn_valid(match.group()) for match in _CARD.finditer(text)):
        found.add('card')

    folded = text.casefold()
    # OCR sometimes drops or adds spaces inside a phrase.
    compact_text = ''.join(folded.split())

    for word in keywords:
        keyword = unicodedata.normalize('NFKC', word).strip().casefold()
        compact_keyword = ''.join(keyword.split())

        if compact_keyword and (
            keyword in folded or compact_keyword in compact_text
        ):
            found.add('keyword')
            break

    return found


class Detector:
    """Read text from images and find sensitive information.

    Use one detector per screen or camera. If the image is unchanged,
    reuse the previous boxes. Changing settings clears the saved results."""

    def __init__(self, keywords=(), enabled=None, padding=10, reader=None):
        if not isinstance(padding, int) or padding < 0:
            raise ValueError('padding must be a nonnegative integer')

        self.padding = padding
        self._reader = reader
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
            self._boxes = ()

    def detect(self, frame) -> list[Box]:
        try:
            import numpy as np

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

                if self._previous is not None and np.array_equal(snapshot, self._previous):
                    return list(self._boxes)


                if self._reader is None:
                    from rapidocr_onnxruntime import RapidOCR

                    self._reader = RapidOCR()

                # This is where OCR reads the image.
                results, _ = self._reader(snapshot)

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
                        float(points[:, 0].min()),
                        float(points[:, 1].min()),
                        float(points[:, 0].max()),
                        float(points[:, 1].max()),
                        text,
                    ))

                # Group text on the same row to catch split emails and keys.
                rows = []

                for region in sorted(regions, key=lambda r: (r[1], r[0])):
                    x1, y1, x2, y2, _ = region

                    for row in rows:
                        overlap = min(row[3], y2) - max(row[1], y1)
                        if overlap >= 0.5 * min(row[3] - row[1], y2 - y1):
                            row[0] = min(row[0], x1)
                            row[1] = min(row[1], y1)
                            row[2] = max(row[2], x2)
                            row[3] = max(row[3], y2)
                            row[4].append(region)
                            break
                    else:
                        rows.append([x1, y1, x2, y2, [region]])

                candidates = regions.copy()

                for x1, y1, x2, y2, parts in rows:
                    if len(parts) > 1:
                        parts.sort(key=lambda r: r[0])

                        # Try joining the text with and without spaces.
                        for separator in (' ', ''):
                            joined_text = separator.join(part[4] for part in parts)
                            candidates.append((x1, y1, x2, y2, joined_text))

                height, width = snapshot.shape[:2]
                boxes = set()

                for x1, y1, x2, y2, text in candidates:
                    categories = classify_text(text, self._keywords) & self._enabled

                    left = max(0, min(width, math.floor(x1) - self.padding))
                    top = max(0, min(height, math.floor(y1) - self.padding))
                    right = max(0, min(width, math.ceil(x2) + self.padding))
                    bottom = max(0, min(height, math.ceil(y2) + self.padding))

                    for category in categories:
                        if right > left and bottom > top:
                            boxes.add((left, top, right - left, bottom - top, category))

                # Save results only after the whole scan succeeds.
                self._boxes = tuple(sorted(boxes))
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
