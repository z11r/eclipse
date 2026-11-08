"""Keeps one cover on each secret, from its first sighting until it is really gone.

OCR changes its mind between reads: it can miss a line it read a moment ago, and its
boxes wobble by a few pixels. A cover can't do either, so each secret gets a track.
A track holds the secret's current place and moves with the text, so it leaves no
trail, and a read that misses it doesn't end it. Only these things do:

- its pixels changed, and reads of the new pixels found nothing there
- it left the frame or the scrolling area
- the same text turned up somewhere else, and its old place no longer shows it
- the detector was reset, when the frame size or source changes
- its category was turned off

Time alone never ends a track. Tracks keep a hash of the text, never the text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

Rect = tuple[float, float, float, float]  # left, top, right, bottom

STAY, MOVE, REREAD = "stay", "move", "reread"  # what a scroll does to a place on screen

_KEEP_READS = 3  # a cover holds this many recent reads, so their wobble doesn't show
_WOBBLE = 3  # an OCR box can shift this far between reads of text that didn't move
_MIN_OVERLAP = 0.3  # share of the smaller box that makes two boxes the same place
_CLEAR_READS = 2  # reads that must find nothing on new pixels before a cover comes off
_RECHECK_FRAMES = 15  # how often to retry a secret OCR lost on pixels that didn't change
_SAME_TOLERANCE = 40  # blurred pixels this close count as the same
_SAME_MAX_MISMATCH = 0.03  # past this share of pixels, the content has changed
_UNSURE_MARGIN = 24  # extra cover while a secret's pixels changed and it wasn't found
_FOLLOW_REACH = 64  # most pixels a camera cover's text is looked for from where it was
_FOLLOW_MIN_SIDE = 6  # text this small has too few pixels to be found again
_LEAD_FRAMES = 4  # frames a lost camera cover keeps stretching along its last move


@dataclass(frozen=True)
class Sighting:
    """One match in this frame's text."""

    category: str
    rect: Rect
    text_hash: int
    fresh: bool  # read from this frame's pixels, not carried over from an earlier read


@dataclass
class _Track:
    category: str
    text_hash: int
    reads: list[Rect]  # oldest first, all moved along with the text
    pixels: np.ndarray  # greyscale, from when the text was last read
    pixels_at: tuple[int, float]  # left and top of where those pixels should be now
    clear_reads: int = 0  # reads in a row that found nothing on changed pixels
    unsure: bool = False  # its pixels changed and nothing has explained why
    check_at: int | None = None  # frame to read this place again, if it needs it
    followed: bool = False  # moved by a checked scroll or window move since it was last read
    velocity: tuple[int, int] = (0, 0)  # its last followed dx, dy, where to look first next
    lost: int = 0  # frames since following last found its text, while it was moving
    lead: Rect | None = None  # where moving text that following lost may have gone

    @property
    def cover(self) -> Rect:
        return _union([*self.reads, self.lead] if self.lead else self.reads)

    def move(self, shift: float, sideways: int = 0) -> None:
        """Slide with a scroll or a moved window. Never trimmed to the scrolling area, whose
        edges are only an estimate: a bad one would cut a cover short for good."""
        self.reads = [
            (x1 + sideways, y1 + shift, x2 + sideways, y2 + shift)
            for x1, y1, x2, y2 in self.reads
        ]
        self.pixels_at = self.pixels_at[0] + sideways, self.pixels_at[1] + shift
        self.followed = True
        self.lead = None

    def touches(self, top: int, bottom: int) -> bool:
        return any(y1 < bottom and y2 > top for _, y1, _, y2 in self.reads)


def _union(rects: Iterable[Rect]) -> Rect:
    lefts, tops, rights, bottoms = zip(*rects)
    return min(lefts), min(tops), max(rights), max(bottoms)


def _overlap(rect: Rect, other: Rect) -> float:
    """How much two rects overlap, as a share of the smaller one."""
    width = min(rect[2], other[2]) - max(rect[0], other[0])
    height = min(rect[3], other[3]) - max(rect[1], other[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min(
        (rect[2] - rect[0]) * (rect[3] - rect[1]), (other[2] - other[0]) * (other[3] - other[1])
    )
    return width * height / max(smaller, 1)


def _contains(rect: Rect, other: Rect) -> bool:
    return (
        rect[0] <= other[0] and rect[1] <= other[1] and rect[2] >= other[2] and rect[3] >= other[3]
    )


def _moved(before: Rect, after: Rect, wobble: float) -> bool:
    """Whether a box really moved: both edges went the same way, past OCR's wobble."""
    for near, far in ((0, 2), (1, 3)):
        first, second = after[near] - before[near], after[far] - before[far]
        if (first > wobble and second > wobble) or (first < -wobble and second < -wobble):
            return True
    return False


class CoverTracks:
    """One detector's tracks. The detector calls these once a frame, in this order:
    due, then follow_scroll on a scroll or follow_move on a moved window (follow_each on
    a camera), then update once the scan has finished."""

    def __init__(self, pixel_scale: int = 1) -> None:
        self._wobble = _WOBBLE * pixel_scale
        self._margin = _UNSURE_MARGIN * pixel_scale
        self._reach = _FOLLOW_REACH * pixel_scale
        self.clear()

    def clear(self) -> None:
        self._tracks: list[_Track] = []
        self._now = 0

    def keep_only(self, categories: Iterable[str]) -> None:
        """Drop every cover outside these categories, for when settings change."""
        wanted = set(categories)
        self._tracks = [track for track in self._tracks if track.category in wanted]

    def places(self) -> list[Rect]:
        """Where each cover is now, so a read of nearby text can take it in whole."""
        return [track.cover for track in self._tracks]

    def due(self) -> list[Rect]:
        """Places to read again this frame, whether their pixels changed or not."""
        self._now += 1
        return [
            track.cover
            for track in self._tracks
            if track.check_at is not None and track.check_at <= self._now
        ]

    def follow_scroll(
        self, action: Callable[[Rect], str], shift: float, top: int, bottom: int, image: np.ndarray
    ) -> list[Rect]:
        """Move tracks with a checked scroll. Returns places that must be read again."""
        unsure = []
        kept = []
        for track in self._tracks:
            todo = action(track.cover)
            if todo == MOVE:
                track.move(shift)
                if not track.touches(top, bottom) and not self._unchanged(track, image):
                    continue  # it left the scrolling area
            elif todo == REREAD:
                track.unsure = True
                unsure.append(track.cover)
            kept.append(track)
        self._tracks = kept
        return unsure

    def follow_move(
        self, action: Callable[[Rect], str], dx: int, dy: int
    ) -> list[Rect]:
        """Move tracks with a window that moved whole. Returns places that must be read
        again: covers whose new place doesn't show exactly the pixels they had."""
        unsure = []
        for track in self._tracks:
            todo = action(track.cover)
            if todo == MOVE:
                track.move(dy, dx)
            elif todo == REREAD:
                track.unsure = True
                unsure.append(track.cover)
        return unsure

    def follow_each(self, image: np.ndarray, changed: np.ndarray) -> bool:
        """For cameras: move each cover whose pixels changed to where its own text went.

        A sheet in a hand moves on its own, apart from the person and anything else, so
        there is no one shift for the frame. Each cover's text, as last read, is looked for
        near where it was, ahead by its last move. The new spot is taken only if it passes
        the same blurred-tolerance check that keeps a cover on still text, so a cover only
        ever follows its own text, never nearby motion. Text that blurred, tilted or grew
        too much fails the check and is left to reads and the unsure margin, as before.
        Returns whether any cover moved.
        """
        height, width = image.shape[:2]
        gray = None
        followed = False
        for track in self._tracks:
            x1, y1, x2, y2 = track.cover
            rows = slice(max(int(y1), 0), max(min(int(y2) + 1, height), 0))
            columns = slice(max(int(x1), 0), max(min(int(x2) + 1, width), 0))
            if not changed[rows, columns].any() or self._unchanged(track, image):
                continue
            if gray is None:  # shrunk to grey and softened once, for every cover
                gray = cv2.blur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3))
            shift = self._find(track, gray)
            if shift is not None:
                before = track.reads, track.pixels_at, track.followed, track.lead
                track.move(shift[1], shift[0])
                if self._unchanged(track, image):
                    track.unsure = False
                    track.velocity = shift
                    track.lost = 0
                    followed = True
                    continue
                track.reads, track.pixels_at, track.followed, track.lead = before
            self._lose(track)
            followed = followed or track.lead is not None
        return followed

    def _lose(self, track: _Track) -> None:
        """Following lost text that was moving, most often to motion blur. Until a read or
        a match finds it again, stretch the cover along its last move, so it covers the
        path the text is likely on. This only ever adds black, never moves a cover off."""
        if track.velocity == (0, 0) or track.lost >= _LEAD_FRAMES:
            track.velocity, track.lost, track.lead = (0, 0), 0, None
            return
        track.lost += 1
        dx, dy = (track.lost * step for step in track.velocity)
        x1, y1, x2, y2 = _union(track.reads)
        track.lead = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _find(self, track: _Track, gray: np.ndarray) -> tuple[int, int] | None:
        """The dx, dy where the track's text best matches now, or None. Only proposes."""
        template = track.pixels
        rows, columns = template.shape
        if (
            rows < _FOLLOW_MIN_SIDE or columns < _FOLLOW_MIN_SIDE
            or int(template.max()) - int(template.min()) <= _SAME_TOLERANCE
        ):
            return None  # blank pixels match anywhere
        height, width = gray.shape
        left, top = track.pixels_at[0], round(track.pixels_at[1])
        ahead = track.lost + 1  # frames of travel since it was last found
        guess_x, guess_y = left + track.velocity[0] * ahead, top + track.velocity[1] * ahead
        x1, y1 = max(guess_x - self._reach, 0), max(guess_y - self._reach, 0)
        x2 = min(guess_x + columns + self._reach, width)
        y2 = min(guess_y + rows + self._reach, height)
        if x2 - x1 < columns or y2 - y1 < rows:
            return None  # it would be partly off the frame
        scores = cv2.matchTemplate(
            gray[y1:y2, x1:x2], cv2.blur(template, (3, 3)), cv2.TM_SQDIFF
        )
        _, _, (best_x, best_y), _ = cv2.minMaxLoc(scores)
        shift = (x1 + best_x - left, y1 + best_y - top)
        return None if shift == (0, 0) else shift

    def update(
        self,
        image: np.ndarray,
        sightings: list[Sighting],
        read: list[tuple[int, int, int, int]],
        pane: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Bring the tracks up to date with a finished scan.

        read is the areas OCR saw this frame, as top, bottom, left, right. A track
        counts as checked only if it lies wholly inside one of them. pane is a pane
        scrolling now, the same way: a track its top or bottom edge cuts through can't
        be checked at all, as no read sees the whole line.
        """
        self._pane = pane
        seen: dict[int, list[Sighting]] = {}
        spare = []
        for sighting in sightings:
            matches = [
                track
                for track in self._tracks
                if track.category == sighting.category
                and _overlap(track.cover, sighting.rect) >= _MIN_OVERLAP
            ]
            for track in matches:
                seen.setdefault(id(track), []).append(sighting)
            if not matches:
                spare.append(sighting)

        spare = [s for s in spare if not self._claim_by_text(s, seen, image)]
        kept = []
        for track in self._tracks:
            if id(track) in seen:
                self._refresh(track, seen[id(track)], image)
                kept.append(track)
            elif self._still_needed(track, image, read):
                kept.append(track)
        for sighting in spare:
            kept.append(self._start(sighting, image))
        self._tracks = self._merged(kept)

    def covers(self) -> list[tuple[str, Rect]]:
        """Every cover this frame needs, as category and rect."""
        found = []
        for track in self._tracks:
            left, top, right, bottom = track.cover
            # The margin is for not knowing where changed text went. A cover that moved
            # with a checked scroll or window move is where its text went, so a margin
            # would only flash black around it, typically as it slides under an edge.
            margin = self._margin if track.unsure and not track.followed else 0
            found.append(
                (track.category, (left - margin, top - margin, right + margin, bottom + margin))
            )
        return found

    def _claim_by_text(
        self, sighting: Sighting, seen: dict[int, list[Sighting]], image: np.ndarray
    ) -> bool:
        """Text that jumped: same text elsewhere, and its old place has changed."""
        for track in self._tracks:
            if (
                id(track) not in seen
                and track.category == sighting.category
                and track.text_hash == sighting.text_hash
                and not self._unchanged(track, image)
            ):
                seen[id(track)] = [sighting]
                return True
        return False

    def _start(self, sighting: Sighting, image: np.ndarray) -> _Track:
        pixels, place = self._grab(sighting.rect, image)
        return _Track(sighting.category, sighting.text_hash, [sighting.rect], pixels, place)

    def _refresh(self, track: _Track, sightings: list[Sighting], image: np.ndarray) -> None:
        rect = _union(sighting.rect for sighting in sightings)
        fresh = any(sighting.fresh for sighting in sightings)
        same_text = any(sighting.text_hash == track.text_hash for sighting in sightings)
        replaced = fresh and not same_text and not self._unchanged(track, image)
        if replaced or _moved(track.reads[-1], rect, self._wobble):
            track.reads = [rect]  # the text moved or was replaced, so old places are history
        elif fresh or not _contains(track.cover, rect):
            track.reads = [*track.reads, rect][-_KEEP_READS:]
        if fresh:
            track.pixels, track.pixels_at = self._grab(rect, image)
            if not same_text:
                track.text_hash = sightings[0].text_hash
        track.clear_reads = 0
        track.unsure = False
        track.check_at = None
        track.followed = False
        track.lost, track.lead = 0, None

    def _still_needed(
        self, track: _Track, image: np.ndarray, read: list[tuple[int, int, int, int]]
    ) -> bool:
        """Decide what to do with a track this scan didn't find."""
        height, width = image.shape[:2]
        left, top = max(track.cover[0], 0), max(track.cover[1], 0)
        right, bottom = min(track.cover[2], width), min(track.cover[3], height)
        checked = any(
            area[2] <= left and area[0] <= top and area[3] >= right and area[1] >= bottom
            for area in read
        ) and not self._cut_by_pane(track)
        if self._unchanged(track, image):
            # The secret is still there and OCR just didn't see it. Keep trying.
            track.clear_reads = 0
            track.unsure = False
            if checked or track.check_at is None:
                track.check_at = self._now + _RECHECK_FRAMES
            return True
        if right <= left or bottom <= top:
            return False  # it left the frame
        track.unsure = True
        if checked and track.lead is None:
            # A read of text that following lost mid-move proves nothing: what lost it,
            # motion blur, also keeps OCR from reading it.
            track.clear_reads += 1
            if track.clear_reads >= _CLEAR_READS:
                return False
        track.check_at = self._now + 1
        return True

    def _cut_by_pane(self, track: _Track) -> bool:
        """Whether the scrolling pane's top or bottom edge cuts through its text."""
        pane = getattr(self, '_pane', None)
        if pane is None:
            return False
        top, bottom, left, right = pane
        x1, y1, x2, y2 = track.cover
        across = x1 < right and x2 > left
        return across and (y1 < top < y2 or y1 < bottom < y2)

    def _grab(self, rect: Rect, image: np.ndarray) -> tuple[np.ndarray, tuple[int, float]]:
        height, width = image.shape[:2]
        left, right = max(int(rect[0]), 0), min(int(rect[2]) + 1, width)
        top, bottom = max(int(rect[1]), 0), min(int(rect[3]) + 1, height)
        return cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY), (left, top)

    def _unchanged(self, track: _Track, image: np.ndarray) -> bool:
        """Whether the text's own pixels still show, wherever its place has moved to."""
        height, width = image.shape[:2]
        left, top = track.pixels_at[0], round(track.pixels_at[1])
        x1, y1 = max(left, 0), max(top, 0)
        x2 = min(left + track.pixels.shape[1], width)
        y2 = min(top + track.pixels.shape[0], height)
        if x2 <= x1 or y2 <= y1:
            return False
        before = track.pixels[y1 - top : y2 - top, x1 - left : x2 - left]
        if int(before.max()) - int(before.min()) <= _SAME_TOLERANCE:
            return False  # only blank pixels are left to compare, which proves nothing
        now = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        different = cv2.absdiff(cv2.blur(now, (3, 3)), cv2.blur(before, (3, 3))) > _SAME_TOLERANCE
        return float(different.mean()) <= _SAME_MAX_MISMATCH

    def _merged(self, tracks: list[_Track]) -> list[_Track]:
        """Join tracks that one read showed to be the same text."""
        kept: dict[tuple[str, Rect], _Track] = {}
        for track in tracks:
            twin = kept.get((track.category, track.reads[-1]))
            if twin is None:
                kept[(track.category, track.reads[-1])] = track
            else:
                twin.reads = [*track.reads[:-1], *twin.reads][-_KEEP_READS:]
        return list(kept.values())
