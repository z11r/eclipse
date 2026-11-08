"""Look-ahead for rows scrolling into view.

Rows coming onto the screen during a scroll can't be read until whole lines show, so
each frame reports them as an 'unread' strip, and they go out black. Output runs behind
the scan, though, so by the time a frame is sent, later frames have often read those
same rows. This module follows each strip's rows through the frames after it and,
once they have been read, tells which of them can go out as they are and where their
secrets are.

A row is only cleared when every step between the two frames is accounted for: each
frame either left its pixels untouched or moved them with a scroll whose shift
reproduces them, checked pixel by pixel as the detector checks any scroll. Rows that
fail a check, leave the view, or are never read stay black. So a wrong guess here
costs a black row, never a shown one.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

Box = tuple[int, int, int, int, str]

_KEEP_FRAMES = 120  # frames remembered if nobody asks for them, so memory stays bounded
_WAITING, _SETTLED, _LOST = 0, 1, 2


@dataclass(frozen=True)
class ScrollStep:
    """How one frame's scroll moved a pane, as the detector checked it."""

    shift: float  # rows down, fractions included
    left: int  # the pane's columns
    right: int
    top: int  # the rows checked against the previous frame
    unexplained: np.ndarray  # those rows, the pane's columns: pixels the shift doesn't explain
    view: tuple[int, int]  # top and bottom of the pane


@dataclass
class _Strip:
    """One frame's black strip of incoming rows, and what later frames found in it."""

    x: int
    y: int
    w: int
    h: int
    edge: int  # the row of the pane's edge its rows came in over
    inward: int  # 1 if they came in moving down, from the top edge; -1 from the bottom
    ink: list  # x, y, w, h over what its rows show; the rest is plain background
    offset: float = 0.0  # how far its rows have scrolled since
    rows: np.ndarray = field(init=False)
    covers: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.rows = np.full(self.h, _WAITING, np.uint8)
        # Why each waiting row last failed to clear: 0 never checked, 1 still unread,
        # 2 cut by the pane's edge, 3 doubtful text. For the live log only.
        self.why = np.zeros(self.h, np.uint8)

    @property
    def waiting(self) -> bool:
        return bool((self.rows == _WAITING).any())

    def _now(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rows of this frame that each of the strip's rows now lies across."""
        place = self.y + index + self.offset
        first = np.floor(place).astype(int)
        return first, np.ceil(place).astype(int)

    def _lose(self, bad_rows: np.ndarray, top: int) -> int:
        """Lose waiting rows that lie on any bad row; bad_rows starts at row top.
        Returns how many were lost."""
        index = np.flatnonzero(self.rows == _WAITING)
        if not index.size:
            return 0
        first, last = self._now(index)
        bad = np.ones(index.size, bool)
        inside = (first >= top) & (last < top + bad_rows.size)
        bad[inside] = bad_rows[first[inside] - top] | bad_rows[last[inside] - top]
        self.rows[index[bad]] = _LOST
        return int(bad.sum())

    def follow(self, changed: np.ndarray, step: ScrollStep | None, lost: dict) -> None:
        """Account for one more frame. changed is every pixel that differs from the last.
        Rows lost are counted in lost by the check that failed, for the live log."""
        height, width = changed.shape
        left, right = max(self.x, 0), min(self.x + self.w, width)
        moved = step is not None and step.left < right and step.right > left
        if moved:
            self.offset += step.shift
        top = math.floor(self.y + self.offset)
        bottom = math.ceil(self.y + self.offset + self.h) + 1
        rows = np.arange(top, bottom)
        off = (rows < 0) | (rows >= height)
        if not moved:
            # Nothing scrolled here: any change at all is something unaccounted for.
            still = np.zeros_like(off)
            still[~off] = changed[rows[~off], left:right].any(axis=1)
            checks = (('view', off), ('still', still))
        else:
            low, high = step.view
            checked = step.top + step.unexplained.shape[0]
            off |= (rows < max(low, step.top)) | (rows >= min(high, checked))
            within = rows[~off]
            # Columns the scroll check covered, then any of the strip's own beyond them.
            a, b = max(left, step.left), min(right, step.right)
            unexplained = np.zeros_like(off)
            unexplained[~off] = step.unexplained[
                within - step.top, a - step.left:b - step.left
            ].any(axis=1)
            edge = np.zeros_like(off)
            for start, stop in ((left, a), (b, right)):
                if stop > start:
                    edge[~off] |= changed[within, start:stop].any(axis=1)
            checks = (('view', off), ('unexplained', unexplained), ('edge', edge))
        for name, bad in checks:
            count = self._lose(bad, top)
            if count:
                lost[name] = lost.get(name, 0) + count

    def settle(
        self, boxes: list[Box], whole_line: int, size: tuple[int, int], held: list,
        cut_known: bool,
    ) -> None:
        """Clear waiting rows this frame has read, with the secrets it found there.

        held is where rows can't count as read, as x, y, w, h: a line the pane's edge
        cuts through, which no read sees whole, and uncovered text that looks like a
        misread secret. Those rows wait for a better read, or stay black. If the pane's
        edges aren't known (cut_known false), rows must be whole_line in from the edge
        they came in over instead.
        """
        index = np.flatnonzero(self.rows == _WAITING)
        if not index.size:
            return
        first, last = self._now(index)
        # why: 1 still unread in this frame, 2 cut by the pane's edge, 3 doubtful text.
        why = np.zeros(index.size, np.uint8)
        if not cut_known:
            inside = (first - self.edge) if self.inward > 0 else (self.edge - 1 - last)
            why[inside < whole_line] = 2
        right = self.x + self.w
        for x, y, w, h, *kind in held:
            if x < right and x + w > self.x:
                on = (first < y + h) & (last >= y) & (why == 0)
                why[on] = 3 if kind == ['doubtful'] else 2
        for x, y, w, h, category in boxes:
            if category == 'unread' and x < right and x + w > self.x:
                why[(first < y + h) & (last >= y)] = 1
        unread = why > 0
        self.why[index[unread]] = why[unread]
        read = index[~unread]
        if not read.size:
            return
        self.rows[read] = _SETTLED
        # Covers near the strip as well as on it, and whole: a cover's padding can hide
        # text beside it, and the strip must go out hiding all that this frame hides.
        top, bottom = self.y + self.offset - whole_line, self.y + self.offset + self.h + whole_line
        for x, y, w, h, category in boxes:
            if category == 'unread' or x >= right + whole_line or x + w <= self.x - whole_line:
                continue
            if y >= bottom or y + h <= top:
                continue
            # Back to where the text was in the strip's own frame, rounded outwards, and
            # kept inside the frame, which is the same size, and inside the pane: past the
            # edge the rows came in over, that text wasn't on screen yet.
            y1 = max(math.floor(y - self.offset), 0)
            y2 = min(math.ceil(y + h - self.offset), size[0])
            if self.inward > 0:
                y1 = max(y1, self.edge)
            else:
                y2 = min(y2, self.edge)
            x1, x2 = max(x, 0), min(x + w, size[1])
            if y2 > y1 and x2 > x1:
                self.covers.add((x1, y1, x2 - x1, y2 - y1, category))

    def boxes(self) -> list[Box]:
        """Covers for its secrets, and black over what every row not cleared shows."""
        out = list(self.covers)
        black = np.flatnonzero(self.rows != _SETTLED)
        if black.size:
            runs = np.split(black, np.flatnonzero(np.diff(black) > 1) + 1)
            for run in runs:
                top, bottom = self.y + int(run[0]), self.y + int(run[-1]) + 1
                for x, y, w, h in self.ink:
                    a, b = max(y, top), min(y + h, bottom)
                    if b > a:
                        out.append((x, a, w, b - a, 'unread'))
        return out


@dataclass
class _Record:
    boxes: list[Box]  # everything but the strips
    strips: list[_Strip]
    pane: tuple[int, int, int, int] | None  # x, y, w, h of the pane its strips came into


class IncomingRows:
    """Follows black incoming rows into later frames. Call record() once per frame, in
    order, and settled() for a frame when it is about to go out."""

    def take_held_rows(self) -> dict:
        """Rows held since last asked, by why: lost, unread, cut or doubtful."""
        held, self._held_rows = self._held_rows, {}
        return held

    def take_lost_rows(self) -> dict:
        """Rows lost since last asked, by the check that failed: view (left the pane or
        the rows checked), unexplained (pixels the scroll doesn't reproduce), edge (a
        change in the strip's columns beside the pane) or still (a change, no scroll)."""
        lost, self._lost_rows = self._lost_rows, {}
        return lost

    def __init__(self, whole_line: int, keep_frames: int = _KEEP_FRAMES) -> None:
        """whole_line: rows a line needs to be whole, as the detector keeps black after
        reading incoming rows, in case the last line was cut."""
        self._frames: OrderedDict = OrderedDict()
        self._held_rows: dict = {}
        self._lost_rows: dict = {}
        self._whole_line = whole_line
        self._keep = keep_frames

    @property
    def following(self) -> bool:
        """Whether any strip still waits on later frames, so they need a changed mask."""
        return any(strip.waiting for record in self._frames.values() for strip in record.strips)

    def clear(self) -> None:
        self._frames.clear()

    def rows_after(self, shift: float) -> tuple[int, int] | None:
        """The rows the strips still waiting will lie across once moved by shift, as
        top and bottom, or None if none wait."""
        spans = [
            (strip.y + strip.offset + shift, strip.y + strip.offset + shift + strip.h)
            for record in self._frames.values() for strip in record.strips if strip.waiting
        ]
        if not spans:
            return None
        return math.floor(min(a for a, _ in spans)) - 1, math.ceil(max(b for _, b in spans)) + 2

    def record(
        self,
        key,
        boxes: list[Box],
        strips: list[tuple[Box, int, int, list]],
        changed: np.ndarray | None,
        step: ScrollStep | None,
        size: tuple[int, int],
        doubtful: list = (),
        cut: list | None = None,
        pane: tuple[int, int, int, int] | None = None,
    ) -> None:
        """A frame has been scanned. boxes is everything it reports, its strips included,
        and strips gives each strip's box with the pane edge row it came in over, 1 if
        it came in moving down or -1 if up, and the boxes over what it shows. changed is
        None if nothing changed. size is the frame's height and width, the same for every
        frame followed. doubtful is text this frame read, as x, y, w, h, that may be a
        misread secret with no cover. cut is where the pane's edges cut through lines,
        the same way, or None if the pane's edges aren't known. pane is the scrolling
        pane, x, y, w, h, which is held, not covered, while rows in it stay unread."""
        # A frame at rest shows the line its pane's edge cuts through as it was read, as
        # one always has once a scroll stops. Earlier frames showing the same pixels, or
        # fewer of them, show nothing more, so at rest the cut line doesn't hold rows.
        at_rest = step is None
        held = [(*rect, 'doubtful') for rect in doubtful] + ([] if at_rest else list(cut or []))
        for record in self._frames.values():
            for strip in record.strips:
                if not strip.waiting:
                    continue
                if changed is not None:
                    strip.follow(changed, step, self._lost_rows)
                strip.settle(boxes, self._whole_line, size, held, at_rest or cut is not None)
        self._frames[key] = _Record(
            [box for box in boxes if box[4] != 'unread'],
            [
                _Strip(x, y, w, h, edge, inward, list(ink))
                for (x, y, w, h, _), edge, inward, ink in strips
            ],
            pane,
        )
        while len(self._frames) > self._keep:
            self._frames.popitem(last=False)

    def settled(self, key) -> list[Box] | None:
        """A frame's boxes with what later frames read in its strips, or None if it isn't
        known. It and every frame before it are forgotten."""
        record = self._frames.get(key)
        if record is None:
            return None
        while self._frames:
            first, _ = self._frames.popitem(last=False)
            if first == key:
                break
        out = list(record.boxes)
        unread = [box for strip in record.strips for box in strip.boxes() if box[4] == 'unread']
        for strip in record.strips:
            out.extend(box for box in strip.boxes() if box[4] != 'unread')
        if unread:
            names = {1: 'unread', 2: 'cut', 3: 'doubtful'}
            for strip in record.strips:
                lost = int((strip.rows == _LOST).sum())
                if lost:
                    self._held_rows['lost'] = self._held_rows.get('lost', 0) + lost
                for code, name in names.items():
                    waiting = strip.rows == _WAITING
                    count = int((waiting & (np.maximum(strip.why, 1) == code)).sum())
                    if count:
                        self._held_rows[name] = self._held_rows.get(name, 0) + count
        # What no later frame could clear, usually a line at the pane's edge, shows the
        # last checked picture there rather than black. Only that, not the whole pane:
        # the rest of the pane moves on, so a scroll doesn't pause for one line.
        out.extend((x, y, w, h, 'held') for x, y, w, h, _ in unread)
        return out
