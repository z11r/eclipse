"""Scanned frames kept unfinished until just before they go out.

The delay exists so that a secret found late still never shows. That only works if
what is found late can still change the frames that are waiting. So a scanned frame is
not drawn when its scan ends: it waits here, with what its own scan found, until its
output deadline is close. Until then:

- the detector's look-ahead may settle more of its incoming rows (settled()), and
- a cover found in a later frame is traced back onto it, if that frame showed the very
  same text pixels, moved or not (trace()).

Nothing on the stream may change unless there is something to redact. So unchecked
rows are never drawn as black, and a traced cover goes only where the secret's own
pixels showed. Only a new cover is traced: not one the frame before already had, and
only once the next few frames show it is real. Where they read its text again, each read
must find it again: OCR that finds a line on some frames and not others is misreading
it, and traced back that would be a second of black on plain text. A trace starts from the text's ink inside the detector's box, and each
step back to the frame scanned before tries the moves the detector followed for that
frame, then no move at all. A step counts only if the text's pixels that were inside
the area it moves with are the same in both frames. A line coming in over a pane's
edge is matched on the rows that already showed, so its first visible rows are covered
too. The walk stops at the first frame where none of the text showed, where its pixels
differ, or where it was covered already.

A frame still holding unread rows at its deadline isn't drawn either: the last safe
picture goes out instead (drawing()). That is the one change made without a known
secret, as fail closed requires, and the pipeline counts it.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from eclipse.types import Detection, Frame

NOT_COVERS = ("unread", "held")  # per-frame placeholders, not covers for a secret
CARET_COLUMNS = 3  # columns allowed to differ, as a blinking caret does; no letter fits
COVERED_SHARE = 0.95  # of a place, under covers already, to count as covered
CONTINUED_SHARE = 0.5  # of a cover, under the frame before's, for it to be the same track
CONFIRM_FRAMES = 6  # frames after a new cover that decide whether it is real: 0.1 s at 60
CONFIRM_MISSES = 1  # of them that may contradict it, as when one read stops short
SOFT_TOLERANCE = 40  # as the detector's scroll check, for moves between pixels
WHOLE_PIXEL = 0.05  # a move this close to whole pixels lands text on the same pixels

Rect = tuple[int, int, int, int]  # x, y, w, h


@dataclass(eq=False)
class Retained:
    """One scanned frame, not yet drawn."""

    frame: Frame
    fresh: bool
    size: int
    found: list[Detection] | None  # what its scan reported; None if the scan failed
    generation: int  # the settings it was scanned under
    spent: float  # seconds its scan took
    motions: list[tuple] | None = None  # the detector's follows since the frame before
    reads: list[tuple] | None = None  # where its scan read pixels; None if not known
    sightings: list[tuple] | None = None  # (category, x, y, w, h) its reads found; None: unknown
    settled: list[Detection] | None = None  # the latest later frames settled for it
    added: list[Detection] = field(default_factory=list)  # covers traced back onto it
    finished: bool = False  # drawn: nothing can change it any more

    def boxes(self) -> list[Detection]:
        """Everything to cover now: the latest settled boxes, or else what its scan
        reported, and every cover traced back onto it."""
        base = self.settled if self.settled is not None else (self.found or [])
        return [*base, *self.added]

    def drawing(self) -> list[Detection] | None:
        """What to draw over this frame, or None to send the last safe picture instead:
        its scan failed, or rows it shows were still unread at its deadline. Unread rows
        are never drawn black, as black is only for secrets."""
        if self.found is None:
            return None
        boxes = self.boxes()
        if any(d.category == "unread" for d in boxes):
            return None
        return boxes


def _covers(boxes: list[Detection]) -> list[Detection]:
    return [d for d in boxes if d.category not in NOT_COVERS]


def _clip(rect: Rect, other: Rect) -> Rect | None:
    x1, y1 = max(rect[0], other[0]), max(rect[1], other[1])
    x2 = min(rect[0] + rect[2], other[0] + other[2])
    y2 = min(rect[1] + rect[3], other[1] + other[3])
    return (x1, y1, x2 - x1, y2 - y1) if x2 > x1 and y2 > y1 else None


def covered(place: Rect, boxes: list[Detection], share: float = COVERED_SHARE) -> bool:
    """Whether covers already hide nearly all of a place, or the given share of it."""
    x, y, w, h = place
    mask = np.zeros((h, w), bool)
    for d in boxes:
        part = _clip(place, (d.x, d.y, d.w, d.h))
        if part is not None:
            px, py, pw, ph = part
            mask[py - y : py - y + ph, px - x : px - x + pw] = True
    return mask.mean() >= share


def _background(area: np.ndarray) -> np.ndarray:
    """The most common colour of an area, exactly."""
    flat = area.reshape(-1, 3).astype(np.int32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    colours, counts = np.unique(packed, return_counts=True)
    value = int(colours[counts.argmax()])
    return np.array([value >> 16, (value >> 8) & 255, value & 255], np.uint8)


SHOWN, HIDDEN, UNEXPLAINED = "shown", "hidden", "unexplained"


def _ink(rows_cols: np.ndarray) -> tuple[int, int, int, int] | None:
    """The bounding box of a mask's True pixels, as x, y, w, h, or None."""
    rows, columns = np.flatnonzero(rows_cols.any(axis=1)), np.flatnonzero(rows_cols.any(axis=0))
    if not rows.size:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1] - columns[0]) + 1, int(rows[-1] - rows[0]) + 1


def text_box(image: np.ndarray, box: Rect) -> tuple[Rect, np.ndarray] | None:
    """Where the text is inside a detector's box, and its background colour. The colour
    is the box's most common one, padding included, as tight around dense text the ink
    could outnumber it. The text is everything else: padding, and blank rows OCR added,
    are left out. The whole box if it is all one colour; None if it lies outside."""
    height, width = image.shape[:2]
    part = _clip(box, (0, 0, width, height))
    if part is None:
        return None
    x, y, w, h = part
    area = image[y : y + h, x : x + w]
    background = _background(area)
    ink = _ink((area != background).any(axis=2))
    if ink is None:
        return part, background
    return (x + ink[0], y + ink[1], ink[2], ink[3]), background


def compare(
    later: np.ndarray, earlier: np.ndarray, box: Rect, clip: Rect, dx: float, dy: float,
    exact: bool = True, background: np.ndarray | None = None,
) -> tuple[str, Rect | None]:
    """Whether the earlier image showed the very text the later one shows in box, if
    content inside clip moved dx, dy from earlier to later. Only the later pixels that
    came from inside clip are compared: the rest of the text wasn't in view yet.

    SHOWN, with where its text was in the earlier image, when those pixels match: all of
    them on an exact screen but a caret's few columns, or within the detector's scroll
    tolerance for a move between pixels. HIDDEN when none of the text was there to
    compare: only its background was. UNEXPLAINED when they differ.
    """
    height, width = later.shape[:2]
    sx, sy = round(dx), round(dy)
    later_box = _clip(box, (0, 0, width, height))
    if later_box is None:
        return HIDDEN, None
    # Later pixels that came from inside clip, and are inside clip and the frame now.
    region = _clip(later_box, clip)
    region = region and _clip(region, (clip[0] + sx, clip[1] + sy, clip[2], clip[3]))
    region = region and _clip(region, (sx, sy, width, height))
    if region is None:
        return HIDDEN, None  # none of it was inside the area it moves with yet
    x, y, w, h = region
    now = later[y : y + h, x : x + w]
    before = earlier[y - sy : y - sy + h, x - sx : x - sx + w]
    lx, ly, lw, lh = later_box
    whole = later[ly : ly + lh, lx : lx + lw]
    if background is None:
        background = _background(whole)
    ink = (now != background).any(axis=2)
    uniform = bool((whole == background).all())
    if uniform:
        ink = np.ones(ink.shape, bool)  # one colour throughout: the colour is the text
    elif not ink.any():
        return HIDDEN, None
    if exact:
        differ = (before != now).any(axis=2)
    else:
        soft = cv2.absdiff(cv2.blur(before, (3, 3)), cv2.blur(now, (3, 3)))
        differ = soft.max(axis=2) > SOFT_TOLERANCE
    if np.count_nonzero(differ.any(axis=0)) > CARET_COLUMNS:
        return UNEXPLAINED, None  # other text, or more of it typed since
    if not (ink & ~differ).any():
        return UNEXPLAINED, None
    # Cover all of the text the earlier frame shows: in its box moved back, inside the
    # area, wherever it isn't the background. Rows that left the view since count too.
    back = _clip((box[0] - sx, box[1] - sy, box[2], box[3]), clip)
    back = back and _clip(back, (0, 0, width, height))
    if back is None:
        return UNEXPLAINED, None
    if uniform:
        return SHOWN, back  # nothing to tell text from background: the whole box
    bx, by, bw, bh = back
    shown = _ink((earlier[by : by + bh, bx : bx + bw] != background).any(axis=2))
    if shown is None:
        return HIDDEN, None
    return SHOWN, (bx + shown[0], by + shown[1], shown[2], shown[3])


def _centre_in(box: Rect, area: Rect) -> bool:
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return area[0] <= cx < area[0] + area[2] and area[1] <= cy < area[1] + area[3]


@dataclass
class _Step:
    record: Retained  # the earlier frame
    place: Rect  # where its text showed in that frame
    shift: tuple[int, int]  # how far the text moved from it to the later frame
    clip: Rect  # the area the text lives in there: a pane, a window, or the frame


def step_back(
    later: Retained, earlier: Retained, box: Rect, clip: Rect,
    background: np.ndarray | None = None,
) -> _Step | None:
    """One frame back: where the earlier frame showed the text in box, or None if it
    didn't show it, or showed something else there.

    A move the detector followed decides when the text's box is centred inside the area
    that moved: the text moved with it, so it lives inside that area from then on.
    Otherwise the text stayed put, and its pixels must be the same."""
    image = later.frame.image
    height, width = image.shape[:2]
    inside = _clip(box, clip)
    inside = inside and _clip(inside, (0, 0, width, height))
    if inside is None:
        return None  # it has left the area it lives in
    for x, y, w, h, dx, dy in later.motions or ():
        # The area this move covered holds both the text's place now and where it came
        # from. It is not narrowed to the area of a later step: a dragged window was
        # somewhere else then.
        area = _clip((x, y, w, h), (0, 0, width, height))
        if area is None or not _centre_in(inside, area):
            continue
        exact = later.frame.source == "screen" and all(
            abs(v - round(v)) < WHOLE_PIXEL for v in (dx, dy)
        )
        verdict, place = compare(
            image, earlier.frame.image, box, area, dx, dy, exact, background
        )
        if verdict == SHOWN:
            return _Step(earlier, place, (round(dx), round(dy)), area)
        if verdict == HIDDEN:
            return None  # it hadn't come into the area yet
    verdict, place = compare(
        image, earlier.frame.image, box, clip, 0, 0, later.frame.source == "screen",
        background,
    )
    if verdict == SHOWN:
        return _Step(earlier, place, (0, 0), clip)
    return None


def _continues(record: Retained, before: Retained, box: Rect, category: str) -> bool:
    """Whether a cover carries on one the frame before already had: the same category
    over at least half of it there, unmoved or moved by a move the detector followed.

    Only a new cover is traced back. One carried on was traced when it began, or was
    there all along; tracing it again would copy wherever it has drifted since, like a
    cover that lags a fast drag or lingers where text left, onto every earlier frame.
    A cover that grew, because more of its text was read, is new where it grew."""
    if not before.found:
        return False
    shifts = {(0, 0)} | {(round(dx), round(dy)) for *_, dx, dy in record.motions or ()}
    same = [d for d in _covers(before.boxes()) if d.category == category]
    moved = [
        Detection(d.x + sx, d.y + sy, d.w, d.h, d.category, d.confidence, d.ttl_ms, d.frame_id)
        for d in same for sx, sy in shifts
    ]
    return bool(moved) and covered(box, moved, CONTINUED_SHARE)


@dataclass(eq=False)
class _Pending:
    """A new cover, waiting for the frames after it to show whether it is real."""

    record: Retained  # the frame it was found in
    found: Detection
    box: Rect  # where it is in the newest frame checked
    checked: int = 0
    misses: int = 0

    def check(self, after: Retained) -> None:
        """Count the next frame for or against it, and follow it there.

        Where that frame's scan read the text again, only a read that found it again
        counts for it: OCR that finds a line on some frames and not others is misreading
        it, and a misread spread back over a second of frames is black on plain text.
        Where nothing was read, the text didn't change, and a cover kept on it counts."""
        x, y, w, h = self.box
        shifts = [(0, 0)] + [
            (round(dx), round(dy)) for mx, my, mw, mh, dx, dy in after.motions or ()
            if _centre_in(self.box, (mx, my, mw, mh))
        ]
        box = (x + shifts[-1][0], y + shifts[-1][1], w, h)
        category = self.found.category
        covers = [d for d in _covers(after.found or []) if d.category == category]
        kept = next((d for d in covers if covered(box, [d], 0.3)), None)
        read = after.reads is None or any(_clip(box, r) is not None for r in after.reads)
        if after.found is None:
            hit = False
        elif read and after.sightings is not None:
            hit = any(
                c == category and _clip(box, (sx, sy, sw, sh)) is not None
                for c, sx, sy, sw, sh in after.sightings
            )
        else:
            hit = kept is not None
        self.checked += 1
        self.misses += not hit
        self.box = (kept.x, kept.y, kept.w, kept.h) if kept is not None else box

    @property
    def decided(self) -> bool:
        return self.checked >= CONFIRM_FRAMES or self.misses > CONFIRM_MISSES

    @property
    def real(self) -> bool:
        return self.misses <= CONFIRM_MISSES


def _carried(later: Retained, earlier: Retained, box: Rect, category: str) -> bool:
    """Whether the earlier frame's scan reported this very cover, unmoved or moved by a
    move the detector followed: the same tracked cover, so it is covered there already.
    Cheap, so a cover held on every frame isn't checked pixel by pixel each time."""
    if not earlier.found:
        return False
    places = {box}
    for _x, _y, _w, _h, dx, dy in later.motions or ():
        places.add((box[0] - round(dx), box[1] - round(dy), box[2], box[3]))
    return any(
        (d.x, d.y, d.w, d.h) in places and d.category == category for d in earlier.found
    )


class RetainedFrames:
    """Scanned frames waiting to be drawn, oldest first. Thread safe: the scanner adds
    and updates them, and the finisher takes each one as its deadline comes."""

    def __init__(self) -> None:
        self._frames: deque[Retained] = deque()
        self._lock = threading.Lock()
        self.traced = 0  # covers traced back onto earlier frames
        self.traced_frames = 0  # earlier frames that gained one
        self.too_late = 0  # covers that would have reached a frame already drawn
        self._last_taken: Retained | None = None  # the frame drawn most recently
        self._pending: list[_Pending] = []  # new covers not yet shown to be real

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def add(self, record: Retained) -> None:
        with self._lock:
            self._frames.append(record)

    def oldest(self) -> Retained | None:
        with self._lock:
            return self._frames[0] if self._frames else None

    def take(self, due: float) -> Retained | None:
        """The oldest frame if it was captured at or before due, now finished."""
        with self._lock:
            if not self._frames or self._frames[0].frame.ts > due:
                return None
            record = self._frames.popleft()
            record.finished = True
            self._last_taken = record
            return record

    def drain(self) -> list[Retained]:
        """Every frame, taken without drawing, as when the pipeline stops."""
        with self._lock:
            out = list(self._frames)
            for record in out:
                record.finished = True
            self._frames.clear()
            self._last_taken = None
            self._pending = []
            return out

    def _waiting(self) -> list[Retained]:
        with self._lock:
            return list(self._frames)

    def _late(self, later: Retained, box: Rect, clip: Rect, background=None) -> None:
        """Count a cover that the frame drawn last needed too: found too late for it."""
        with self._lock:
            earlier = self._last_taken
        if earlier is None or earlier.frame.image.shape != later.frame.image.shape:
            return
        if later.frame.frame_id - earlier.frame.frame_id < 1:
            return
        step = step_back(later, earlier, box, clip, background)
        if step is not None and not covered(step.place, _covers(earlier.boxes())):
            with self._lock:
                self.too_late += 1

    def refresh(self, settle) -> None:
        """Ask settle(frame) again for every waiting frame, keeping the last answer when
        it has none, as after a failed scan."""
        for record in self._waiting():
            if record.found is None:
                continue
            boxes = settle(record.frame)
            if boxes is not None:
                with self._lock:
                    if not record.finished:
                        record.settled = boxes

    def trace(self, record: Retained) -> int:
        """Note the record's new covers, count it for or against the new covers before
        it, and trace back each one the frames after it have shown to be real, onto the
        earlier frames still waiting that showed the same text. Returns how many covers
        were added."""
        frames = self._waiting()
        try:
            index = frames.index(record)
        except ValueError:
            return 0
        added = 0
        still = []
        for pending in self._pending:
            pending.check(record)
            if not pending.decided:
                still.append(pending)
            elif pending.real:
                added += self._trace_pending(pending, frames)
        self._pending = still
        if not record.found:
            return added
        for found in _covers(record.found):
            detector_box: Rect = (found.x, found.y, found.w, found.h)
            if record.reads is not None and not any(
                _clip(detector_box, read) is not None for read in record.reads
            ):
                continue  # carried from earlier frames, not found in this one
            if index and _continues(record, frames[index - 1], detector_box, found.category):
                continue  # an existing cover carried on, not a new one
            self._pending.append(_Pending(record, found, detector_box))
        return added

    def flush(self) -> int:
        """Decide every new cover still waiting on what has been seen of it, as at the end
        of a stream. Returns how many covers were added."""
        frames = self._waiting()
        pending, self._pending = self._pending, []
        return sum(self._trace_pending(p, frames) for p in pending if p.real)

    def _trace_pending(self, pending: _Pending, frames: list) -> int:
        try:
            index = frames.index(pending.record)
        except ValueError:
            return 0  # drawn already
        return self._trace_one(pending.record, pending.found, frames[:index][::-1])

    def _trace_one(self, record: Retained, found: Detection, earlier_frames: list) -> int:
        """Trace one confirmed new cover of record back through earlier_frames, newest
        first. Returns how many covers were added."""
        whole = (0, 0, record.frame.image.shape[1], record.frame.image.shape[0])
        detector_box: Rect = (found.x, found.y, found.w, found.h)
        text = text_box(record.frame.image, detector_box)
        if text is None:
            return 0
        box, background = text
        clip, later = whole, record
        added = 0
        for earlier in earlier_frames + [None]:
            if earlier is None:
                self._late(later, box, clip, background)  # past every frame waiting
                break
            if (
                earlier.frame.source != later.frame.source
                or earlier.frame.image.shape != later.frame.image.shape
            ):
                break
            if _carried(later, earlier, detector_box, found.category):
                break  # the detector's own cover was there already
            step = step_back(later, earlier, box, clip, background)
            if step is None:
                break  # not shown there, or something else was there
            if covered(step.place, _covers(earlier.boxes())):
                break  # covered from here back already
            cover = Detection(
                *step.place, found.category, found.confidence, found.ttl_ms,
                earlier.frame.frame_id,
            )
            with self._lock:
                if earlier.finished:
                    self.too_late += 1
                    break
                if not earlier.added:
                    self.traced_frames += 1
                earlier.added.append(cover)
                self.traced += 1
            added += 1
            sx, sy = step.shift
            box = (box[0] - sx, box[1] - sy, box[2], box[3])
            detector_box = (
                detector_box[0] - sx, detector_box[1] - sy, detector_box[2], detector_box[3]
            )
            clip, later = step.clip, earlier
        return added
