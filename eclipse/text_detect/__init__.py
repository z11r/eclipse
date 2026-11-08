"""Text layer: connects the OCR detector to the shared Frame and Detection types."""

from __future__ import annotations

from typing import Iterable

from eclipse.text_detect.detection import CATEGORIES, Detector
from eclipse.types import Category, Detection, Frame

TTL_MS = 500  # for the shared contract. CoverTracks decides when a cover really ends.
UNREAD_TTL_MS = 0  # unread rows are covered only while each frame still reports them
_CATEGORY_NAMES = {"2fa": "two_fa"}
_DETECTOR_NAMES = {shared: name for name, shared in _CATEGORY_NAMES.items()}


class TextDetector:
    """Use one per source. Raises DetectionError if a scan fails."""

    def __init__(
        self,
        keywords: Iterable[str] = (),
        reader=None,
        pixel_scale: int = 1,
        noise_floor: int = 0,
        ocr_threads: int = 6,
        rescue=None,
    ) -> None:
        """noise_floor above 0 ignores sensor noise; use it for cameras, not screens.

        ocr_threads sizes the RapidOCR reader made when no reader is given. A second
        detector running beside the screen's should take fewer, so the two share the cores.

        rescue is a slower, careful reader for the few lines the fast read leaves in doubt,
        like a "password" label with no value read. None, the default, never rereads.
        """
        self._detector = Detector(
            keywords=keywords, reader=reader, pixel_scale=pixel_scale, noise_floor=noise_floor,
            ocr_threads=ocr_threads, rescue=rescue,
        )
        self._source = None
        self._read_ahead = reader if hasattr(reader, "prepare") else None

    @property
    def stats(self) -> dict:
        return self._detector.stats

    def prepare(self, frame: Frame) -> bool:
        """Start work on a frame before its turn. False means it can't be read in time."""
        return self._read_ahead is None or self._read_ahead.prepare(frame)

    @property
    def reused_read(self) -> bool:
        """Whether the last frame needed no OCR, as earlier reads still applied."""
        return self._detector.reused_read

    def configure(
        self, keywords: Iterable[str] = (), enabled: Iterable[Category] | None = None
    ) -> None:
        """Change what counts as sensitive, all at once. Raises and changes nothing if bad.

        enabled uses the shared category names. Those the text layer doesn't handle are
        ignored, so a full list from the panel can be passed straight through.
        """
        wanted = None
        if enabled is not None:
            names = {_DETECTOR_NAMES.get(category, category) for category in enabled}
            wanted = names & CATEGORIES
        self._detector.configure(keywords, wanted)

    def cancel(self) -> None:
        """Stop scanning at the next chance. A read inside the OCR engine runs to its end."""
        self._detector.cancel()

    def __call__(self, frame: Frame) -> list[Detection]:
        if frame.source != self._source:
            self._detector.reset()  # another source's positions mean nothing here
            self._source = frame.source
        if self._read_ahead is not None:
            self._read_ahead.begin_frame(frame.frame_id)
        return _detections(self._detector.detect(frame.image, frame.frame_id), frame)

    def look_ahead(self) -> None:
        """Start following rows that scroll in, so settled() can clear them later."""
        self._detector.look_ahead()

    def settled(self, frame: Frame, forget: bool = True) -> list[Detection] | None:
        """This frame's detections again, with its incoming rows cleared where frames
        scanned since have read them. Rows still unread stay covered. None if the frame
        is no longer known, as after a failed scan; then use what it first reported.
        Ask in frame order: it forgets every frame before this one, and this one too
        unless forget is False, so it can be asked again after later frames."""
        if frame.source != self._source:
            return None
        boxes = self._detector.settled(frame.frame_id, forget)
        return None if boxes is None else _detections(boxes, frame)

    def forget_before(self, frame: Frame) -> None:
        """Stop following frames before this one, as none of them will be asked again."""
        if frame.source == self._source:
            self._detector.forget_before(frame.frame_id)

    def motions(self) -> list[tuple] | None:
        """How the last frame scanned moved content since the one scanned before it:
        (x, y, w, h, dx, dy) per followed scroll or window move, the area in the later
        frame's pixels. Empty if nothing was followed; None if not known. A hint only."""
        return self._detector.motions()

    def sightings(self) -> list[tuple] | None:
        """Secrets the last frame's own reads found, as (category, x, y, w, h), with the
        detector's category names. None if not known."""
        found = self._detector.sightings()
        if found is None:
            return None
        return [(_CATEGORY_NAMES.get(c, c), x, y, w, h) for c, x, y, w, h in found]

    def reads(self) -> list[tuple] | None:
        """Where the last frame scanned was read, as (x, y, w, h). A cover elsewhere was
        carried from earlier frames, so it says nothing new about them. None if not known."""
        return self._detector.reads()


def _detections(boxes, frame: Frame) -> list[Detection]:
    return [
        Detection(
            x,
            y,
            w,
            h,
            _CATEGORY_NAMES.get(category, category),
            1.0,
            UNREAD_TTL_MS if category in ("unread", "held") else TTL_MS,
            frame.frame_id,
        )
        for x, y, w, h, category in boxes
    ]


detect = TextDetector()
