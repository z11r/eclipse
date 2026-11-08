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
    ) -> None:
        """noise_floor above 0 ignores sensor noise; use it for cameras, not screens.

        ocr_threads sizes the RapidOCR reader made when no reader is given. A second
        detector running beside the screen's should take fewer, so the two share the cores.
        """
        self._detector = Detector(
            keywords=keywords, reader=reader, pixel_scale=pixel_scale, noise_floor=noise_floor,
            ocr_threads=ocr_threads,
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
        return [
            Detection(
                x,
                y,
                w,
                h,
                _CATEGORY_NAMES.get(category, category),
                1.0,
                UNREAD_TTL_MS if category == "unread" else TTL_MS,
                frame.frame_id,
            )
            for x, y, w, h, category in self._detector.detect(frame.image)
        ]


detect = TextDetector()
