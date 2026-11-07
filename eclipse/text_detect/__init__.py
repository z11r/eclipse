"""Text layer: connects the OCR detector to the shared Frame and Detection types."""

from __future__ import annotations

from typing import Iterable

from eclipse.text_detect.detection import Detector
from eclipse.types import Detection, Frame

TTL_MS = 500  # for the shared contract. CoverTracks decides when a cover really ends.
UNREAD_TTL_MS = 0  # unread rows are covered only while each frame still reports them
_CATEGORY_NAMES = {"2fa": "two_fa"}


class TextDetector:
    """Use one per source. Raises DetectionError if a scan fails."""

    def __init__(self, keywords: Iterable[str] = (), reader=None, pixel_scale: int = 1) -> None:
        self._detector = Detector(keywords=keywords, reader=reader, pixel_scale=pixel_scale)
        self._source = None

    @property
    def stats(self) -> dict:
        return self._detector.stats

    def __call__(self, frame: Frame) -> list[Detection]:
        if frame.source != self._source:
            self._detector.reset()  # another source's positions mean nothing here
            self._source = frame.source
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
