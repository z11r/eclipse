"""Text layer: connects the OCR detector to the shared Frame and Detection types."""

from __future__ import annotations

from typing import Iterable

from eclipse.text_detect.detection import Detector
from eclipse.types import Detection, Frame

TTL_MS = 500  # holds a box through a frame or two where OCR misses it
_CATEGORY_NAMES = {"2fa": "two_fa"}


class TextDetector:
    """Use one per source. Raises DetectionError if a scan fails."""

    def __init__(self, keywords: Iterable[str] = (), reader=None, pixel_scale: int = 1) -> None:
        self._detector = Detector(keywords=keywords, reader=reader, pixel_scale=pixel_scale)

    @property
    def stats(self) -> dict:
        return self._detector.stats

    def __call__(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                x, y, w, h, _CATEGORY_NAMES.get(category, category), 1.0, TTL_MS, frame.frame_id
            )
            for x, y, w, h, category in self._detector.detect(frame.image)
        ]


detect = TextDetector()
