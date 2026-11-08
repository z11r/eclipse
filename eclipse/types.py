"""The shared contract: every module talks only through Frame and Detection.

Change these fields only after agreeing with the team. Every module depends on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Source = Literal["screen", "webcam", "irl"]

Category = Literal[
    # Text layer
    "two_fa",
    "email",
    "phone",
    "api_key",
    "card",
    "keyword",
    "unread",  # text coming onto the screen, kept covered until it can be read
    "held",  # a pane not yet checked: it shows its last checked picture instead
    # Visual layer
    "street_sign",
    "house_number",
    "licence_plate",
    "mail_label",
    "window_view",
    "map",
    "storefront",
]


@dataclass(frozen=True, eq=False)
class Frame:
    """One captured image. Treat image as read-only; the compositor blurs a copy."""

    image: np.ndarray  # BGR, uint8, shape (height, width, 3)
    ts: float  # capture time in seconds, from time.monotonic()
    source: Source
    frame_id: int


@dataclass(frozen=True)
class Detection:
    """One sensitive area in a frame.

    Never put the matched text here. Detections reach the panel's log, so a raw
    secret stored on one would end up on screen.
    """

    x: int  # left edge, in pixels of the frame in frame_id
    y: int  # top edge
    w: int
    h: int
    category: Category
    confidence: float  # 0 to 1
    ttl_ms: int  # how long the blur holds after the clue was last seen
    frame_id: int  # the frame this box was found in


# scanned: OCR read this frame. cached: nothing needed reading, so earlier reads were
# reused. held: this frame's scan failed, so an older safe image stayed up. blackout:
# there was nothing safe to show.
OutputState = Literal["scanned", "cached", "held", "blackout"]


@dataclass(frozen=True, eq=False)
class Rendered:
    """One image ready for output, and what it really shows.

    Internal to the compositor, the delay buffer and the eval. A held image keeps the
    identity of the frame it came from, so what it shows can always be traced. Like a
    Detection, this must never carry screen text.
    """

    image: np.ndarray  # BGR, uint8, shape (height, width, 3)
    state: OutputState
    source: Source | None  # where the pixels came from; None for a blackout
    source_frame_id: int  # the frame the pixels came from; -1 for a blackout
    ts: float  # when those pixels were captured, on the clock of Frame.ts
    generation: int = 0  # the policy they were checked under
