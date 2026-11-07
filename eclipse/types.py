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
