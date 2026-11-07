"""Ground truth: where each planted secret is in a clip."""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
CLIPS_DIR = EVAL_DIR / "clips"
LABELS_DIR = EVAL_DIR / "labels"


def load_labels(name: str) -> dict:
    return json.loads((LABELS_DIR / f"{name}.json").read_text())


def is_on_screen(secret: dict, frame_index: int) -> bool:
    return secret["first_frame"] <= frame_index <= secret["last_frame"]


def box_at(secret: dict, frame_index: int) -> tuple[int, int, int, int]:
    """The secret's x, y, w, h in a given frame."""
    x, y, w, h = secret["box"]
    steps = frame_index - secret["first_frame"]
    return x + secret["move"][0] * steps, y + secret["move"][1] * steps, w, h
