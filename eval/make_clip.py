"""Generates the test clip and its labels. Run: python -m eval.make_clip

Every secret here is fake. Never add a real secret, name or address, because this
repo is public.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from eval.truth import CLIPS_DIR, LABELS_DIR, box_at, is_on_screen

NAME = "planted_secrets"
WIDTH, HEIGHT, FPS, FRAMES = 1280, 720, 30, 150
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.9
THICKNESS = 2

# (id, category, text, first frame, last frame, x, y, move per frame)
PLANTED = [
    ("still_api_key", "api_key", "AKIAIOSFODNN7EXAMPLE", 0, 149, 80, 150, (0, 0)),
    ("email", "email", "alex.fake@example.com", 30, 120, 80, 260, (0, 0)),
    ("popup_2fa", "two_fa", "482913", 60, 100, 1040, 60, (0, 0)),
    ("moving_card", "card", "4242 4242 4242 4242", 20, 140, 80, 520, (3, 0)),
    ("phone", "phone", "780-555-0142", 45, 149, 960, 660, (0, 0)),
]

FILLER = [
    (80, 340, "def build_overlay(theme):"),
    (120, 380, "panel = Panel(width=320)"),
    (120, 420, "return panel.render(theme)"),
]


def make_labels() -> dict:
    secrets = []
    for secret_id, category, text, first, last, x, y, move in PLANTED:
        (width, height), baseline = cv2.getTextSize(text, FONT, SCALE, THICKNESS)
        secrets.append(
            {
                "id": secret_id,
                "category": category,
                "text": text,
                "first_frame": first,
                "last_frame": last,
                "box": [x, y, width, height + baseline],
                "move": list(move),
            }
        )
    return {
        "clip": f"{NAME}.mp4",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frames": FRAMES,
        "secrets": secrets,
    }


def draw_frame(labels: dict, index: int) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (46, 30, 30), dtype=np.uint8)
    for x, y, text in FILLER:
        cv2.putText(image, text, (x, y), FONT, SCALE, (200, 200, 170), THICKNESS)
    cv2.putText(image, f"editor.py  {index:03d}", (80, 60), FONT, SCALE, (150, 150, 150), THICKNESS)
    for secret in labels["secrets"]:
        if is_on_screen(secret, index):
            x, y, _, height = box_at(secret, index)
            cv2.putText(
                image, secret["text"], (x, y + height - 6), FONT, SCALE, (255, 255, 255), THICKNESS
            )
    return image


def write_clip(path: Path, labels: dict) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"can't write clip: {path}")
    for index in range(FRAMES):
        writer.write(draw_frame(labels, index))
    writer.release()


def main() -> None:
    labels = make_labels()
    CLIPS_DIR.mkdir(exist_ok=True)
    LABELS_DIR.mkdir(exist_ok=True)
    write_clip(CLIPS_DIR / labels["clip"], labels)
    (LABELS_DIR / f"{NAME}.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Wrote {labels['clip']}: {FRAMES} frames, {len(labels['secrets'])} planted secrets")


if __name__ == "__main__":
    main()
