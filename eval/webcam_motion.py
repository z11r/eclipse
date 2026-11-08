"""Moves a held sheet with a secret on it through a noisy camera picture, frame by frame,
and measures how the covers behave: secret glyphs left showing, black that jumps around
the secret (jitter), and black away from it (swelling).

    python -m eval.webcam_motion
    python -m eval.webcam_motion --speed 30 --miss 0.3 --compare

The reader is a stand-in for OCR that knows where every line is, but misses reads the way
OCR does on a camera: at random, and always while the sheet is blurred by fast motion.
That isolates what the detector does with its covers, which is what this measures. It
runs anywhere, no Apple OCR needed. Grading never trusts the detector: every glyph pixel
of the secret is known on every frame. Every secret here is fake.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.main import WEBCAM_NOISE_FLOOR
from eclipse.text_detect import TextDetector
from eclipse.text_detect.covers import CoverTracks
from eclipse.types import Frame
EDGE = 3  # black that moved this little is a cover edge wobbling, not a jump, as in motion_flicker


def _changed_past_edge(now: np.ndarray, before: np.ndarray) -> int:
    """Black that appeared or vanished more than EDGE pixels from where black was.

    The same measure as eval.motion_flicker's, which can't be imported off a Mac."""
    kernel = np.ones((2 * EDGE + 1, 2 * EDGE + 1), np.uint8)
    grown_now = cv2.dilate(now.astype(np.uint8), kernel) > 0
    grown_before = cv2.dilate(before.astype(np.uint8), kernel) > 0
    return int((now & ~grown_before).sum() + (before & ~grown_now).sum())


WIDTH, HEIGHT = 1280, 720
FPS = 30
ALLOW = PADDING + 16  # black this close to the secret counts as its cover
LINES = [
    ("Shopping list for Saturday", False),
    ("alex.fake@example.com", True),
    ("bring the blue folder", False),
]
FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
SHEET_W, SHEET_H, LINE_GAP = 460, 190, 50


def _sheet():
    """The sheet, each line's box on it, and the secret's glyph mask on it."""
    sheet = np.full((SHEET_H, SHEET_W, 3), 236, np.uint8)
    glyphs = np.zeros((SHEET_H, SHEET_W), bool)
    boxes = []
    for row, (text, secret) in enumerate(LINES):
        (w, h), base = cv2.getTextSize(text, FONT, SCALE, THICK)
        x, y = 24, 40 + row * LINE_GAP
        cv2.putText(sheet, text, (x, y + h), FONT, SCALE, (30, 30, 30), THICK, cv2.LINE_AA)
        boxes.append((text, (x - 2, y - 2, x + w + 2, y + h + base + 2)))
        if secret:
            ink = (sheet[y - 2 : y + h + base + 2, x - 2 : x + w + 2] < 150).all(axis=2)
            glyphs[y - 2 : y + h + base + 2, x - 2 : x + w + 2] = ink
    return sheet, boxes, glyphs


def _room(rng) -> np.ndarray:
    """A still background with some texture, like a wall and a desk."""
    room = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    room[:] = (70, 90, 110)
    room[HEIGHT * 2 // 3 :] = (40, 60, 90)
    for _ in range(40):
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        cv2.circle(room, (x, y), int(rng.integers(4, 30)), tuple(int(c) for c in rng.integers(30, 160, 3)), -1)
    return room


def _path(frames: int, speed: float) -> list[tuple[int, int, float]]:
    """Where the sheet's corner is each frame, and how far it moved since the last one.

    Held still, then waved left and right and up and down at up to speed pixels a frame,
    then still again, the way someone shows a page to a camera.
    """
    rest = FPS

    def place(index):
        t = min(max((index - rest) / (frames - 2 * rest), 0.0), 1.0)
        t = t - math.sin(2 * math.pi * t) / (2 * math.pi)  # a hand speeds up, not jumps
        # Two sideways sweeps and a slower bob.
        return math.sin(2 * math.pi * 2 * t), 0.4 * math.sin(2 * math.pi * t)

    # Scale the sweep so the fastest frame moves speed pixels.
    steps = [place(i) for i in range(frames)]
    fastest = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(steps, steps[1:]))
    span = speed / fastest
    points = [
        (int(np.clip(400 + span * sx, 0, WIDTH - SHEET_W)),
         int(np.clip(260 + span * sy, 0, HEIGHT - SHEET_H)))
        for sx, sy in steps
    ]
    out = []
    for index, (x, y) in enumerate(points):
        px, py = points[index - 1] if index else (x, y)
        out.append((x, y, math.hypot(x - px, y - py)))
    return out


class _Reader:
    """OCR that knows the truth but misses reads, as OCR on a camera does."""

    def __init__(self, rng, miss: float, blur_speed: float):
        self.rng, self.miss, self.blur_speed = rng, miss, blur_speed
        self.lines: list[tuple[str, tuple]] = []  # text and frame box, this frame
        self.speed = 0.0

    def read_area(self, image, top, left):
        height, width = image.shape[:2]
        found = []
        for text, (x1, y1, x2, y2) in self.lines:
            inside = x1 >= left and y1 >= top and x2 <= left + width and y2 <= top + height
            if not inside or self.speed > self.blur_speed or self.rng.random() < self.miss:
                continue
            box = [[x1 - left, y1 - top], [x2 - left, y1 - top], [x2 - left, y2 - top], [x1 - left, y2 - top]]
            found.append((box, text, 1.0))
        return found, None


def run(speed: float, miss: float, frames: int, follow: bool, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    room = _room(rng)
    sheet, boxes, glyphs = _sheet()
    blur_speed = 24.0  # past this, motion blur smears the text past reading and matching
    reader = _Reader(np.random.default_rng(seed + 1), miss, blur_speed)
    detect = TextDetector(reader=reader, noise_floor=WEBCAM_NOISE_FLOOR)
    tracks = detect._detector._tracks
    if not follow:
        tracks.follow_each = lambda image, changed: False
    timed = []
    real_follow = CoverTracks.follow_each

    def timed_follow(image, changed):
        started = time.perf_counter()
        result = real_follow(tracks, image, changed)
        timed.append((time.perf_counter() - started) * 1000)
        return result

    if follow:
        tracks.follow_each = timed_follow
    compositor = Compositor()
    exposed, jitter, stray, scan_ms = [], [], [], []
    before_rel, before_stray = None, None
    for index, (x, y, moved) in enumerate(_path(frames, speed)):
        image = room.copy()
        part = sheet
        if moved > 4:  # smear along the motion, as a camera's exposure does
            length = max(int(moved / 3), 1)
            part = cv2.blur(sheet, (length, 1))
        image[y : y + SHEET_H, x : x + SHEET_W] = part
        noise = rng.integers(-6, 7, image.shape)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        reader.lines = [(t, (bx1 + x, by1 + y, bx2 + x, by2 + y)) for t, (bx1, by1, bx2, by2) in boxes]
        reader.speed = moved
        started = time.monotonic()
        output = compositor.process(Frame(image, time.monotonic(), "webcam", index), detect)
        scan_ms.append((time.monotonic() - started) * 1000)
        black = (output == 0).all(axis=2)
        shown_here = black[y : y + SHEET_H, x : x + SHEET_W]
        exposed.append(int((glyphs & ~shown_here).sum()))
        # The black around the secret, in the sheet's own coordinates.
        sx1, sy1, sx2, sy2 = boxes[1][1]
        top, left = y + sy1 - ALLOW, x + sx1 - ALLOW
        rel = np.zeros((sy2 - sy1 + 2 * ALLOW, sx2 - sx1 + 2 * ALLOW), bool)
        t, l = max(top, 0), max(left, 0)
        b, r = min(top + rel.shape[0], HEIGHT), min(left + rel.shape[1], WIDTH)
        rel[t - top : b - top, l - left : r - left] = black[t:b, l:r]
        if before_rel is not None:
            jitter.append(_changed_past_edge(rel, before_rel))
        near = np.zeros_like(black)
        near[t:b, l:r] = True
        away = black & ~near
        stray.append(int(away.sum()))
        before_rel, before_stray = rel, away
    return {
        "follow": follow,
        "speed": speed,
        "miss": miss,
        "frames": frames,
        "exposed_glyph_px": int(sum(exposed)),
        "frames_exposed": int(sum(1 for e in exposed if e)),
        "cover_jitter_px": int(sum(jitter)),
        "stray_black_mean_px": int(np.mean(stray)),
        "scan_ms_mean": round(float(np.mean(scan_ms)), 2),
        "follow_ms_mean": round(float(np.mean(timed)), 2) if timed else None,
        "follow_ms_max": round(float(np.max(timed)), 2) if timed else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="webcam_motion")
    parser.add_argument("--speed", type=float, default=30, help="peak px per frame")
    parser.add_argument("--miss", type=float, default=0.3, help="share of reads OCR misses")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--compare", action="store_true", help="also run with following off")
    args = parser.parse_args(argv)
    for follow in ([False, True] if args.compare else [True]):
        print(json.dumps(run(args.speed, args.miss, args.frames, follow)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
