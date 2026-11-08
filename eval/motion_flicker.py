"""Plays window drags and pane scrolls through the real Detector and Apple OCR, frame by
frame, and measures how the black boxes behave: secret glyphs left showing, black away
from any secret, and black that blinks, on or off for only a few frames.

    python -m eval.motion_flicker
    python -m eval.motion_flicker --scene notes_scroll_drag

Grading never trusts the detector: every planted secret's glyph pixels are known on every
frame. Only numbers are printed. Every secret here is fake.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.platform import AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.text_detect.detection import _UNREAD_READ_AT_AHEAD
from eclipse.types import Frame
from eval.stress_60 import Layer, render_layer

WIDTH, HEIGHT = 2400, 1500
FPS = 60
TITLE = 56
DESKTOP = (92, 64, 40)  # BGR, never black
PAD = PADDING * 2  # the compositor's padding on Retina frames, as in eval.stress_60
ALLOW = PAD + 16  # black this close to a secret counts as its cover

# All fake: documentation examples and test numbers.
FAKE = [
    ("email", "contact: ", "alex.fake@example.com"),
    ("card", "card on file ", "4242 4242 4242 4242"),
    ("phone", "support line ", "780-555-0142"),
    ("api_key", "key ", "AKIAIOSFODNN7EXAMPLE"),
    ("email", "billing: ", "pat.sample@example.org"),
]
FILLER = [
    "Meeting notes for the Friday review",
    "    the panel keeps the last safe frame",
    "Lunch is at noon in the big room",
    "    rows move with the scroll offset",
    "Ship the build after the demo",
]


def lines(rows: int, every: int = 4, seed: int = 0, dupes: bool = False) -> list:
    out = []
    for row in range(rows):
        if row % every == every // 2:
            index = 0 if dupes else (row // every + seed) % len(FAKE)
            name, prefix, value = FAKE[index]
            out.append((prefix, (name, value)))
        else:
            out.append((FILLER[(row + seed) % len(FILLER)], None))
    return out


@dataclass
class Window:
    """A window: a title bar over a view of a page, which may be taller than the view."""

    page: Layer
    width: int
    height: int  # of the whole window, title bar included

    @classmethod
    def make(cls, rows, width=1300, height=820, **kwargs) -> "Window":
        page = render_layer(lines(rows, **kwargs), width, max(rows * 40 + 32, height))
        return cls(page, width, height)

    def draw(self, image, x: int, y: int, offset: int = 0) -> list[tuple]:
        """Draw at (x, y), scrolled down by offset; return its secrets as the screen shows them."""
        view = self.height - TITLE
        window = np.empty((self.height, self.width, 3), np.uint8)
        window[:TITLE] = (60, 56, 56)
        window[TITLE:] = self.page.image[offset : offset + view, : self.width]
        _paste(image, window, x, y)
        secrets = []
        for s in self.page.secrets:
            top = TITLE + s.y - offset  # in the window
            rows = slice(max(TITLE - top, 0), min(view + TITLE - top, s.mask.shape[0]))
            if rows.stop <= rows.start:
                continue
            mask = s.mask[rows]
            sx, sy = x + s.x, y + top + rows.start
            clip = _clip(sx, sy, mask, image.shape)
            if clip is not None:
                # the unclipped origin, so a cover can be compared with the text it follows
                secrets.append((s.name, *clip, (id(s), sx - clip[0], sy - clip[1])))
        return secrets


def _paste(image, part, x, y) -> None:
    height, width = image.shape[:2]
    top, left = max(y, 0), max(x, 0)
    bottom, right = min(y + part.shape[0], height), min(x + part.shape[1], width)
    if bottom > top and right > left:
        image[top:bottom, left:right] = part[top - y : bottom - y, left - x : right - x]


def _clip(x, y, mask, shape):
    height, width = shape[:2]
    top, left = max(y, 0), max(x, 0)
    bottom, right = min(y + mask.shape[0], height), min(x + mask.shape[1], width)
    if bottom <= top or right <= left:
        return None
    part = mask[top - y : bottom - y, left - x : right - x]
    return (left, top, part) if part.any() else None


def _desktop(width=WIDTH, height=HEIGHT) -> np.ndarray:
    return render_layer(
        [(f"Desktop label {row}", None) for row in range(height // 40)], width, height,
        background=DESKTOP, left=40,
    ).image


def _ramp(rest: int, move: int, step) -> list:
    """Rest, move out, rest, move back, rest: one offset per frame."""
    out, at = [], np.zeros(2)
    out += [tuple(at)] * rest
    for _ in range(move):
        at = at + step
        out.append(tuple(at))
    out += [tuple(at)] * rest
    for _ in range(move):
        at = at - step
        out.append(tuple(at))
    out += [tuple(at)] * rest
    return [(int(dx), int(dy)) for dx, dy in out]


class Scene:
    """frame(index) gives (image, secrets) with secrets as (name, x, y, glyph mask)."""

    def __init__(self, name, frames, draw, shape=(HEIGHT, WIDTH)):
        self.name, self.frames, self._draw, self.shape = name, frames, draw, shape

    def frame(self, index):
        return self._draw(index)


def _drag(name, step, rest=16, move=24) -> Scene:
    window = Window.make(18)
    desktop = _desktop()
    path = _ramp(rest, move, np.array(step))

    def draw(index):
        image = desktop.copy()
        dx, dy = path[index]
        return image, window.draw(image, 300 + dx, 200 + dy)

    return Scene(name, len(path), draw)


def _reversals() -> Scene:
    window = Window.make(18)
    desktop = _desktop()
    path, x = [], 0
    for _ in range(12):
        path.append(x)
    for leg in range(6):
        for _ in range(6):
            x += 22 if leg % 2 == 0 else -22
            path.append(x)
    path += [x] * 16

    def draw(index):
        image = desktop.copy()
        return image, window.draw(image, 300 + path[index], 220 + path[index] // 3)

    return Scene("reversals", len(path), draw)


def _scroll() -> Scene:
    window = Window.make(80, height=1100)
    desktop = _desktop()
    offsets = [abs(dy) for _, dy in _ramp(16, 30, np.array((0, 14)))]

    def draw(index):
        image = desktop.copy()
        return image, window.draw(image, 300, 150, offsets[index])

    return Scene("scroll", len(offsets), draw)


def _two_windows() -> Scene:
    """One window's pane scrolls while another window is dragged sideways, both at once."""
    left = Window.make(80, width=1000, height=1100, seed=1)
    right = Window.make(18, width=900, height=700, seed=3)
    desktop = _desktop()
    offsets = [abs(dy) for _, dy in _ramp(12, 28, np.array((0, 12)))]
    drags = _ramp(12, 28, np.array((-14, 6)))

    def draw(index):
        image = desktop.copy()
        secrets = left.draw(image, 80, 150, offsets[index])
        dx, dy = drags[index]
        secrets += right.draw(image, 1400 + dx, 300 + dy)
        return image, secrets

    return Scene("two_windows", len(offsets), draw)


def _duplicates() -> Scene:
    """The same fake email on several lines, dragged diagonally and back."""
    window = Window.make(18, dupes=True)
    desktop = _desktop()
    path = _ramp(16, 24, np.array((18, 10)))

    def draw(index):
        image = desktop.copy()
        dx, dy = path[index]
        return image, window.draw(image, 300 + dx, 200 + dy)

    return Scene("duplicates", len(path), draw)


def _edit_in_place() -> Scene:
    """A still window where a line's secret is replaced every 12 frames, then cleared."""
    pages = [Window.make(18, seed=seed) for seed in range(4)]
    desktop = _desktop()

    def draw(index):
        image = desktop.copy()
        return image, pages[index // 12 % len(pages)].draw(image, 300, 200)

    return Scene("edit", 72, draw)


def _clip_off() -> Scene:
    """Dragged most of the way off the right and bottom edges, then back."""
    window = Window.make(18)
    desktop = _desktop()
    path = _ramp(12, 30, np.array((40, 22)))

    def draw(index):
        image = desktop.copy()
        dx, dy = path[index]
        return image, window.draw(image, 900 + dx, 500 + dy)

    return Scene("clip", len(path), draw)


def _vanish() -> Scene:
    """A window shown, closed, then opened elsewhere."""
    window = Window.make(18)
    desktop = _desktop()

    def draw(index):
        image = desktop.copy()
        if index < 24:
            return image, window.draw(image, 300, 200)
        if index < 48:
            return image, []
        return image, window.draw(image, 900, 500)

    return Scene("vanish", 72, draw)


def _resize() -> Scene:
    """The source changes size halfway, as when a capture switches displays."""
    window = Window.make(18)
    big, small = _desktop(), _desktop(2000, 1300)

    def draw(index):
        image = (big if index < 30 else small).copy()
        return image, window.draw(image, 300 + (index % 30) * 6, 200)

    return Scene("resize", 60, draw)


NOTES_FONT = "/System/Library/Fonts/Helvetica.ttc"
NOTES_LINES = [  # (points at Retina scale, text, fake secret or None)
    (30, "ECLIPSE TEST PAGE - all secrets below are fake", None),
    (26, "Email: ", ("email", "jordan.test@example.net")),
    (26, "Phone ", ("phone", "(780) 555-0199")),
    (26, "Card ", ("card", "4000 0566 5566 5556")),
    (26, "Your code is ", ("two_fa", "731904")),
    (26, "Meeting at 4512 Oak Street on Friday", None),
    (17, "small: ", ("email", "tiny.fake@example.com")),
    (17, "backup ", ("phone", "780-555-0123")),
    (18, "key: ", ("api_key", "sk-FAKEFAKEFAKE0000000000")),
    (26, "The quarterly notes are due on Friday.", None),
    (26, "Server runs on port 8080", None),
    (26, "contact ", ("email", "morgan.sample@example.org")),
    (26, "Lunch is at noon in the big room", None),
]


def _notes_layer(width=1300, height=820, repeat=1) -> Layer:
    """A Notes-like page: a proportional font, mixed sizes, 8-9 pt lines among 13 pt ones."""
    from PIL import Image, ImageDraw, ImageFont
    from eval.stress_60 import TEXT, TEXT_CONTRAST, Planted, _to_bgr

    background = (33, 30, 30)
    canvas = Image.new("RGB", (width, height), background[::-1])
    draw = ImageDraw.Draw(canvas)
    sidebar = 300
    draw.rectangle((0, 0, sidebar, height), fill=(48, 44, 44))
    for row in range(10):
        draw.text((24, 90 + row * 64), f"Note {row} title", font=ImageFont.truetype(NOTES_FONT, 24),
                  fill=(200, 200, 200))
    places, y = [], 30
    for size, text, secret in NOTES_LINES * repeat:
        font = ImageFont.truetype(NOTES_FONT, size)
        x = sidebar + 40
        draw.text((x, y), text, font=font, fill=TEXT[::-1])
        if secret is not None:
            name, value = secret
            sx = x + int(font.getlength(text))
            draw.text((sx, y), value, font=font, fill=TEXT[::-1])
            places.append((name, sx, y, int(font.getlength(value)) + 2, int(size * 1.4)))
        y += int(size * 1.6)
    image = _to_bgr(canvas)
    planted = []
    for name, x, y, length, tall in places:
        crop = image[y : y + tall, x : x + length].astype(int)
        mask = np.abs(crop - np.array(background)).max(axis=2) > TEXT_CONTRAST
        planted.append(Planted(name, x, y, mask))
    return Layer(image, tuple(planted))


def _notes_drag(name, legs) -> Scene:
    """legs: (frames, dx, dy[, scroll]) per frame of each leg, so drags can turn, reverse
    and stop, and the note can scroll inside the window while it moves."""
    window = Window(_notes_layer(height=2400, repeat=3), 1300, 820 + TITLE)
    desktop = _desktop()
    path, at = [], (0, 0, 0)
    for frames, dx, dy, *scroll in legs:
        for _ in range(frames):
            at = (at[0] + dx, at[1] + dy, at[2] + (scroll[0] if scroll else 0))
            path.append(at)

    def draw(index):
        image = desktop.copy()
        dx, dy, offset = path[index]
        return image, window.draw(image, 400 + dx, 250 + dy, offset)

    return Scene(name, len(path), draw)


def _momentum(legs) -> list[int]:
    """Scroll offsets that speed up and coast to a stop, as a trackpad flick does.
    legs: (frames, peak rows per frame), negative peaks scrolling back up."""
    out, at = [], 0
    for frames, peak in legs:
        for frame in range(frames):
            at = max(at + round(peak * (1 - frame / frames) ** 2) // 2 * 2, 0)
            out.append(at)
    return out


def _notes_beside_obs(name="notes_obs", legs=((30, 44), (14, 0), (24, -36), (10, 0), (30, 60), (14, 0))) -> Scene:
    """A Notes pane scrolled with flicks while OBS sits beside it, as in a live test.
    OBS's preview mirrors the screen, so it changes on every frame the pane scrolls,
    and its audio meter moves on its own. Neither is part of the scroll."""
    window = Window(_notes_layer(height=2400, repeat=3), 1300, 820 + TITLE)
    desktop = _desktop()
    offsets = [0] * 10 + _momentum(legs)
    preview = (1640, 180, 720, 450)  # x, y, width, height of OBS's mirror of the screen
    meter = (1580, 1050, 26, 300)
    rng = np.random.default_rng(7)
    last = desktop.copy()

    def draw(index):
        nonlocal last
        image = desktop.copy()
        secrets = window.draw(image, 200, 160, offsets[index])
        x, y, w, h = preview
        image[y : y + h, x : x + w] = cv2.resize(last, (w, h), interpolation=cv2.INTER_AREA)
        x, y, w, h = meter
        level = int(h * rng.uniform(0.2, 0.9))
        image[y : y + h, x : x + w] = (40, 40, 40)
        image[y + h - level : y + h, x : x + w] = (60, 200, 60)
        last = image.copy()
        return image, secrets

    return Scene(name, len(offsets), draw)


def _preview(image, index) -> None:
    """A box beside the windows that changes every frame, like a live OBS preview."""
    box = image[150:750, 1700:2350]
    box[:] = (30, 30, 30)
    for row in range(14):
        y = 20 + row * 40
        x = 10 + (index * 7 + row * 13) % 120
        cv2.putText(box, "preview line of moving text", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)


def _notes_scroll(name, speeds, preview=False) -> Scene:
    """A still Notes window whose note scrolls by speeds[i] rows on frame i."""
    window = Window(_notes_layer(height=2400, repeat=3), 1300, 820 + TITLE)
    desktop = _desktop()
    offsets, at = [], 0
    for speed in speeds:
        at = max(at + speed, 0)
        offsets.append(at)

    def draw(index):
        image = desktop.copy()
        secrets = window.draw(image, 300, 250, offsets[index])
        if preview:
            _preview(image, index)
        return image, secrets

    return Scene(name, len(offsets), draw)


_SCROLL_SPEEDS = [0] * 8 + [6, 12, 20, 28, 34, 34, 28, 20, 12, 6, 3, 1] + [0] * 8 + [-14] * 12 + [0] * 10


SCENES = {
    "notes_obs": _notes_beside_obs,
    "notes_obs_slow": lambda: _notes_beside_obs("notes_obs_slow", ((60, 12), (20, 0), (60, -12), (20, 0))),
    "notes_scroll": lambda: _notes_scroll("notes_scroll", _SCROLL_SPEEDS),
    "notes_scroll_preview": lambda: _notes_scroll("notes_scroll_preview", _SCROLL_SPEEDS, True),
    "notes_drag": lambda: _notes_drag("notes_drag", [
        (12, 0, 0), (20, 18, 10), (10, 0, 0), (12, -30, 4), (6, 0, 0), (10, 26, -22), (16, 0, 0),
    ]),
    "notes_scroll_drag": lambda: _notes_drag("notes_scroll_drag", [
        (8, 0, 0, 0), (20, 16, 10, 0), (16, 0, 0, 14), (20, 14, -8, 12), (20, 0, 0, 0),
    ]),
    "notes_clip": lambda: _notes_drag("notes_clip", [
        (8, 0, 0), (24, 40, 22), (12, 0, 0), (24, -40, -22), (12, 0, 0),
    ]),
    "h_drag": lambda: _drag("h_drag", (20, 0)),
    "v_drag": lambda: _drag("v_drag", (0, 14)),
    "d_drag": lambda: _drag("d_drag", (20, 12)),
    "slow_d_drag": lambda: _drag("slow_d_drag", (3, 2), move=40),
    "fast_h_drag": lambda: _drag("fast_h_drag", (64, 0), move=10),
    "fast_d_drag": lambda: _drag("fast_d_drag", (56, 34), move=10),
    "reversals": _reversals,
    "scroll": _scroll,
    "two_windows": _two_windows,
    "duplicates": _duplicates,
    "edit": _edit_in_place,
    "clip": _clip_off,
    "vanish": _vanish,
    "resize": _resize,
}





EDGE = 3  # black that moved this little is a cover edge wobbling, not a blink


def _changed_past_edge(now: np.ndarray, before: np.ndarray) -> int:
    """Black that appeared or vanished more than EDGE pixels from where black was."""
    kernel = np.ones((2 * EDGE + 1, 2 * EDGE + 1), np.uint8)
    grown_now = cv2.dilate(now.astype(np.uint8), kernel) > 0
    grown_before = cv2.dilate(before.astype(np.uint8), kernel) > 0
    return int((now & ~grown_before).sum() + (before & ~grown_now).sum())


def _outputs(scene: Scene, look_ahead: int, scan_ms: list):
    """Each frame's image, secrets and output, as the pipeline would send them. With
    look_ahead, a frame is drawn only after that many later frames have been scanned."""
    detect = TextDetector(reader=AppleVisionReader(), pixel_scale=2)
    if look_ahead:
        detect.look_ahead()
    # Live runs read ahead in workers, which makes incoming scroll rows read every frame.
    # Read in the scan here, so set that the same way.
    detect._detector._unread_read_at = _UNREAD_READ_AT_AHEAD * 2
    compositor = Compositor(padding=PAD)
    waiting = []
    for index in range(scene.frames + look_ahead):
        if index < scene.frames:
            image, secrets = scene.frame(index)
            frame = Frame(image, time.monotonic(), "screen", index)
            started = time.monotonic()
            found = detect(frame)
            scan_ms.append((time.monotonic() - started) * 1000)
            waiting.append((frame, secrets, found))
        if len(waiting) > look_ahead or index >= scene.frames:
            frame, secrets, found = waiting.pop(0)
            settled = detect.settled(frame) if look_ahead else None
            boxes = found if settled is None else settled
            output = compositor.process(frame, lambda _: boxes)
            held = np.zeros(frame.image.shape[:2], bool)
            for d in boxes:
                if d.category == "held":
                    held[max(d.y, 0) : d.y + d.h, max(d.x, 0) : d.x + d.w] = True
            yield frame.image, secrets, output, held


def _scored(image, output, held):
    """Held areas show the picture already sent, and already scored, in an earlier frame.
    Scored again here, old text landing where a secret is now would count as shown."""
    if not held.any():
        return image, output
    output = output.copy()
    output[held] = image[held] ^ 1  # never equal to the frame: not shown, not black
    return image, output


def run(scene: Scene, keep: bool = False, look_ahead: int = 0) -> dict:
    exposed, jitter, stray_area, stray_flicker, scan_ms = [], [], [], [], []
    held_frames, held_run, held_longest = 0, 0, 0
    before_rel, before_stray = {}, None
    for image, secrets, output, held in _outputs(scene, look_ahead, scan_ms):
        if held.any():
            held_frames += 1
            held_run += 1
            held_longest = max(held_longest, held_run)
        else:
            held_run = 0
        image, output = _scored(image, output, held)
        is_black = (output == 0).all(axis=2) & ~(image == 0).all(axis=2)
        height, width = is_black.shape
        near = np.zeros_like(is_black)
        shown, rel = 0, {}
        for _, x, y, mask, (key, ox, oy) in secrets:
            h, w = mask.shape
            shown += int((mask & (output[y : y + h, x : x + w] == image[y : y + h, x : x + w]).all(axis=2)).sum())
            near[max(y - ALLOW, 0) : y + h + ALLOW, max(x - ALLOW, 0) : x + w + ALLOW] = True
            # the black around this secret, in the secret's own coordinates (whole text, unclipped)
            top, left = y + oy - ALLOW, x + ox - ALLOW
            box = np.zeros((h - oy + 2 * ALLOW + 40, w - ox + 2 * ALLOW + 200), bool)
            t, l = max(top, 0), max(left, 0)
            b, r = min(top + box.shape[0], height), min(left + box.shape[1], width)
            box[t - top : b - top, l - left : r - left] = is_black[t:b, l:r]
            rel[key] = box
        exposed.append(shown)
        jitter.append(sum(
            _changed_past_edge(rel[k], before_rel[k])
            for k in rel if k in before_rel and rel[k].shape == before_rel[k].shape
        ))
        stray = is_black & ~near
        stray_area.append(int(stray.sum()))
        stray_flicker.append(0 if before_stray is None else _changed_past_edge(stray, before_stray))
        before_rel, before_stray = rel, stray
    result = {
        "scene": scene.name,
        "frames": scene.frames,
        "exposed_glyph_px": int(sum(exposed)),
        "frames_exposed": int(sum(1 for e in exposed if e)),
        "cover_jitter_px": int(sum(jitter)),  # black around secrets changing shape, past edge wobble
        "frames_jitter": int(sum(1 for j in jitter if j > 400)),
        "stray_flicker_px": int(sum(stray_flicker)),  # black away from secrets on or off, past wobble
        "stray_black_mean_px": int(np.mean(stray_area)),
        "scan_ms_mean": round(float(np.mean(scan_ms)), 1),
        "held_pane_frames": held_frames,  # panes showing an earlier picture: a stutter
        "longest_hold": held_longest,
    }
    if keep:
        result["per_frame"] = list(zip(range(len(exposed)), exposed, jitter, stray_flicker))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="motion_flicker")
    parser.add_argument("--scene", choices=[*SCENES, "all"], default="all", nargs="?")
    parser.add_argument("--scenes", help="comma-separated scene names")
    parser.add_argument("--frames", action="store_true", help="print every frame's numbers")
    parser.add_argument(
        "--look-ahead", type=int, default=0,
        help="frames scanned before each frame is drawn, as the pipeline's look-ahead does",
    )
    args = parser.parse_args(argv)
    names = args.scenes.split(",") if args.scenes else (list(SCENES) if args.scene == "all" else [args.scene])
    for name in names:
        print(json.dumps(run(SCENES[name](), args.frames, args.look_ahead)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
