"""Runs hard screen activity through the real pipeline at 60 fps, with planted fake secrets.

Renders Retina-size frames, so it needs no Screen Recording and gives the same test every
time. Everything after the source is real: Vision in its worker processes, the Detector, the
compositor and the fit to the output size. Because the scene is generated, every glyph of
every planted secret is known on every frame, so a leak is counted directly in the output's
pixels instead of being inferred from OCR failing to read it.

    python -m eval.stress_60
    python -m eval.stress_60 --scenario scroll --workers 4 --seconds 10

Sustained 60 means fresh frames arrive at 60 per second. Repeats, or a longer delay, don't
count. Every secret here is fake, and nothing is written to disk except numbers.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.main import StatsLog
from eclipse.output.virtual_cam import DiscardingCam, VirtualCam
from eclipse.platform import AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.text_detect.read_ahead import OcrPool, ReadAhead

WIDTH, HEIGHT = 3024, 1964
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_SIZE = 26  # 13 pt on a Retina screen
LINE_HEIGHT = 40
TITLE_BAR = 112  # rows at the top that never scroll
BACKGROUND = (34, 30, 30)  # BGR
TEXT = (225, 225, 225)
TEXT_CONTRAST = 40

FILLER = [
    "def build_overlay(theme, width=320):",
    "    panel = Panel(width=width, theme=theme)",
    "    for row in panel.rows():",
    "        row.render(offset=panel.scroll_position)",
    "    return panel",
    "",
    "class Compositor:",
    "    # keeps the last safe frame on screen",
    "    def process(self, frame, detect):",
    "        return self.redact(frame, detect(frame))",
    "",
]
# All fake: documentation examples and test numbers.
SECRETS = [
    ("api_key", "aws_access_key = ", "AKIAIOSFODNN7EXAMPLE"),
    ("email", "contact: ", "alex.fake@example.com"),
    ("card", "card on file ", "4242 4242 4242 4242"),
    ("phone", "support line ", "780-555-0142"),
    ("api_key", "token = ", "sk-FAKEFAKEFAKEFAKE1234567890"),
    ("two_fa", "verification code ", "482913"),
]


@dataclass(frozen=True)
class Planted:
    """One secret on a layer: its glyph pixels, at a position on that layer."""

    name: str
    x: int
    y: int
    mask: np.ndarray  # True on the secret's text pixels


@dataclass(frozen=True)
class Layer:
    image: np.ndarray
    secrets: tuple[Planted, ...]


def _font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, FONT_SIZE)


def _to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image)[:, :, ::-1].copy()


def render_layer(
    lines: list[tuple[str, tuple[str, str] | None]],
    width: int,
    height: int,
    background: tuple[int, int, int] = BACKGROUND,
    left: int = 96,
    top: int = 16,
) -> Layer:
    """Draw lines of (text, None) or (prefix, (name, secret)), recording each secret's glyphs."""
    font = _font()
    rgb = background[::-1]
    canvas = Image.new("RGB", (width, height), rgb)
    draw = ImageDraw.Draw(canvas)
    places = []
    for row, (text, secret) in enumerate(lines):
        y = top + row * LINE_HEIGHT
        if y + LINE_HEIGHT > height:
            break
        draw.text((left, y), text, font=font, fill=TEXT[::-1])
        if secret is None:
            continue
        name, value = secret
        x = left + int(font.getlength(text))
        draw.text((x, y), value, font=font, fill=TEXT[::-1])
        places.append((name, x, y, int(font.getlength(value)) + 2))
    image = _to_bgr(canvas)
    planted = []
    for name, x, y, length in places:
        crop = image[y : y + LINE_HEIGHT, x : x + length].astype(int)
        mask = np.abs(crop - np.array(background)).max(axis=2) > TEXT_CONTRAST
        planted.append(Planted(name, x, y, mask))
    return Layer(image, tuple(planted))


def page_lines(rows: int, seed: int = 0, every: int = 17) -> list:
    lines: list[tuple[str, tuple[str, str] | None]] = []
    for row in range(rows):
        if row % every == every // 2:
            name, prefix, value = SECRETS[(row // every + seed) % len(SECRETS)]
            lines.append((prefix, (name, value)))
        else:
            lines.append((FILLER[(row + seed) % len(FILLER)], None))
    return lines


def uncovered(output: np.ndarray, x: int, y: int, mask: np.ndarray, top_limit: int = 0) -> tuple:
    """(glyph pixels still showing, glyph pixels on screen) for a secret at (x, y)."""
    height, width = output.shape[:2]
    top, bottom = max(y, top_limit), min(y + mask.shape[0], height)
    left, right = max(x, 0), min(x + mask.shape[1], width)
    if bottom <= top or right <= left:
        return 0, 0
    glyphs = mask[top - y : bottom - y, left - x : right - x]
    showing = output[top:bottom, left:right].any(axis=2)  # covers are solid black
    return int((glyphs & showing).sum()), int(glyphs.sum())


class Scenario:
    name = ""

    def __init__(self, frames: int) -> None:
        self.frames = frames

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        """The picture, and (name, x, y, mask, top_limit) for each secret then on screen."""
        raise NotImplementedError


class Scroll(Scenario):
    """A page moving like a trackpad flick: speeding up, gliding, then easing to a stop."""

    name = "scroll"

    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        rows = frames * 30 // LINE_HEIGHT + HEIGHT // LINE_HEIGHT + 4
        self.page = render_layer(page_lines(rows), WIDTH, rows * LINE_HEIGHT + 32)
        self.base = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
        cv2.rectangle(self.base, (0, 0), (WIDTH, TITLE_BAR), (50, 46, 46), -1)
        offsets, position = [], 0.0
        for index in range(frames):
            phase = index % 240
            position += min(phase * 0.5, 28.0) if phase < 180 else max(28.0 - (phase - 180), 0.0)
            offsets.append(int(position))
        self.offsets = offsets

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        offset = self.offsets[index]
        image = self.base.copy()
        image[TITLE_BAR:] = self.page.image[offset : offset + HEIGHT - TITLE_BAR]
        secrets = [
            (s.name, s.x, TITLE_BAR + s.y - offset, s.mask, TITLE_BAR)
            for s in self.page.secrets
            if -LINE_HEIGHT < TITLE_BAR + s.y - offset < HEIGHT
        ]
        return image, secrets


class Typing(Scenario):
    """Typing at 14 keys a second with a blinking caret, ending each run with a secret."""

    name = "typing"
    PER_SECOND = 14
    PREFIX = "export TOKEN="
    VALUE = "sk-FAKEFAKEFAKEFAKE1234567890"
    ROW = 22

    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        self.page = render_layer([(line, None) for line, _ in page_lines(46)], WIDTH, HEIGHT)
        self.font = _font()
        self.char = self.font.getlength("M")
        self.left, self.top = 96, 16 + self.ROW * LINE_HEIGHT
        self.page.image[self.top : self.top + LINE_HEIGHT] = BACKGROUND
        full = self.PREFIX + self.VALUE
        line = Image.new("RGB", (WIDTH, LINE_HEIGHT), BACKGROUND[::-1])
        ImageDraw.Draw(line).text((0, 0), full, font=self.font, fill=TEXT[::-1])
        self.line = _to_bgr(line)
        self.length = len(full)
        start = int(len(self.PREFIX) * self.char)
        secret = self.line[:, start : int(self.length * self.char) + 2].astype(int)
        self.mask = np.abs(secret - np.array(BACKGROUND)).max(axis=2) > TEXT_CONTRAST
        self.secret_x = self.left + start

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        cycle = index % (self.frames // 2 or 1)
        typed = min(int(cycle * self.PER_SECOND / 60), self.length)
        image = self.page.image.copy()
        width = int(typed * self.char)
        if width:
            image[self.top : self.top + LINE_HEIGHT, self.left : self.left + width] = self.line[
                :, :width
            ]
        if index // 30 % 2 == 0:
            caret = self.left + width
            image[self.top + 4 : self.top + 34, caret + 2 : caret + 6] = TEXT
        # A secret can only be recognised once all of it is on screen.
        secrets = []
        if typed == self.length:
            secrets.append(("api_key", self.secret_x, self.top, self.mask, 0))
        return image, secrets


class Notification(Scenario):
    """A banner sliding in over a page with a fake code on it, resting, then sliding out."""

    name = "notification"
    SLIDE = 20
    REST = 150
    PAUSE = 40

    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        self.page = render_layer([(line, None) for line, _ in page_lines(46)], WIDTH, HEIGHT)
        self.banner = render_layer(
            [
                ("Bank alert", None),
                ("verification code ", ("two_fa", "482913")),
                ("sent to ", ("email", "alex.fake@example.com")),
            ],
            860,
            144,
            background=(64, 58, 58),
            left=28,
            top=12,
        )
        self.y = 80
        self.span = self.SLIDE * 2 + self.REST + self.PAUSE

    def _position(self, index: int) -> int | None:
        step = index % self.span
        shown = WIDTH - 900
        if step < self.SLIDE:
            eased = 1 - (1 - step / self.SLIDE) ** 3
            return int(WIDTH - eased * 900)
        if step < self.SLIDE + self.REST:
            return shown
        if step < self.SLIDE * 2 + self.REST:
            eased = ((step - self.SLIDE - self.REST) / self.SLIDE) ** 3
            return int(shown + eased * 900)
        return None

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        image = self.page.image.copy()
        x = self._position(index)
        if x is None:
            return image, []
        tile = self.banner.image
        visible = min(tile.shape[1], WIDTH - x)
        image[self.y : self.y + tile.shape[0], x : x + visible] = tile[:, :visible]
        secrets = [(s.name, x + s.x, self.y + s.y, s.mask, 0) for s in self.banner.secrets]
        return image, secrets


class WindowChange(Scenario):
    """Switching between two full-screen windows, so every pixel moves during the slide."""

    name = "window"
    SLIDE = 18
    REST = 102

    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        self.pages = [
            render_layer(page_lines(48, seed=seed, every=9), WIDTH, HEIGHT) for seed in (1, 4)
        ]
        self.span = self.SLIDE + self.REST

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        step = index % self.span
        current = self.pages[(index // self.span) % 2]
        following = self.pages[(index // self.span + 1) % 2]
        if step < self.REST:
            secrets = [(s.name, s.x, s.y, s.mask, 0) for s in current.secrets]
            return current.image.copy(), secrets
        eased = (step - self.REST + 1) / self.SLIDE
        shift = int(WIDTH * eased * eased * (3 - 2 * eased))
        image = np.empty_like(current.image)
        image[:, : WIDTH - shift] = current.image[:, shift:]
        image[:, WIDTH - shift :] = following.image[:, : shift]
        secrets = [(s.name, s.x - shift, s.y, s.mask, 0) for s in current.secrets]
        secrets += [(s.name, s.x + WIDTH - shift, s.y, s.mask, 0) for s in following.secrets]
        return image, secrets


class Video(Scenario):
    """The worst case: every pixel changes every frame, and no scroll tracking can help.

    Sideways motion of a full screen of text, so each frame needs a full read.
    """

    name = "video"
    STEP = 11

    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        self.page = render_layer(page_lines(48, seed=2, every=9), WIDTH, HEIGHT)

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        shift = index * self.STEP % WIDTH
        image = np.empty_like(self.page.image)
        image[:, : WIDTH - shift] = self.page.image[:, shift:]
        image[:, WIDTH - shift :] = self.page.image[:, :shift]
        secrets = []
        for s in self.page.secrets:
            for x in (s.x - shift, s.x - shift + WIDTH):
                secrets.append((s.name, x, s.y, s.mask, 0))
        return image, secrets


class DenseVideo(Video):
    """Beyond what any screen does for long: a full screen of long lines moving every frame.

    Exists to push the readers past capacity and show what happens then.
    """

    name = "dense"
    STEP = 11

    def __init__(self, frames: int) -> None:
        Scenario.__init__(self, frames)
        lines = []
        for prefix, secret in page_lines(48, seed=2, every=9):
            if secret is None:
                lines.append(((prefix + "   # " + " ".join(FILLER)) [:165], None))
            else:
                lines.append((prefix, secret))
        self.page = render_layer(lines, WIDTH, HEIGHT)


class AppSwitching(Scenario):
    """Switching apps three times a second, scrolling in between, under a still menu bar and Dock.

    Most of the screen changes at once, so the detector reads whole frames, while rows at
    the top and bottom never change. A real screen did this: 30 to 60 whole-frame reads
    every 2 seconds, and reads ahead covered only the rows that changed.
    """

    name = "switching"
    MENU = 74
    TOOLBAR = 110
    DOCK = 160
    EVERY = 20
    STEP = 9

    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        visible = HEIGHT - self.MENU - self.TOOLBAR - self.DOCK
        rows = (self.EVERY * self.STEP + visible) // LINE_HEIGHT + 4
        self.apps = [
            render_layer(page_lines(rows, seed=seed, every=7), WIDTH, rows * LINE_HEIGHT + 32)
            for seed in (0, 3, 5)
        ]
        self.toolbars = []
        for shade in (58, 72, 86):
            toolbar = np.full((self.TOOLBAR, WIDTH, 3), shade, np.uint8)
            toolbar[self.TOOLBAR - 2 :] = 30
            self.toolbars.append(toolbar)
        self.base = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
        self.base[: self.MENU] = (52, 48, 48)
        self.base[HEIGHT - self.DOCK :] = (64, 60, 60)

    def frame(self, index: int) -> tuple[np.ndarray, list[tuple]]:
        app = (index // self.EVERY) % len(self.apps)
        offset = (index % self.EVERY) * self.STEP
        top = self.MENU + self.TOOLBAR
        bottom = HEIGHT - self.DOCK
        image = self.base.copy()
        image[self.MENU : top] = self.toolbars[app]
        image[top:bottom] = self.apps[app].image[offset : offset + bottom - top]
        secrets = []
        for s in self.apps[app].secrets:
            y = top + s.y - offset
            # A secret partly under the Dock is hidden by it, not covered, so leave it out.
            if top - s.mask.shape[0] < y and y + s.mask.shape[0] <= bottom:
                secrets.append((s.name, s.x, y, s.mask, top))
        return image, secrets


SCENARIOS = {
    kind.name: kind for kind in (Scroll, Typing, Notification, WindowChange, AppSwitching)
}
HARDEST = {kind.name: kind for kind in (Video, DenseVideo)}


def add_spinner(image: np.ndarray, index: int) -> None:
    """A small bar that moves every frame, so every frame is a new picture."""
    x = 200 + index * 9 % 1200
    image[HEIGHT - 28 : HEIGHT - 8, x : x + 60] = (0, 200, 255)


class ScenarioSource:
    """Plays a scenario at a steady rate, as the capture system would deliver it."""

    paced = True

    def __init__(self, scenario: Scenario, fps: int) -> None:
        from eclipse.types import Frame

        self._frame_type = Frame
        self._scenario = scenario
        self._interval = 1 / fps
        self._next = 0.0
        self._index = 0
        self.truth: dict[int, list[tuple]] = {}
        self.stats = {"grab_seconds": 0.0, "convert_seconds": 0.0}

    def __enter__(self) -> "ScenarioSource":
        self._next = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        pass

    def read(self):
        time.sleep(max(self._next - time.monotonic(), 0))
        self._next = max(self._next + self._interval, time.monotonic() - self._interval)
        started = time.monotonic()
        image, secrets = self._scenario.frame(self._index % self._scenario.frames)
        add_spinner(image, self._index)
        image.flags.writeable = False
        self.stats["grab_seconds"] += time.monotonic() - started
        self.truth[self._index] = secrets
        frame = self._frame_type(image, time.monotonic(), "screen", self._index)
        self._index += 1
        return frame


class CheckingSink:
    """Checks each output for planted glyphs, then does all the real output work."""

    def __init__(self, source: ScenarioSource, fps: int, camera: bool) -> None:
        self._source = source
        self._inner = (VirtualCam if camera else DiscardingCam)(fps=fps)
        self._last = None
        self.pipeline = None
        self.visible_frames = 0  # outputs with a secret on screen
        self.leaking_frames = 0
        self.leaked_pixels = 0
        self.on_screen_pixels = 0
        self.worst_share = 0.0
        self.leaks_by_secret: dict[str, int] = {}
        self.whole_leak_frames = 0  # a secret fully on screen and not fully covered
        self.partial_leak_frames = 0  # only part of a secret was on screen, so unreadable
        self.leak_runs: list[int] = []  # frame ids with a whole secret showing
        self.whole_leaks: list[dict] = []  # exactly what showed, for the first few

    @property
    def stats(self) -> dict:
        return self._inner.stats

    def __enter__(self) -> "CheckingSink":
        self._inner.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._inner.__exit__(*exc)

    def send(self, image: np.ndarray) -> None:
        if image is not self._last:
            self._last = image
            self._check(image, self.pipeline.stats.last_sent_id)
        self._inner.send(image)

    def _check(self, image: np.ndarray, frame_id: int) -> None:
        secrets = self._source.truth.pop(frame_id, [])
        for older in [key for key in list(self._source.truth) if key < frame_id]:
            self._source.truth.pop(older, None)
        leaked = total = 0
        whole = partial = False
        for name, x, y, mask, top_limit in secrets:
            showing, on_screen = uncovered(image, x, y, mask, top_limit)
            leaked += showing
            total += on_screen
            if showing:
                self.leaks_by_secret[name] = self.leaks_by_secret.get(name, 0) + showing
                if on_screen == int(mask.sum()):
                    whole = True
                    if len(self.whole_leaks) < 12:
                        self.whole_leaks.append(
                            {
                                "frame": frame_id,
                                "secret": name,
                                "x": x,
                                "y": y,
                                "width": mask.shape[1],
                                "showing": showing,
                                "of": on_screen,
                            }
                        )
                else:
                    partial = True
        self.whole_leak_frames += whole
        self.partial_leak_frames += partial and not whole
        if whole:
            self.leak_runs.append(frame_id)
        if total:
            self.visible_frames += 1
            self.on_screen_pixels += total
        if leaked:
            self.leaking_frames += 1
            self.leaked_pixels += leaked
            self.worst_share = max(self.worst_share, leaked / total)


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * share), len(ordered) - 1)]


def run_one(scenario: Scenario, detect, delay: float, fps: int, camera: bool, log: str) -> dict:
    source = ScenarioSource(scenario, fps)
    sink = CheckingSink(source, fps, camera)
    compositor = Compositor(padding=PADDING * 2)
    pipeline = DelayedPipeline(source, sink, detect, compositor, delay, fps, scenario.frames)
    sink.pipeline = pipeline
    cpu_before = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    started = time.monotonic()
    reader = getattr(detect, "_read_ahead", None)
    work_before = reader.stats["work_seconds"] if reader else 0.0
    read_before = dict(reader.stats) if reader else {}
    with StatsLog(log, pipeline, {"capture": source, "detect": detect, "output": sink, "read": reader}):
        stats = pipeline.run()
    elapsed = time.monotonic() - started
    active = stats.capture_seconds or elapsed
    latencies = stats.latencies
    return {
        "scenario": scenario.name,
        "seconds": round(active, 1),
        "fresh_captured_fps": round(stats.fresh / active, 1),
        "fresh_sent_fps": round(stats.fresh_shown / active, 1),
        "repeats": stats.repeated,
        "dropped": stats.dropped,
        "skipped": stats.skipped,
        "id_gaps": stats.id_gaps,
        "held": stats.held,
        "latency_ms_p50": round(_percentile(latencies, 0.5)),
        "latency_ms_p95": round(_percentile(latencies, 0.95)),
        "latency_ms_max": round(_percentile(latencies, 1)),
        "frames_with_secret": sink.visible_frames,
        "frames_leaking": sink.leaking_frames,
        "frames_whole_secret_leaking": sink.whole_leak_frames,
        "frames_partial_secret_leaking": sink.partial_leak_frames,
        "whole_leak_frame_ids": sink.leak_runs[:40],
        "whole_leaks": sink.whole_leaks,
        "leaked_glyph_pixels": sink.leaked_pixels,
        "leaked_share": round(sink.leaked_pixels / max(sink.on_screen_pixels, 1), 5),
        "worst_frame_share": round(sink.worst_share, 3),
        "leaks_by_secret": sink.leaks_by_secret,
        "main_process_cores": round(
            (sum(resource.getrusage(resource.RUSAGE_SELF)[:2]) - cpu_before) / elapsed, 1
        ),
        "read_ahead": {
            name: round(reader.stats.get(name, 0) - read_before.get(name, 0))
            for name in ("hits", "misses", "shed", "cancelled")
        }
        if reader
        else {},
        "shed": stats.shed,
        "worker_cores": round(
            ((reader.stats["work_seconds"] if reader else 0.0) - work_before) / elapsed, 1
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stress_60")
    parser.add_argument("--scenario", choices=[*SCENARIOS, *HARDEST, "all"], default="all")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--frames", type=int, help="frames to play, in place of --seconds")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--workers", type=int, default=4, help="0 reads in frame order; use with a low --fps"
    )
    parser.add_argument("--camera", action="store_true", help="also send to the virtual camera")
    parser.add_argument("--log", default="eval/output/stress.log")
    parser.add_argument("--out", help="also append JSON results to this file")
    args = parser.parse_args(argv)
    cv2.setNumThreads(2)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    pool = OcrPool(args.workers) if args.workers else None
    results = []
    try:
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for name in names:
            kind = {**SCENARIOS, **HARDEST}[name]
            scenario = kind(args.frames or int(args.seconds * args.fps))
            reader = ReadAhead(pool) if pool else AppleVisionReader()
            detect = TextDetector(reader=reader, pixel_scale=2)
            result = run_one(scenario, detect, args.delay, args.fps, args.camera, args.log)
            results.append(result)
            print(json.dumps(result), flush=True)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as out:
                    out.write(json.dumps(result) + "\n")
    finally:
        if pool:
            pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
