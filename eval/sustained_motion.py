"""Runs the real pipeline on generated motion, and checks every picture it sends.

Run from the repo root: python -m eval.sustained_motion --seconds 60

The real capture loop, detector, tracks, compositor, delay buffer, gate and controller
all run together, against fake OCR that only sees pixels. While the page scrolls, this
throws in what goes wrong on a real stream: one-frame notifications, failed OCR reads,
a panic and resume, and a new keyword. Each picture is checked as it reaches the output.

This says the pipeline's logic holds up over time. It is not a live test: no real
screen, OCR engine or OBS is involved. Fake secrets only.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.panel.controller import LivePipelineController
from eclipse.panel.schemas import ALL_CATEGORIES, PanelConfig
from eclipse.platform import peak_memory_mb
from eclipse.text_detect import TextDetector
from eclipse.types import Frame, Rendered
from eval.scroll_compare import OCR_WOBBLE, SECRETS, draw, exposed_pixels, scroll_script
from eval.scroll_scene import FAKE_PHONE, SceneReader, ScrollScene

FAKE_NOTE = "note@example.com"  # a notification that shows for a single frame
NOTE_EVERY = 97  # frames between notifications
KEYWORD = "side note 5"  # already on screen, until it becomes a keyword
_TRACKED = 256  # output images kept for the checker, enough to outlast the delay buffer
_MEMORY_LIMIT_MB = 512  # more growth than this over a run counts as a failure


class Motion:
    """What each frame looks like, and what must stay hidden in it."""

    def __init__(self) -> None:
        self.scene = ScrollScene(secrets=SECRETS, side_secrets={3: FAKE_PHONE})
        self.script = scroll_script()
        self.note = self.scene.overlay(FAKE_NOTE, 120, 100, 240)
        self.word = next(line for line in self.scene.side_lines if line.text == KEYWORD)

    def picture(self, index: int) -> tuple:
        offset, blink = self.script[index % len(self.script)]
        image = draw(self.scene, offset, blink)
        boxes = list(self.scene.secret_boxes(offset).values())
        if index % NOTE_EVERY == NOTE_EVERY - 1:
            self.scene.draw_overlay(image, self.note)
            line = self.note
            boxes.append((line.x, line.y, line.x + line.w, line.y + line.h))
        return image, boxes

    def word_box(self) -> tuple[int, int, int, int]:
        line = self.word
        return line.x, line.y, line.x + line.w, line.y + line.h


class Source:
    def __init__(self, motion: Motion) -> None:
        self.motion = motion
        self.next_id = 0

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def read(self) -> Frame:
        image, _ = self.motion.picture(self.next_id)
        frame = Frame(image, time.monotonic(), "screen", self.next_id)
        self.next_id += 1
        return frame


class FlakyReader:
    """Fake OCR that can be told to fail for a while."""

    def __init__(self, reader: SceneReader) -> None:
        self._reader = reader
        self.failing = threading.Event()

    def __call__(self, image):
        if self.failing.is_set():
            raise RuntimeError("injected OCR failure")
        return self._reader(image)


class TracingCompositor(Compositor):
    """Remembers what each output image is, so the checker can tell what it was sent."""

    def __init__(self, padding: int) -> None:
        super().__init__(padding)
        self.images: OrderedDict[int, Rendered] = OrderedDict()
        self._lock = threading.Lock()

    def render(self, frame: Frame, detect, generation: int = 0) -> Rendered:
        rendered = super().render(frame, detect, generation)
        with self._lock:
            self.images[id(rendered.image)] = rendered
            if len(self.images) > _TRACKED:
                self.images.popitem(last=False)
        return rendered

    def lookup(self, image) -> Rendered | None:
        with self._lock:
            return self.images.get(id(image))


class CheckingSink:
    """Stands in for the camera, at the camera's rate, and checks each new picture."""

    def __init__(self, motion: Motion, compositor: TracingCompositor, fps: int) -> None:
        self.motion = motion
        self.compositor = compositor
        self.interval = 1 / fps
        self.due = 0.0
        self.last = None
        self.sends = self.checked = self.black = self.exposed = 0
        self.keyword_exposed = self.frames_before_resume = self.not_ours = 0
        self.panic_at: float | None = None
        self.panic_to_black: float | None = None
        self.keyword_from: float | None = None  # time the keyword was added
        self.resumed_at: float | None = None
        self.final_was_black = False

    def __enter__(self) -> "CheckingSink":
        self.due = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        pass

    def send(self, image) -> None:
        self.sends += 1
        now = time.monotonic()
        black = not image.any()
        self.final_was_black = black
        if black:
            self.black += 1
            if self.panic_at is not None and self.panic_to_black is None:
                self.panic_to_black = now - self.panic_at
        elif image is not self.last:
            self._check(image, now)
        self.last = image
        self.due = max(self.due + self.interval, time.monotonic())
        time.sleep(max(self.due - time.monotonic(), 0))

    def _check(self, image, now: float) -> None:
        rendered = self.compositor.lookup(image)
        if rendered is None:
            self.not_ours += 1  # only the compositor's own images may reach the output
            return
        self.checked += 1
        frame_id = rendered.source_frame_id
        raw, boxes = self.motion.picture(frame_id)
        self.exposed += exposed_pixels(raw, image, boxes)
        if self.keyword_from is not None and now >= self.keyword_from:
            self.keyword_exposed += exposed_pixels(raw, image, [self.motion.word_box()])
        if self.resumed_at is not None and rendered.ts < self.resumed_at:
            self.frames_before_resume += 1


def run(
    seconds: float,
    fps: int = 60,
    delay: float = 0.25,
    config_dir: Path | None = None,
    blind_to: tuple[str, ...] = (),
    add_keyword: bool = True,
) -> dict:
    """Run for this long. blind_to and add_keyword=False sabotage a run on purpose.

    A checker that can't fail proves nothing, so the tests make sure these get caught.
    """
    motion = Motion()
    plain_reader = SceneReader(motion.scene, wobble=OCR_WOBBLE)
    plain_reader.blind.update(blind_to)
    reader = FlakyReader(plain_reader)
    detect = TextDetector(reader=reader)
    compositor = TracingCompositor(PADDING)
    sink = CheckingSink(motion, compositor, fps)
    pipeline = DelayedPipeline(Source(motion), sink, detect, compositor, delay, fps)
    with tempfile.TemporaryDirectory() as folder:
        path = (config_dir or Path(folder)) / "config.json"
        controller = LivePipelineController(pipeline, detect.configure, path)
        worker = threading.Thread(target=pipeline.run, name="eclipse-output", daemon=True)
        started = time.monotonic()
        worker.start()

        def wait_until_share(share: float) -> None:
            time.sleep(max(started + seconds * share - time.monotonic(), 0))

        wait_until_share(0.2)
        # Measured from here, as the buffers and this checker's own images are full by now.
        # A leak goes on growing past this point. A working set doesn't.
        memory_warmed_up = peak_memory_mb()
        reader.failing.set()
        wait_until_share(0.3)
        reader.failing.clear()

        wait_until_share(0.45)
        sink.panic_at = time.monotonic()
        panic_state = controller.panic()
        wait_until_share(0.55)
        sink.resumed_at = time.monotonic()
        controller.resume()

        wait_until_share(0.75)
        sink.keyword_from = time.monotonic()
        if add_keyword:
            controller.set_config(PanelConfig(enabled=list(ALL_CATEGORIES), keywords=[KEYWORD]))

        wait_until_share(1.0)
        pipeline._stop.set()
        worker.join(timeout=10)
        ran_for = time.monotonic() - started

    stats = pipeline.stats
    return {
        "seconds": round(ran_for, 2),
        "fps_asked": fps,
        "delay_s": delay,
        "captured": stats.captured,
        "captured_fps": round(stats.captured / ran_for, 1),
        "camera_sends": sink.sends,
        "pictures_checked": sink.checked,
        "black_sends": sink.black,
        "exposed_secret_pixels": sink.exposed,
        "exposed_keyword_pixels": sink.keyword_exposed,
        "pictures_from_before_resume": sink.frames_before_resume,
        "pictures_not_from_the_compositor": sink.not_ours,
        "ended_on_black": sink.final_was_black,
        "panic_reported_black": bool(panic_state.blackout_confirmed),
        "panic_to_black_ms": round((sink.panic_to_black or 0) * 1000, 1),
        "ocr_failures_held": stats.rendered_held + stats.rendered_blackout,
        "invalidated": stats.invalidated,
        "scans_read": stats.rendered_scanned,
        "scans_cached": stats.rendered_cached,
        "dropped": stats.dropped,
        "peak_in_flight_mb": round(stats.peak_in_flight_bytes / 1024**2, 1),
        "capture_to_output_p95_ms": round(stats.latency.percentile(0.95), 1),
        "longest_freeze_ms": round(stats.max_frozen_seconds * 1000, 1),
        "scan_mean_ms": round(stats.scan_seconds / max(stats.scanned, 1) * 1000, 2),
        "stalled_workers": list(stats.stalled_workers),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "memory_growth_after_warmup_mb": round(peak_memory_mb() - memory_warmed_up, 1),
    }


def failures(report: dict) -> list[str]:
    """Everything in a report that means protection failed."""
    problems = [
        name
        for name in (
            "exposed_secret_pixels",
            "exposed_keyword_pixels",
            "pictures_from_before_resume",
            "pictures_not_from_the_compositor",
            "stalled_workers",
        )
        if report[name]
    ]
    if not report["ended_on_black"]:
        problems.append("ended_on_black")
    if report["memory_growth_after_warmup_mb"] > _MEMORY_LIMIT_MB:
        problems.append("memory_growth_after_warmup_mb")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sustained_motion")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--out", help="write the results here as JSON")
    args = parser.parse_args(argv)
    report = run(args.seconds, args.fps, args.delay)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    problems = failures(report)
    if problems:
        print("FAILED: " + ", ".join(problems))
        return 1
    print("Synthetic run passed. This is not a live OBS result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
