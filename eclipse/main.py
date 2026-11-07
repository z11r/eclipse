"""Wires the pipeline. Run with: python -m eclipse.main

Every frame goes through the text layer and the compositor before the delay buffer
hands it to the output, so there is no way to send an unscanned frame.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2

from eclipse.capture.pattern import TestPattern
from eclipse.capture.screen import ScreenCapture
from eclipse.capture.webcam import WebcamCapture
from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.output.virtual_cam import VirtualCam
from eclipse.platform import AppleVisionReader, screen_capture_allowed
from eclipse.text_detect import TextDetector


RETINA_MAX_WIDTH = 6016  # wide enough never to shrink a built-in or Studio Display


class StatsLog:
    """Writes pipeline numbers to a file every few seconds. Never any screen text."""

    def __init__(
        self, path: str, pipeline: DelayedPipeline, parts: tuple, every: float = 2.0
    ) -> None:
        self._path = path
        self._pipeline = pipeline
        self._parts = parts  # source, detector and sink, each with its own stats
        self._every = every
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="eclipse-stats", daemon=True)

    def __enter__(self) -> "StatsLog":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        self._thread.join(timeout=5)

    def _snapshot(self) -> dict:
        numbers = {
            name: value
            for name, value in vars(self._pipeline.stats).items()
            if isinstance(value, (int, float))
        }
        for part in self._parts:
            numbers.update(getattr(part, "stats", {}))
        # User plus system CPU time for the whole process, on every OS.
        numbers["cpu_seconds"] = time.process_time()
        return numbers

    def _run(self) -> None:
        with open(self._path, "a", encoding="utf-8") as log:
            log.write(f"--- run started {time.strftime('%H:%M:%S')}\n")
            before = self._snapshot()
            while not self._done.wait(self._every):
                now = self._snapshot()
                log.write(self._line(before, now) + "\n")
                log.flush()
                before = now

    def _line(self, before: dict, now: dict) -> str:
        change = {name: now[name] - before.get(name, 0) for name in now}

        def ms(name: str, count: float) -> float:
            return change.get(name, 0) / max(count, 1) * 1000

        scans = change.get("full_scans", 0) + change.get("partial_scans", 0)
        reads = change.get("reads", 0)
        return (
            f"{time.strftime('%H:%M:%S')} "
            f"capture {change['captured'] / self._every:.1f}fps "
            f"sent {change['shown'] / self._every:.1f}fps "
            f"dropped {change['dropped']} skipped {change['skipped']} "
            f"repeats {change['repeated']} backlog {now['backlog']} | "
            f"grab {ms('grab_seconds', change['captured']):.0f} "
            f"resize {ms('resize_seconds', change['captured']):.0f} "
            f"diff {ms('diff_seconds', change.get('frames', 0)):.1f} | "
            f"scans full {change.get('full_scans', 0)} partial {change.get('partial_scans', 0)} "
            f"scrolls {change.get('scrolls', 0)} "
            f"avg {ms('scan_seconds', scans):.0f} area {change.get('scanned_share', 0) / max(scans, 1):.0%} "
            f"lines/scan {change.get('lines', 0) / max(reads, 1):.0f} "
            f"det {ms('det_seconds', reads):.0f} rec {ms('rec_seconds', reads):.0f} "
            f"rules {ms('find_seconds', scans):.1f} busy {change.get('scan_seconds', 0) / self._every:.0%} | "
            f"fit {ms('fit_seconds', change.get('fits', 0)):.1f} "
            f"send {ms('send_seconds', change.get('sends', 0)):.1f} "
            f"cores {change['cpu_seconds'] / self._every:.1f}"
        )


def p95(values: list[float]) -> float:
    return sorted(values)[int(len(values) * 0.95)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eclipse")
    parser.add_argument(
        "--test-pattern",
        action="store_true",
        help="send a moving pattern with a fake secret hidden by the compositor",
    )
    parser.add_argument(
        "--webcam",
        type=int,
        metavar="INDEX",
        help="capture this camera instead of the screen (0 is usually the built-in one)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds behind live")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--retina", action="store_true", help="capture every Retina pixel")
    parser.add_argument(
        "--ocr",
        choices=["rapid", "apple"],
        default="rapid",
        help="apple is much faster on macOS, finds fewer secrets, and scans Retina frames",
    )
    parser.add_argument("--log", default="eclipse-run.log", help="file for live stats")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    cv2.setNumThreads(2)  # leave the cores to OBS

    padding = PADDING
    if args.test_pattern:
        source = TestPattern()
        detect = source.detect
    elif args.webcam is not None:
        source = WebcamCapture(args.webcam)
        detect = TextDetector()
    else:
        if not screen_capture_allowed():
            print(
                "Screen Recording is off for this app, so frames would be blank.\n"
                "Allow it in System Settings > Privacy & Security > Screen Recording, "
                "then restart the app."
            )
            return 1
        if args.ocr == "apple":
            # Apple's OCR needs every pixel, and all distances double with them.
            source = ScreenCapture(max_width=RETINA_MAX_WIDTH, retina=True)
            detect = TextDetector(reader=AppleVisionReader(), pixel_scale=2)
            padding = PADDING * 2
        else:
            source = ScreenCapture(max_width=args.max_width, retina=args.retina)
            detect = TextDetector()

    compositor = Compositor(padding=padding)
    sink = VirtualCam(fps=args.fps)
    pipeline = DelayedPipeline(
        source, sink, detect, compositor, args.delay, args.fps, args.max_frames
    )
    with StatsLog(args.log, pipeline, (source, detect, sink)):
        stats = pipeline.run()
    if stats.latencies:
        print(
            f"Captured {stats.captured / stats.capture_seconds:.1f} fps, "
            f"sent {stats.shown / stats.capture_seconds:.1f} new frames per second, "
            f"{stats.dropped} dropped, {stats.skipped} skipped, {compositor.held_frames} held"
        )
        print(
            f"Capture to output p95 {p95(stats.latencies):.0f} ms, longest backlog "
            f"{stats.max_backlog} frames, average scan "
            f"{stats.scan_seconds / max(stats.scanned, 1) * 1000:.0f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
