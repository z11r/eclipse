"""Wires the pipeline. Run with: python -m eclipse.main

Every frame goes through the text layer and the compositor before the delay buffer
hands it to the output, so there is no way to send an unscanned frame. The panel and
the panic key drive the same running pipeline, through one controller.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from eclipse.capture.clip import LiveClip
from eclipse.capture.dual import PIP_SIZE, VIEWS, DualCapture
from eclipse.capture.pattern import TestPattern
from eclipse.capture.preferred import PreferredCapture
from eclipse.capture.screen import ScreenCapture
from eclipse.capture.webcam import WebcamCapture
from eclipse.compositor.compositor import PADDING, STYLES, Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.compositor.pip import PictureInPictureCompositor, PictureInPictureDetector
from eclipse.output.virtual_cam import DiscardingCam, VirtualCam
from eclipse.panel.app import PANEL_HOST, PANEL_PORT, start_panel
from eclipse.panel.controller import LivePipelineController
from eclipse.panel.hotkey import PANIC_KEYS, VIEW_KEYS, PanicHotkey
from eclipse.platform import (
    AppleVisionReader,
    WindowsOCRReader,
    screen_capture_allowed,
    screen_capture_kit_available,
)
from eclipse.text_detect import TextDetector
from eclipse.text_detect.read_ahead import OcrPool, ReadAhead, ReadError
from eclipse.types import Frame


RETINA_MAX_WIDTH = 6016  # wide enough never to shrink a built-in or Studio Display
# A camera's 8x8 block averages wander up to about 6 from noise alone. New text moves
# them far more, so only the parts of the picture that really changed are read again.
WEBCAM_NOISE_FLOOR = 8
# OCR threads for the camera when it runs beside the screen. Its reads are small, and
# the screen's reader keeps its six, so together they still leave cores for OBS.
WEBCAM_OCR_THREADS = 2
# Seconds of the delay a scanned frame waits for later frames to read the rows scrolling
# into it. Never more of the delay than the share, leaving the rest for slow scans.
LOOK_AHEAD = 0.3
LOOK_AHEAD_SHARE = 0.3


class StatsLog:
    """Writes pipeline numbers to a file every few seconds. Never any screen text."""

    def __init__(
        self,
        path: str,
        pipeline: DelayedPipeline,
        parts: dict,
        every: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = path
        self._pipeline = pipeline
        self._parts = parts  # name -> source, detector or sink, each with its own stats
        self._every = every
        self._clock = clock
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="eclipse-stats", daemon=True)

    def __enter__(self) -> "StatsLog":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        self._thread.join(timeout=5)

    def _snapshot(self) -> dict:
        # Each name carries its stage, so two stages can't overwrite each other's.
        numbers = {
            f"pipeline.{name}": value
            for name, value in vars(self._pipeline.stats).items()
            if isinstance(value, (int, float))
        }
        for stage, part in self._parts.items():
            for name, value in dict(getattr(part, "stats", {})).items():
                numbers[f"{stage}.{name}"] = value
        # User plus system CPU time for the whole process, on every OS.
        # (resource.getrusage doesn't exist on Windows.)
        numbers["process.cpu_seconds"] = time.process_time()
        numbers["clock.seconds"] = self._clock()
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
        stats = self._pipeline.stats
        present = _drain(stats.present_intervals)
        scan_times = _drain(stats.detect_times)
        ages = _drain(stats.scan_ages)
        slack = _drain(stats.slack)
        # What really passed. A log line can run late, and a rate must not pretend it didn't.
        elapsed = max(change["clock.seconds"], 1e-9)

        def count(name: str) -> int:
            return round(change.get(name, 0))

        def fps(name: str) -> float:
            return change.get(name, 0) / elapsed

        def ms(name: str, frames: float) -> float:
            return change.get(name, 0) / max(frames, 1) * 1000

        def busy(*names: str) -> float:
            return sum(change.get(name, 0) for name in names) / elapsed

        captured = count("pipeline.captured")
        scanned = count("pipeline.scanned")
        sends = count("output.sends")
        fits = count("output.fits")
        reads = count("detect.reads")
        scans = count("detect.full_scans") + count("detect.partial_scans")
        detect_ms = ms("pipeline.detect_seconds", scanned)
        copy_ms = ms("detect.copy_seconds", scanned)
        diff_ms = ms("detect.diff_seconds", scanned)
        ocr_ms = ms("detect.scan_seconds", scanned)
        rules_ms = ms("detect.find_seconds", scanned)
        track_ms = ms("detect.track_seconds", scanned)
        known_ms = copy_ms + diff_ms + ocr_ms + rules_ms + track_ms
        webcam = ""
        if "detect.webcam_scans" in now:
            webcam_scans = count("detect.webcam_scans")
            webcam = (
                f"webcam scans {webcam_scans} repeats {count('detect.webcam_repeats')} "
                f"reads {count('detect.webcam_reads')} "
                f"ms/scan {ms('detect.webcam_scan_seconds', webcam_scans):.1f} "
                f"camera {fps('capture.webcam_frames'):.1f}fps "
                f"screen {fps('capture.screen_frames'):.1f}fps | "
            )
        age = -1.0
        if stats.last_presented_at is not None:
            age = (now["clock.seconds"] - stats.last_presented_at) * 1000
        return (
            f"{time.strftime('%H:%M:%S')} "
            f"capture {fps('pipeline.captured'):.1f}fps fresh {fps('pipeline.fresh'):.1f}fps "
            f"sent {fps('pipeline.shown'):.1f}fps fresh-sent {fps('pipeline.fresh_shown'):.1f}fps | "
            f"repeats {count('pipeline.repeated')} dropped {count('pipeline.dropped')} "
            f"(shed on arrival {count('pipeline.shed')}) "
            f"skipped {count('pipeline.skipped')} id-gaps {count('pipeline.id_gaps')} "
            f"held {count('pipeline.held')} backlog {now['pipeline.backlog']} "
            f"last-id {now['pipeline.last_sent_id']} | "
            f"tries {count('pipeline.capture_attempts')} admitted {count('pipeline.admitted')} "
            f"no-room {count('pipeline.capture_skips')} refused {count('pipeline.discarded')} "
            f"in-flight {now['pipeline.in_flight_frames']} "
            f"{now['pipeline.in_flight_bytes'] / 1024**2:.0f}MB | "
            f"results read {count('pipeline.rendered_scanned')} "
            f"cached {count('pipeline.rendered_cached')} "
            f"held {count('pipeline.rendered_held')} "
            f"blackout {count('pipeline.rendered_blackout')} "
            f"held-panes {count('pipeline.held_panes')} | "
            f"camera-sends {count('pipeline.sends')} black {count('pipeline.black_sends')} "
            f"new-pictures {count('pipeline.distinct_presented')} "
            f"protected-age-ms {age:.0f} frozen-max-ms {stats.max_frozen_seconds * 1000:.0f} | "
            f"present-ms p50 {_percentile(present, 0.5):.1f} p95 {_percentile(present, 0.95):.1f} "
            f"max {_percentile(present, 1):.1f} | "
            f"queue-ms p50 {_percentile(ages, 0.5):.1f} p95 {_percentile(ages, 0.95):.1f} "
            f"slack-min {min(slack, default=0):.0f} | "
            f"capture-ms read {ms('pipeline.grab_seconds', captured):.1f} "
            f"grab {ms('capture.grab_seconds', captured):.1f} "
            f"convert {ms('capture.convert_seconds', captured):.1f} | "
            f"detect-ms total {detect_ms:.1f} p95 {_percentile(scan_times, 0.95):.1f} "
            f"max {max(scan_times, default=0):.1f} "
            f"over-frame {sum(t > 1000 / 60 for t in scan_times)} "
            f"copy {copy_ms:.1f} diff {diff_ms:.1f} "
            f"ocr {ocr_ms:.1f} rules {rules_ms:.1f} track {track_ms:.1f} "
            f"other {detect_ms - known_ms:.1f} | "
            f"diff-parts areas {ms('detect.areas_seconds', scanned):.1f} "
            f"scroll {ms('detect.scroll_seconds', scanned):.1f} "
            f"narrow {ms('detect.narrow_seconds', scanned):.1f} "
            f"move {ms('detect.move_seconds', scanned):.1f} "
            f"ahead {ms('detect.ahead_seconds', scanned):.1f} "
            f"cut-short {count('detect.follow_cut')} repeated {count('detect.moves_repeated')} of {count('detect.repeat_tries')} "
            f"beside-others {count('detect.repeat_crowded')} missed {count('detect.repeat_missed')} "
            f"upright {count('detect.repeat_upright')} "
            f"miss-by inside {count('detect.miss_inside')} near {count('detect.miss_near')} "
            f"far {count('detect.miss_far')} lost {count('detect.miss_lost')} "
            f"estimate {ms('detect.estimate_seconds', scanned):.1f} "
            f"columns {ms('detect.columns_seconds', scanned):.1f} "
            f"pixels {ms('detect.pixels_seconds', scanned):.1f} "
            f"grow {ms('detect.grow_seconds', scanned):.1f} | "
            f"held-rows lost {count('detect.held_lost')} unread {count('detect.held_unread')} "
            f"cut {count('detect.held_cut')} doubtful {count('detect.held_doubtful')} "
            f"lost-by view {count('detect.lost_view')} "
            f"unexplained {count('detect.lost_unexplained')} edge {count('detect.lost_edge')} "
            f"still {count('detect.lost_still')} skipped-tries {count('detect.follow_skipped')} | "
            f"read-ahead jobs {count('read.jobs')} hits {count('read.hits')} "
            f"misses {count('read.misses')} failed {count('read.failed_reads')} "
            f"shed {count('read.shed')} cancelled {count('read.cancelled')} "
            f"prepare-ms {ms('read.prepare_seconds', captured):.1f} "
            f"work-ms {ms('read.work_seconds', count('read.jobs')):.1f} "
            f"wait-ms {ms('read.wait_seconds', count('read.jobs')):.1f} | "
            f"composite-ms {ms('pipeline.composite_seconds', scanned):.1f} | "
            f"output-ms fit {ms('output.fit_seconds', fits):.1f} "
            f"send {ms('output.send_seconds', sends):.1f} late {count('output.late_ticks')} | "
            f"scans full {count('detect.full_scans')} partial {count('detect.partial_scans')} "
            f"scrolls {count('detect.scrolls')} moves {count('detect.moves')} "
            f"left-out {count('detect.areas_left_out')} "
            f"area {change.get('detect.scanned_share', 0) / max(scans, 1):.0%} "
            f"ms/read {ms('detect.scan_seconds', reads):.0f} "
            f"lines/read {count('detect.lines') / max(reads, 1):.0f} "
            f"det {ms('detect.det_seconds', reads):.0f} rec {ms('detect.rec_seconds', reads):.0f} | "
            f"accuracy uncertain {count('detect.uncertain')} "
            f"doubtful {count('detect.doubtful_lines')} rescues {count('detect.rescues')} "
            f"cached {count('detect.rescue_cached')} skipped {count('detect.rescue_skipped')} "
            f"failed {count('detect.rescue_failed')} "
            f"rescue-ms {ms('detect.rescue_seconds', count('detect.rescues')):.0f} | "
            f"{webcam}"
            f"busy capture {busy('capture.grab_seconds', 'capture.convert_seconds'):.0%} "
            f"scan {busy('pipeline.scan_seconds'):.0%} "
            f"output {busy('output.fit_seconds', 'output.send_seconds'):.0%} "
            f"workers {busy('read.work_seconds'):.1f} "
            f"cores {busy('process.cpu_seconds'):.1f}"
        )


def _drain(samples: deque) -> list[float]:
    """Take everything recorded since the last log line."""
    taken = []
    while samples:
        taken.append(samples.popleft())
    return taken


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * share), len(ordered) - 1)]


def screen_source(capture: str, max_width: int, retina: bool, fps: int):
    """The screen backend to use, with polling kept as the fallback."""

    def polling(first_id: int = 0) -> ScreenCapture:
        return ScreenCapture(max_width=max_width, retina=retina, first_id=first_id)

    if capture == "mss" or not screen_capture_kit_available():
        return polling()
    from eclipse.capture.sck import ScreenCaptureKitCapture

    return PreferredCapture(ScreenCaptureKitCapture(max_width, retina, fps), polling)


def warm_up(detector) -> None:
    """Load a detector's OCR model now, on its own thread, so the first switch to a
    view that needs it doesn't freeze the picture while the model loads. A blank
    picture of another size, so the first real frame starts afresh."""

    def run() -> None:
        try:
            detector(Frame(np.zeros((64, 64, 3), np.uint8), 0.0, "webcam", -1))
        except Exception:
            pass  # a real frame will try again, and fail closed if it must

    threading.Thread(target=run, name="eclipse-warm-up", daemon=True).start()


def camera_input(value: str):
    """A camera index, or a recorded clip standing in for a camera."""
    if value.isdigit():
        return WebcamCapture(int(value))
    if not Path(value).is_file():
        raise FileNotFoundError(value)
    return LiveClip(value, "webcam", loop=True)


def _stop_on_sigterm() -> Callable[[], None]:
    """Treat a polite kill like Ctrl+C, so it ends in black too. Returns how to undo it."""

    def interrupt(signum, stack) -> None:
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGTERM, interrupt)
    except ValueError:  # only the main thread can set signals
        return lambda: None
    return lambda: signal.signal(signal.SIGTERM, previous)


def start_controls(controller: LivePipelineController, panel: bool, hotkey: bool):
    """Start the panel and the panic key. The stream runs without them, but says so."""
    server = None
    if panel:
        server = start_panel(controller)
        if server is None:
            print(f"Panel unavailable: {PANEL_HOST}:{PANEL_PORT} didn't start")
        else:
            print(f"Panel at http://{PANEL_HOST}:{PANEL_PORT}")
    keys = []
    if hotkey:
        panic = PanicHotkey(controller.panic)
        if panic.start():
            keys.append(panic)
            print(f"Panic key: {PANIC_KEYS}")
        else:
            print(
                "Panic key unavailable. On macOS, allow Input Monitoring in System Settings "
                "> Privacy & Security. The panel's panic button still works."
            )
        if controller.views:
            switch = PanicHotkey(controller.next_view, VIEW_KEYS)
            if switch.start():
                keys.append(switch)
                print(f"Switch view key: {VIEW_KEYS} ({' > '.join(controller.views)})")
    return server, keys


def stop_controls(server, keys) -> None:
    for key in keys or ():
        key.stop()
    if server is not None:
        server.should_exit = True


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
    parser.add_argument(
        "--clip",
        metavar="PATH",
        help="play a recorded video through the live pipeline instead of the screen",
    )
    parser.add_argument(
        "--clip-source",
        choices=["screen", "webcam", "irl"],
        default="screen",
        help="what the clip stands in for, as tagged on its frames",
    )
    parser.add_argument("--loop", action="store_true", help="replay the clip when it ends")
    parser.add_argument(
        "--with-webcam",
        metavar="INDEX_OR_CLIP",
        help="add this camera in a corner of the screen (or clip); each is scanned "
        "separately. A video file path stands in for the camera, looped",
    )
    parser.add_argument(
        "--view",
        choices=VIEWS,
        default="both",
        help="with --with-webcam, what to show first: the screen, both, or the webcam. "
        "Switch live with the panel's PUT /api/view or the switch view key",
    )
    parser.add_argument(
        "--pip-size",
        type=float,
        default=PIP_SIZE,
        help="the camera's width in the corner, as a share of the screen's",
    )
    parser.add_argument(
        "--cover",
        choices=STYLES,
        default="black",
        help="how covers look: solid black, or blurred from the colours around them. "
        "Both hide the same areas. Switch live from the panel",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds behind live")
    parser.add_argument(
        "--look-ahead",
        type=float,
        default=None,
        help="seconds of the delay each frame waits for later frames to read the rows "
        f"scrolling into it, which then need no black strip; 0 turns it off. Default "
        f"{LOOK_AHEAD} s, at most {LOOK_AHEAD_SHARE:.0%} of the delay",
    )
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--retina", action="store_true", help="capture every Retina pixel")
    parser.add_argument(
        "--ocr",
        choices=["rapid", "apple", "windows"],
        default="rapid",
        help="rapid is portable; apple uses macOS Vision; windows uses native Windows OCR",
    )
    parser.add_argument(
        "--capture",
        choices=["auto", "mss"],
        default="auto",
        help="auto uses ScreenCaptureKit where it works; mss always polls",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=4,
        help="processes reading Apple OCR ahead of the scan; 0 reads one area at a time",
    )
    parser.add_argument(
        "--rescue-reads",
        action="store_true",
        help="with --ocr apple, reread lines that name a secret but show no readable value "
        "with Apple's accurate OCR, within a time budget. Off by default",
    )
    parser.add_argument("--log", default="eclipse-run.log", help="file for live stats")
    parser.add_argument(
        "--no-output", action="store_true", help="measure every stage but send frames nowhere"
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-panel", action="store_true", help="no local control panel, as with --max-frames"
    )
    parser.add_argument(
        "--no-hotkey", action="store_true", help="no panic key, as with --max-frames"
    )
    args = parser.parse_args(argv)
    chosen = [args.test_pattern, args.webcam is not None, args.clip is not None]
    if sum(chosen) > 1:
        parser.error("pick one of --test-pattern, --webcam and --clip")
    if args.loop and args.clip is None:
        parser.error("--loop only works with --clip")
    if args.with_webcam is not None and (args.test_pattern or args.webcam is not None):
        parser.error("--with-webcam adds a camera to the screen or a --clip")
    if not 0 < args.pip_size <= 1:
        parser.error("--pip-size must be above 0 and at most 1")
    cv2.setNumThreads(2)  # leave the cores to OBS

    padding = PADDING
    read_ahead = None
    if args.test_pattern:
        source = TestPattern()
        detect = source.detect
    elif args.webcam is not None:
        source = WebcamCapture(args.webcam)
        detect = TextDetector(noise_floor=WEBCAM_NOISE_FLOOR)
    elif args.clip is not None:
        if not Path(args.clip).is_file():
            print(f"No clip at {args.clip}")
            return 1
        source = LiveClip(args.clip, args.clip_source, args.loop)
        # A webcam recording carries sensor noise, so it needs the webcam's noise floor.
        noise = WEBCAM_NOISE_FLOOR if args.clip_source == "webcam" else 0
        detect = TextDetector(noise_floor=noise)
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
            source = screen_source(args.capture, RETINA_MAX_WIDTH, True, args.fps)
            reader = AppleVisionReader()
            if args.ocr_workers > 0:
                try:
                    reader = read_ahead = ReadAhead(OcrPool(args.ocr_workers))
                except ReadError:
                    print("OCR workers didn't start, so reading one area at a time")
            # Careful rereads are opt-in, so the default detector is built exactly as before.
            careful = {"rescue": AppleVisionReader(accurate=True)} if args.rescue_reads else {}
            detect = TextDetector(reader=reader, pixel_scale=2, **careful)
            padding = PADDING * 2
        elif args.ocr == "windows":
            source = screen_source(args.capture, args.max_width, args.retina, args.fps)
            detect = TextDetector(reader=WindowsOCRReader())
        else:
            source = screen_source(args.capture, args.max_width, args.retina, args.fps)
            detect = TextDetector()

    compositor = Compositor(padding=padding)
    views = None
    if args.with_webcam is not None:
        try:
            camera = camera_input(args.with_webcam)
        except FileNotFoundError:
            print(f"No camera clip at {args.with_webcam}")
            return 1
        source = views = DualCapture(source, camera, args.pip_size, args.fps, args.view)
        # RapidOCR for the camera even with Apple OCR on the screen: Vision allows one
        # read per process, and misses small text on a camera's lower resolution.
        corner = TextDetector(noise_floor=WEBCAM_NOISE_FLOOR, ocr_threads=WEBCAM_OCR_THREADS)
        alone = TextDetector(noise_floor=WEBCAM_NOISE_FLOOR)  # the camera at full size
        detect = PictureInPictureDetector(detect, corner, alone)
        compositor = PictureInPictureCompositor(detect, padding=padding)
        warm_up(corner)
        warm_up(alone)
    compositor.style = args.cover  # switchable live from the panel
    sink = (DiscardingCam if args.no_output else VirtualCam)(fps=args.fps)
    if args.look_ahead is None:
        args.look_ahead = min(LOOK_AHEAD, max(args.delay, 0) * LOOK_AHEAD_SHARE)
    elif not 0 <= args.look_ahead < args.delay:
        parser.error("--look-ahead must be at least 0 and less than --delay")
    pipeline = DelayedPipeline(
        source, sink, detect, compositor, args.delay, args.fps, args.max_frames,
        look_ahead=args.look_ahead,
    )
    controller = LivePipelineController(
        pipeline, getattr(detect, "configure", None), views=views, compositor=compositor
    )
    measuring = args.max_frames is not None  # a bounded run is a measurement, not a stream
    server, keys = start_controls(
        controller, not (args.no_panel or measuring), not (args.no_hotkey or measuring)
    )
    allow_sigterm = _stop_on_sigterm()
    parts = {"capture": source, "detect": detect, "output": sink, "read": read_ahead}
    try:
        with StatsLog(args.log, pipeline, parts):
            stats = pipeline.run()
    finally:
        allow_sigterm()
        stop_controls(server, keys)
        if read_ahead is not None:
            read_ahead.close()
    if stats.latencies:
        print(
            f"Captured {stats.captured / stats.capture_seconds:.1f} fps "
            f"({stats.fresh / stats.capture_seconds:.1f} fresh), "
            f"sent {stats.shown / stats.capture_seconds:.1f} new frames per second "
            f"({stats.fresh_shown / stats.capture_seconds:.1f} fresh), "
            f"{stats.repeated} repeats, {stats.dropped} dropped, {stats.skipped} skipped, "
            f"{compositor.held_frames} held"
        )
        print(
            f"Capture to output p95 {stats.latency.percentile(0.95):.0f} ms, longest backlog "
            f"{stats.max_backlog} frames, average scan "
            f"{stats.scan_seconds / max(stats.scanned, 1) * 1000:.0f} ms"
        )
    if stats.shown and not stats.final_black_sent:
        print("Warning: the last frame sent was not confirmed black. Check OBS.")
        return 1
    if stats.stalled_workers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
