"""Measures how covers behave during a scripted scroll. Fake secrets only.

Run from the repo root: python -m eval.scroll_compare

Reports exposed secret pixels, covered area that nothing on screen explains (trails),
how much cover edges jump around, and the output rate through the real delay pipeline.
It only uses interfaces older versions also have, so running a copy of it from an old
checkout gives a fair before and after. These numbers are synthetic: they say nothing
about a live OBS run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.output.virtual_cam import OUTPUT_HEIGHT, OUTPUT_WIDTH, fit
from eclipse.platform import AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.types import Frame
from eval.scroll_scene import (
    BACKGROUND,
    FAKE_EMAIL,
    FAKE_KEY,
    FAKE_PHONE,
    SceneReader,
    ScrollScene,
)

SECRETS = {4: FAKE_EMAIL, 9: FAKE_KEY, 15: FAKE_EMAIL + ".au", 24: FAKE_KEY + "9"}
OCR_WOBBLE = 2


def scroll_script(scale: int = 1) -> list[tuple[int, bool]]:
    """Offsets for each frame, and whether something blinks beside the first secret."""
    offset, script = 0, []

    def hold(frames: int, blink: bool = False) -> None:
        script.extend((offset, blink and index % 2 == 1) for index in range(frames))

    def scroll(frames: int, step: int) -> None:
        nonlocal offset
        for _ in range(frames):
            offset += step * scale
            script.append((offset, False))

    hold(10)
    scroll(60, 6)  # steady
    hold(20)
    scroll(40, -6)  # reversal
    hold(20)
    scroll(12, 40)  # flick
    hold(20)
    scroll(30, -12)
    hold(40, blink=True)  # still, with its line read again and again
    return script


def draw(scene: ScrollScene, offset: int, blink: bool) -> np.ndarray:
    image = scene.image(offset)
    if blink:
        boxes = list(scene.secret_boxes(offset).values())
        left, top = boxes[0][:2]
        image[top + 8 : top + 12, max(left - 6, 0) : max(left - 3, 1)] = 255
    return image


def exposed_pixels(raw, output, boxes) -> int:
    total = 0
    for left, top, right, bottom in boxes:
        ink = (raw[top:bottom, left:right] != BACKGROUND).any(axis=2)
        same = (output[top:bottom, left:right] == raw[top:bottom, left:right]).all(axis=2)
        total += int((ink & same).sum())
    return total


def exposed_after_fit(raw, output, boxes) -> int:
    """Secret pixels still bright once resized and letterboxed for the camera."""
    fitted = fit(output, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    height, width = raw.shape[:2]
    scale = min(OUTPUT_WIDTH / width, OUTPUT_HEIGHT / height)
    pad_x = (OUTPUT_WIDTH - round(width * scale)) // 2
    pad_y = (OUTPUT_HEIGHT - round(height * scale)) // 2
    total = 0
    for left, top, right, bottom in boxes:
        crop = fitted[
            int(top * scale) + pad_y : int(np.ceil(bottom * scale)) + pad_y,
            int(left * scale) + pad_x : int(np.ceil(right * scale)) + pad_x,
        ]
        total += int((crop.max(axis=2) > 60).sum())
    return total


def edge_gaps(areas, box) -> tuple[int, ...] | None:
    """How far the cover over a secret reaches past it on each side."""
    left, top, right, bottom = box
    over = [a for a in areas if a[0] < right and a[2] > left and a[1] < bottom and a[3] > top]
    if not over:
        return None
    return (
        left - min(a[0] for a in over),
        top - min(a[1] for a in over),
        max(a[2] for a in over) - right,
        max(a[3] for a in over) - bottom,
    )


def measure_covers(scene, reader, script, pixel_scale, padding, video=None) -> dict:
    detect = TextDetector(reader=reader, pixel_scale=pixel_scale)
    compositor = Compositor(padding=padding)
    exposed = fitted_exposed = frames_exposed = 0
    trail_shares, unread_shares, tallest = [], [], 0
    jumps, previous_gaps, gap_sizes = [], {}, []
    started = time.monotonic()
    for frame_id, (offset, blink) in enumerate(script):
        image = draw(scene, offset, blink)
        frame = Frame(image, frame_id / 60, "screen", frame_id)
        reported = []

        def recording(frame: Frame):
            reported[:] = detect(frame)
            return list(reported)

        output = compositor.process(frame, recording)
        boxes = scene.secret_boxes(offset)
        height, width = image.shape[:2]
        areas = compositor.covered_areas(height, width)
        mask = np.zeros((height, width), dtype=bool)
        for left, top, right, bottom in areas:
            mask[top:bottom, left:right] = True

        shown = exposed_pixels(image, output, boxes.values())
        exposed += shown
        frames_exposed += shown > 0
        fitted_exposed += exposed_after_fit(image, output, boxes.values())

        explained = np.zeros_like(mask)
        for left, top, right, bottom in boxes.values():
            explained[
                max(top - padding, 0) : bottom + padding, max(left - padding, 0) : right + padding
            ] = True
        unread = np.zeros_like(mask)
        for found in reported:
            if found.category == "unread":
                unread[
                    max(found.y - padding, 0) : found.y + found.h + padding,
                    max(found.x - padding, 0) : found.x + found.w + padding,
                ] = True
        trail_shares.append(float((mask & ~explained & ~unread).mean()))
        unread_shares.append(float((mask & unread & ~explained).mean()))
        tallest = max(tallest, int(mask.sum(axis=0).max()))

        gaps_now = {}
        for name, box in boxes.items():
            gaps = edge_gaps(areas, box)
            cut_off = box[1] == 0 or box[3] == height  # its edges are the frame's right now
            if gaps is not None and not cut_off:
                gaps_now[name] = gaps
                gap_sizes.append(max(gaps))
                if name in previous_gaps:
                    jumps.append(max(abs(a - b) for a, b in zip(gaps, previous_gaps[name])))
        previous_gaps = gaps_now
        if video is not None:
            video(frame_id, output)

    elapsed = time.monotonic() - started
    return {
        "frames": len(script),
        "held_frames": compositor.held_frames,
        "exposed_secret_pixels": exposed,
        "frames_with_exposed_pixels": frames_exposed,
        "exposed_pixels_after_resize": fitted_exposed,
        "trail_area_mean_pct": round(100 * float(np.mean(trail_shares)), 3),
        "trail_area_peak_pct": round(100 * float(np.max(trail_shares)), 3),
        "unread_area_mean_pct": round(100 * float(np.mean(unread_shares)), 3),
        "tallest_covered_band_px": tallest,
        "edge_jump_mean_px": round(float(np.mean(jumps)), 2) if jumps else 0.0,
        "edge_jump_p95_px": float(np.percentile(jumps, 95)) if jumps else 0.0,
        "cover_past_secret_p95_px": float(np.percentile(gap_sizes, 95)) if gap_sizes else 0.0,
        "ms_per_frame": round(elapsed / len(script) * 1000, 2),
    }


class ScriptedSource:
    def __init__(self, scene, script):
        self._scene = scene
        self._script = script
        self._next = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self) -> Frame:
        offset, blink = self._script[self._next % len(self._script)]
        frame = Frame(draw(self._scene, offset, blink), time.monotonic(), "screen", self._next)
        self._next += 1
        return frame


class PacedSink:
    """Stands in for the camera: fits each new image, and takes one send per tick."""

    def __init__(self, fps: int):
        self._interval = 1 / fps
        self._due = 0.0
        self._last = None

    def __enter__(self):
        self._due = time.monotonic()
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image) -> None:
        if image is not self._last:
            fit(image, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            self._last = image
        self._due = max(self._due + self._interval, time.monotonic())
        time.sleep(max(self._due - time.monotonic(), 0))


def measure_pipeline(scene, reader, script, pixel_scale, padding, seconds, fps, delay) -> dict:
    compositor = Compositor(padding=padding)
    pipeline = DelayedPipeline(
        ScriptedSource(scene, script),
        PacedSink(fps),
        TextDetector(reader=reader, pixel_scale=pixel_scale),
        compositor,
        delay,
        fps,
        int(seconds * fps),
    )
    stats = pipeline.run()
    latencies = sorted(stats.latencies)
    return {
        "fps_asked": fps,
        "captured_fps": round(stats.captured / stats.capture_seconds, 1),
        "fresh_output_fps": round(stats.shown / stats.capture_seconds, 1),
        "repeated_ticks": stats.repeated,
        "dropped": stats.dropped,
        "skipped": stats.skipped,
        "held": compositor.held_frames,
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "added_over_delay_ms": round(latencies[int(len(latencies) * 0.95)] - delay * 1000, 1),
        "scan_mean_ms": round(stats.scan_seconds / max(stats.scanned, 1) * 1000, 2),
    }


def video_writer(path: Path, scene: ScrollScene):
    size = (scene.width, scene.height)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    return writer, lambda frame_id, output: writer.write(output)


def blocks_scene(seconds_scale: int = 1) -> ScrollScene:
    return ScrollScene(secrets=SECRETS, side_secrets={3: FAKE_PHONE})


def retina_text_scene() -> ScrollScene:
    return ScrollScene(
        secrets=SECRETS,
        side_secrets={3: FAKE_PHONE},
        width=3024,
        height=1964,
        pane=(80, 2200),
        side=(2300, 3000),
        lines=60,
        font_scale=1.6,
        pitch=76,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scroll_compare")
    parser.add_argument("--label", default="current code")
    parser.add_argument("--out", help="write the results here as JSON")
    parser.add_argument("--video", help="write the clean-OCR run here as a video")
    parser.add_argument("--apple", action="store_true", help="also use real Apple OCR (macOS)")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args(argv)

    script = scroll_script()
    results: dict = {"label": args.label}

    scene = blocks_scene()
    writer = None
    on_frame = None
    if args.video:
        writer, on_frame = video_writer(Path(args.video), scene)
    results["clean_ocr"] = measure_covers(
        scene, SceneReader(scene, wobble=OCR_WOBBLE), script, 1, PADDING, on_frame
    )
    if writer is not None:
        writer.release()

    scene = blocks_scene()
    results["flaky_ocr"] = measure_covers(
        scene, SceneReader(scene, wobble=OCR_WOBBLE, miss_rate=0.08), script, 1, PADDING
    )

    scene = blocks_scene()
    results["pipeline_fake_ocr"] = measure_pipeline(
        scene,
        SceneReader(scene, wobble=OCR_WOBBLE),
        script,
        1,
        PADDING,
        args.seconds,
        args.fps,
        args.delay,
    )

    if args.apple:
        scene = retina_text_scene()
        retina_script = scroll_script(scale=2)
        results["apple_ocr"] = measure_covers(
            scene, AppleVisionReader(), retina_script, 2, PADDING * 2
        )
        results["pipeline_apple_ocr"] = measure_pipeline(
            scene,
            AppleVisionReader(),
            retina_script,
            2,
            PADDING * 2,
            args.seconds,
            args.fps,
            args.delay,
        )

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
