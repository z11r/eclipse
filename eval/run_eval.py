"""Measures leaks and latency. Run from the repo root: python -m eval.run_eval

Exits 0 with no leaks, 1 with leaks. A frame leaks if either
check finds a planted secret still visible in the output:

- Pixels (every frame): is the secret's text still there, even a small part of it?
- OCR (sampled frames): can real OCR still read it? This catches weak redaction
  that changes the pixels but leaves the text readable.

Every check is against the content actually on show, not the frame that just came in.
A held image is the old frame, so it is scored with that frame's secrets and raw
pixels, and it can go on leaking after the secret has left the clip.

Two clocks are in play and they are never mixed. Clip time (Frame.ts here) says when
things happen in the video. time.monotonic() only times how long our own code runs.
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from eclipse.capture.clip import ClipCapture
from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.output.virtual_cam import fit
from eclipse.platform import AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.types import Frame
from eval.fake_detectors import FAKE_DETECTORS
from eval.truth import CLIPS_DIR, EVAL_DIR, box_at, is_on_screen, load_labels

OUTPUT_DIR = EVAL_DIR / "output"

TEXT_CONTRAST = 40  # how far a pixel must differ from the background to count as text
SAME_PIXEL = 25  # how close an output pixel must be to count as unchanged
VISIBLE_LIMIT = 0.03  # share of a secret's text that may survive, to allow for noise
OCR_MATCH = 0.75  # how close OCR text must be to count as reading the secret


def visible_share(raw: np.ndarray, output: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """The share of a secret's text pixels that still show as text in the output.

    A pixel counts only if it kept its value and still stands out from the output's
    own background. Text edges can match a solid fill's colour by chance.
    """
    x, y, w, h = box
    raw_crop = cv2.cvtColor(raw[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(int)
    output_crop = cv2.cvtColor(output[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(int)
    text = np.abs(raw_crop - np.median(raw_crop)) > TEXT_CONTRAST
    if not text.any():
        raise ValueError("label box contains no text; check the labels")
    still_text = np.abs(output_crop - np.median(output_crop)) > TEXT_CONTRAST
    unchanged = np.abs(raw_crop - output_crop) < SAME_PIXEL
    return float((unchanged & still_text)[text].mean())


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ocr_reads(ocr, image: np.ndarray, secret_text: str) -> bool:
    """Whether OCR can read the secret anywhere in the image."""
    result, _ = ocr(image)
    target = normalise(secret_text)
    for _, text, _ in result or []:
        found = normalise(text)
        if target in found:
            return True
        matcher = difflib.SequenceMatcher(None, target, found)
        block = matcher.find_longest_match(0, len(target), 0, len(found))
        window = found[max(block.b - block.a, 0) : max(block.b - block.a, 0) + len(target)]
        if difflib.SequenceMatcher(None, target, window).ratio() >= OCR_MATCH:
            return True
    return False


def fitted_box(
    box: tuple[int, int, int, int], height: int, width: int, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Where a box lands after fit() scales the image to size and adds black bars."""
    x, y, w, h = box
    scale = min(size[0] / width, size[1] / height)
    left = (size[0] - round(width * scale)) // 2
    top = (size[1] - round(height * scale)) // 2
    x1, y1 = math.floor(x * scale) + left, math.floor(y * scale) + top
    x2, y2 = math.ceil((x + w) * scale) + left, math.ceil((y + h) * scale) + top
    return x1, y1, x2 - x1, y2 - y1


def p95(values: list[float]) -> float:
    return sorted(values)[int(len(values) * 0.95)] if values else 0.0


def evaluate(
    clip_path: str,
    labels: dict,
    detect,
    ocr=None,
    ocr_every: int = 15,
    on_frame=None,
    delay: float = 0.0,
    output_size: tuple[int, int] | None = None,
) -> dict:
    with ClipCapture(clip_path) as clip:
        frames = iter(clip.read, None)
        return evaluate_frames(frames, labels, detect, ocr, ocr_every, on_frame, delay, output_size)


def evaluate_frames(
    frames: Iterable[Frame],
    labels: dict,
    detect,
    ocr=None,
    ocr_every: int = 15,
    on_frame=None,
    delay: float = 0.0,
    output_size: tuple[int, int] | None = None,
) -> dict:
    """Score one output per frame, as viewers would see it.

    delay is the configured buffer, in seconds, for the modelled presentation time.
    output_size (width, height) scores the picture after the final resize, as sent to
    the camera, rather than as the compositor made it.
    """
    compositor = Compositor()
    secrets = labels["secrets"]
    leaks_by_secret = {secret["id"]: 0 for secret in secrets}
    sightings = 0
    leaked_frames = 0
    outputs = 0
    compute_ms: list[float] = []
    presented_ms: list[float] = []
    false_blur_shares: list[float] = []
    ocr_checks = ocr_baseline_hits = ocr_leaks = 0
    fresh_frames = repeated_frames = blackout_frames = dropped_inputs = 0
    in_clip = hidden_by_detection = hidden_by_fallback = 0
    frozen_total = frozen_longest = 0.0  # in seconds of clip time
    last_input = None  # id of the frame before this one
    raw_on_show = None  # the raw frame behind what is on show now
    on_show = None  # the id of the content on show
    shown_since = 0.0  # when that content first went up, in clip time
    previous_ts = 0.0

    for frame in frames:
        started = time.monotonic()
        rendered = compositor.render(frame, detect)
        spent = (time.monotonic() - started) * 1000
        compute_ms.append(spent)
        output = rendered.image
        outputs += 1
        if on_frame is not None:
            on_frame(frame.image, output)

        if last_input is not None:
            dropped_inputs += max(frame.frame_id - last_input - 1, 0)
        last_input = frame.frame_id

        current = rendered.state in ("scanned", "cached")
        if current:
            raw_on_show = frame.image
        blacked_out = rendered.state == "blackout"
        content_id = rendered.source_frame_id
        if blacked_out:
            blackout_frames += 1
            on_show = None
        elif content_id != on_show:
            fresh_frames += 1
            on_show = content_id
            shown_since = frame.ts
        else:
            repeated_frames += 1
            frozen_total += frame.ts - previous_ts
            frozen_longest = max(frozen_longest, frame.ts - shown_since)
        previous_ts = frame.ts
        if not blacked_out:
            age = (frame.ts - rendered.ts) * 1000  # clip time, so a held frame's age shows
            presented_ms.append(age + max(delay * 1000, spent))

        visible = [] if blacked_out else [s for s in secrets if is_on_screen(s, content_id)]
        sample_ocr = ocr is not None and frame.frame_id % ocr_every == 0
        scored_raw, scored_output = raw_on_show, output
        if output_size is not None and not blacked_out:
            scored_raw = fit(raw_on_show, *output_size)
            scored_output = fit(output, *output_size)
        frame_leaked = False
        leaked_now = set()
        for secret in visible:
            sightings += 1
            box = box_at(secret, content_id)
            if output_size is not None:
                box = fitted_box(box, *raw_on_show.shape[:2], output_size)
            leaked = visible_share(scored_raw, scored_output, box) > VISIBLE_LIMIT
            if sample_ocr:
                ocr_checks += 1
                ocr_baseline_hits += ocr_reads(ocr, scored_raw, secret["text"])
                if ocr_reads(ocr, scored_output, secret["text"]):
                    ocr_leaks += 1
                    leaked = True
            if leaked:
                leaks_by_secret[secret["id"]] += 1
                leaked_now.add(secret["id"])
                frame_leaked = True
        leaked_frames += frame_leaked

        for secret in secrets:
            if is_on_screen(secret, frame.frame_id):
                in_clip += 1
                if not current:
                    hidden_by_fallback += 1  # this frame never aired at all
                elif secret["id"] not in leaked_now:
                    hidden_by_detection += 1

        if current:
            height, width = output.shape[:2]
            covered = np.zeros((height, width), dtype=bool)
            for left, top, right, bottom in compositor.covered_areas(height, width):
                covered[top:bottom, left:right] = True
            for secret in visible:
                x, y, w, h = box_at(secret, content_id)
                # Padding around a real secret isn't false blur.
                top, left = max(y - PADDING, 0), max(x - PADDING, 0)
                covered[top : y + h + PADDING, left : x + w + PADDING] = False
            false_blur_shares.append(float(covered.mean()))

    if not outputs:
        raise ValueError("no frames to evaluate")
    leaked_sightings = sum(leaks_by_secret.values())
    return {
        "frames": outputs,
        "leaked_frames": leaked_frames,
        "leak_rate": leaked_frames / outputs,
        "leaks_by_secret": {k: v for k, v in leaks_by_secret.items() if v},
        "recall": 1 - leaked_sightings / sightings if sightings else 1.0,
        "compute_p95_ms": p95(compute_ms),
        "capture_to_present_p95_ms": p95(presented_ms),
        "fresh_frames": fresh_frames,
        "repeated_frames": repeated_frames,
        "blackout_frames": blackout_frames,
        "dropped_inputs": dropped_inputs,
        "frozen_longest_ms": frozen_longest * 1000,
        "frozen_total_ms": frozen_total * 1000,
        "detected_coverage": hidden_by_detection / in_clip if in_clip else 1.0,
        "fallback_coverage": hidden_by_fallback / in_clip if in_clip else 0.0,
        "false_blur": float(np.mean(false_blur_shares)) if false_blur_shares else 0.0,
        "held_frames": compositor.held_frames,
        "ocr_checks": ocr_checks,
        "ocr_baseline_hits": ocr_baseline_hits,
        "ocr_leaks": ocr_leaks,
    }


def split_screen_writer(path: Path, labels: dict):
    """Returns a function that writes raw and output side by side, and the writer."""
    size = (labels["width"] * 2, labels["height"])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), labels["fps"], size)
    if not writer.isOpened():
        raise RuntimeError(f"can't write video: {path}")

    def write(raw: np.ndarray, output: np.ndarray) -> None:
        panels = [raw.copy(), output.copy()]
        for panel, title in zip(panels, ("RAW CLIP", "ECLIPSE OUTPUT")):
            cv2.putText(panel, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        writer.write(np.hstack(panels))

    return write, writer


def print_report(name: str, report: dict) -> None:
    print(f"\nEval: {name}")
    print(f"  Leaked frames   {report['leaked_frames']} of {report['frames']}")
    for secret_id, count in report["leaks_by_secret"].items():
        print(f"    {secret_id}: {count} frames")
    print(f"  Recall          {report['recall']:.1%} of secret sightings hidden")
    print(
        f"  Compute         p95 {report['compute_p95_ms']:.1f} ms in the detector and compositor "
        "(offline: no capture, delay buffer or output)"
    )
    print(
        f"  Presentation    p95 {report['capture_to_present_p95_ms']:.1f} ms from capture to "
        "output (modelled from clip time and the delay, not measured at the camera)"
    )
    print(
        f"  Outputs         {report['fresh_frames']} fresh, {report['repeated_frames']} repeated, "
        f"{report['blackout_frames']} black, {report['dropped_inputs']} inputs missing"
    )
    print(
        f"  Frozen          {report['frozen_longest_ms']:.0f} ms longest, "
        f"{report['frozen_total_ms']:.0f} ms in all"
    )
    print(
        f"  Coverage        {report['detected_coverage']:.1%} hidden by detection, "
        f"{report['fallback_coverage']:.1%} kept off air by fallback"
    )
    print(f"  False blur      {report['false_blur']:.2%} of the screen covered needlessly")
    print(f"  Held frames     {report['held_frames']}")
    if report["ocr_checks"]:
        print(
            f"  OCR check       read {report['ocr_baseline_hits']} of {report['ocr_checks']} "
            f"in the raw clip, {report['ocr_leaks']} in the output"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval")
    parser.add_argument("--clip", default="planted_secrets", help="name in eval/labels")
    parser.add_argument(
        "--detector",
        choices=["text-layer", *FAKE_DETECTORS],
        default="text-layer",
        help="text-layer is the real one; the rest are fakes for checking the eval",
    )
    parser.add_argument("--reader", choices=["rapid", "apple"], default="rapid")
    parser.add_argument("--no-ocr", action="store_true", help="pixel check only, faster")
    parser.add_argument("--ocr-every", type=int, default=15, help="OCR every Nth frame")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="buffer to model in presentation time, seconds"
    )
    parser.add_argument(
        "--save", action="store_true", help="write a raw vs output video to eval/output"
    )
    args = parser.parse_args(argv)

    labels = load_labels(args.clip)
    if args.detector == "text-layer":
        detect = TextDetector(reader=AppleVisionReader() if args.reader == "apple" else None)
    else:
        detect = FAKE_DETECTORS[args.detector](labels)

    ocr = None
    if not args.no_ocr:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()

    on_frame = writer = None
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        video_path = OUTPUT_DIR / f"{args.clip}_{args.detector}.mp4"
        on_frame, writer = split_screen_writer(video_path, labels)

    report = evaluate(
        str(CLIPS_DIR / labels["clip"]), labels, detect, ocr, args.ocr_every, on_frame, args.delay
    )
    print_report(args.detector, report)
    if writer is not None:
        writer.release()
        print(f"  Video           {video_path.relative_to(EVAL_DIR.parent)}")
    return 1 if report["leaked_frames"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
