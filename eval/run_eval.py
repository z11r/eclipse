"""Measures leaks and latency. Run from the repo root: python -m eval.run_eval

Exits 0 with no leaks, 1 with leaks, 2 if it can't run. A frame leaks if either
check finds a planted secret still visible in the output:

- Pixels (every frame): is the secret's text still there, even a small part of it?
- OCR (sampled frames): can real OCR still read it? This catches weak redaction
  that changes the pixels but leaves the text readable.
"""

from __future__ import annotations

import argparse
import difflib
import re
import time
from pathlib import Path

import cv2
import numpy as np

from eclipse.capture.clip import ClipCapture
from eclipse.compositor.compositor import PADDING, Compositor
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


def evaluate(
    clip_path: str, labels: dict, detect, ocr=None, ocr_every: int = 15, on_frame=None
) -> dict:
    compositor = Compositor()
    secrets = labels["secrets"]
    leaks_by_secret = {secret["id"]: 0 for secret in secrets}
    sightings = 0
    leaked_frames = 0
    latencies: list[float] = []
    false_blur_shares: list[float] = []
    ocr_checks = ocr_baseline_hits = ocr_leaks = 0

    with ClipCapture(clip_path) as clip:
        while (frame := clip.read()) is not None:
            held_before = compositor.held_frames
            started = time.monotonic()
            output = compositor.process(frame, detect)
            latencies.append((time.monotonic() - started) * 1000)
            held = compositor.held_frames > held_before
            if on_frame is not None:
                on_frame(frame.image, output)

            active = [s for s in secrets if is_on_screen(s, frame.frame_id)]
            sample_ocr = ocr is not None and frame.frame_id % ocr_every == 0
            frame_leaked = False
            for secret in active:
                sightings += 1
                box = box_at(secret, frame.frame_id)
                leaked = visible_share(frame.image, output, box) > VISIBLE_LIMIT
                if sample_ocr:
                    ocr_checks += 1
                    ocr_baseline_hits += ocr_reads(ocr, frame.image, secret["text"])
                    if ocr_reads(ocr, output, secret["text"]):
                        ocr_leaks += 1
                        leaked = True
                if leaked:
                    leaks_by_secret[secret["id"]] += 1
                    frame_leaked = True
            leaked_frames += frame_leaked

            if not held:
                height, width = output.shape[:2]
                covered = np.zeros((height, width), dtype=bool)
                for left, top, right, bottom in compositor.covered_areas(height, width):
                    covered[top:bottom, left:right] = True
                for secret in active:
                    x, y, w, h = box_at(secret, frame.frame_id)
                    # Padding around a real secret isn't false blur.
                    top, left = max(y - PADDING, 0), max(x - PADDING, 0)
                    covered[top : y + h + PADDING, left : x + w + PADDING] = False
                false_blur_shares.append(float(covered.mean()))

    if not latencies:
        raise ValueError(f"no frames in clip: {clip_path}")
    leaked_sightings = sum(leaks_by_secret.values())
    return {
        "frames": len(latencies),
        "leaked_frames": leaked_frames,
        "leak_rate": leaked_frames / len(latencies),
        "leaks_by_secret": {k: v for k, v in leaks_by_secret.items() if v},
        "recall": 1 - leaked_sightings / sightings if sightings else 1.0,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
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
    print(f"  Scan + redact   p95 {report['p95_ms']:.1f} ms (no capture or output)")
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
    parser.add_argument("--no-ocr", action="store_true", help="pixel check only, faster")
    parser.add_argument("--ocr-every", type=int, default=15, help="OCR every Nth frame")
    parser.add_argument(
        "--save", action="store_true", help="write a raw vs output video to eval/output"
    )
    args = parser.parse_args(argv)

    labels = load_labels(args.clip)
    if args.detector == "text-layer":
        try:
            from eclipse.text_detect import detect
        except ImportError:
            print(
                "eclipse.text_detect has no detect() yet.\n"
                "Use --detector perfect or another fake to check the eval itself."
            )
            return 2
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
        str(CLIPS_DIR / labels["clip"]), labels, detect, ocr, args.ocr_every, on_frame
    )
    print_report(args.detector, report)
    if writer is not None:
        writer.release()
        print(f"  Video           {video_path.relative_to(EVAL_DIR.parent)}")
    return 1 if report["leaked_frames"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
