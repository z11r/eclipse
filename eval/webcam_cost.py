"""What adding the webcam (--with-webcam) costs, and the frame rate that leaves.

Takes one real picture from the camera, then moves a face-sized patch over it in four
ways, so every run sees the same motion:

    still     only sensor noise
    talking   a head drifting a few pixels a frame
    leaning   big moves across the picture
    card      talking, plus a card with a fake key held up

For each it times the camera's scan of one new camera picture, and the extra the
scan thread pays per frame to put the camera in its corner:

    python -m eval.webcam_cost --screen-ms 10

The screen doesn't wait for the camera's scan (see eclipse/compositor/pip.py), so the
screen keeps its own rate, less the corner's small extra, and the corner shows as many
camera pictures as its scans keep up with, up to the camera's 30 fps. Neither counts
the two scans competing for the same cores, so treat both as ceilings and check with a
real run's log.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import sys
import time

import cv2
import numpy as np

from eclipse.capture.dual import PIP_SIZE, DualFrame, pip_layout
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.pip import PictureInPictureCompositor, PictureInPictureDetector
from eclipse.main import WEBCAM_NOISE_FLOOR, WEBCAM_OCR_THREADS
from eclipse.platform import IS_MAC, camera_backend
from eclipse.text_detect.apple_process import AppleProcessReader
from eclipse.text_detect import TextDetector
from eclipse.types import Frame

_PICTURES = 40
_SETTLE = 30  # pictures to throw away while the camera's exposure settles
_CAMERA_FPS = 30
_RENDERS = 200


class _NoScreen:
    """Stands in for the screen's detector, so only the camera's cost is timed."""

    reused_read = True
    stats: dict = {}

    def __call__(self, frame):
        return []


def camera_picture(index: int) -> np.ndarray:
    camera = cv2.VideoCapture(index, camera_backend())
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    picture = None
    for _ in range(_SETTLE):
        ok, image = camera.read()
        if ok:
            picture = image
    camera.release()
    if picture is None:
        raise SystemExit(f"camera {index} sent no pictures")
    return picture


def scenes(base: np.ndarray):
    rng = np.random.default_rng(0)
    face = cv2.GaussianBlur(rng.integers(60, 200, (360, 280, 3), dtype=np.uint8), (31, 31), 0)

    def noisy(image):
        noise = rng.integers(-5, 6, image.shape)
        return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)

    def still(i):
        return noisy(base)

    def talking(i):
        out = base.copy()
        x, y = 500 + int(20 * np.sin(i / 4)), 200 + int(10 * np.cos(i / 5))
        out[y : y + 360, x : x + 280] = face
        return noisy(out)

    def leaning(i):
        out = base.copy()
        x = 100 + (i * 37) % 800
        out[150:510, x : x + 280] = face
        out[510:720, max(x - 100, 0) : x + 380] = face[0, 0]  # shoulders
        return noisy(out)

    def card(i):
        out = talking(i)
        x, y = 300 + (i % 6) * 3, 420
        cv2.rectangle(out, (x, y), (x + 520, y + 110), (240, 240, 240), -1)
        cv2.putText(
            out, "sk-FAKE0000TESTKEY1234", (x + 12, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2,
        )
        return out

    return {"still": still, "talking": talking, "leaning": leaning, "card": card}


def screen_fps(screen_ms: float, extra_ms: float, fps: int = 60) -> float:
    return min(fps, 1000 / (screen_ms + extra_ms))


def corner_fps(camera_ms: float) -> float:
    return min(_CAMERA_FPS, 1000 / camera_ms)


def corner_extra_ms(screen: np.ndarray, scan: np.ndarray, pip: np.ndarray, at) -> float:
    """What a frame with the camera in it costs the scan thread over one without, when
    its camera picture has already been scanned: the paste, and moving the covers."""
    detect = PictureInPictureDetector(_NoScreen(), _NoScreen())
    with_camera = PictureInPictureCompositor(detect)
    plain = Compositor()

    def dual(i):
        return DualFrame(screen, 0.0, "screen", i, webcam=scan, pip=pip, pip_at=at, webcam_id=0)

    with_camera.render(dual(0), detect)  # the camera picture's scan, done once
    started = time.perf_counter()
    for i in range(1, _RENDERS + 1):
        with_camera.render(dual(i), detect)
    middle = time.perf_counter()
    for i in range(_RENDERS):
        plain.render(dual(i), _NoScreen())
    finished = time.perf_counter()
    return max((middle - started) - (finished - middle), 0) / _RENDERS * 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.webcam_cost")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--screen-ms", type=float, default=10.0,
                        help="the screen's scan time per frame, from a screen-only run's log")
    parser.add_argument("--screen", default="1920x1080",
                        help="output size, for where the camera goes; 3024x1964 for a Retina run")
    parser.add_argument("--webcam-ocr", choices=["rapid", "apple"], default="rapid")
    parser.add_argument("--webcam-scan", choices=["output", "capture"], default="capture")
    args = parser.parse_args(argv)
    if args.webcam_ocr == "apple" and not IS_MAC:
        parser.error("Apple webcam OCR needs macOS")
    with ExitStack() as resources:
        reader = None
        if args.webcam_ocr == "apple":
            reader = AppleProcessReader()
            resources.callback(reader.close)
        return measure(args, reader)


def measure(args, reader):
    width, height = (int(n) for n in args.screen.split("x"))
    screen = np.full((height, width, 3), 230, np.uint8)
    base = camera_picture(args.camera)
    at, shown, scanned = pip_layout(screen.shape, base.shape, PIP_SIZE,
                                    (1920,1080) if args.webcam_scan == "output" else None)
    print(
        f"camera {base.shape[1]}x{base.shape[0]}, scanned at {scanned[0]}x{scanned[1]}, "
        f"shown at {shown[0]}x{shown[1]}; screen scan {args.screen_ms:.1f} ms per frame"
    )
    print(f"{'motion':<8} {'scan ms':>8} {'p95':>7} {'reads':>6} {'area':>5}  corner fps")
    for name, make in scenes(base).items():
        detect = TextDetector(reader=reader, noise_floor=WEBCAM_NOISE_FLOOR,
                              ocr_threads=WEBCAM_OCR_THREADS)
        pictures = [
            cv2.resize(make(i), scanned, interpolation=cv2.INTER_AREA)
            for i in range(_PICTURES + 1)
        ]
        detect(Frame(pictures[0], 0.0, "webcam", 0))  # the first full read, and the model
        before = dict(detect.stats)
        times = []
        for i, picture in enumerate(pictures[1:], 1):
            started = time.perf_counter()
            detect(Frame(picture, 0.0, "webcam", i))
            times.append((time.perf_counter() - started) * 1000)
        times.sort()
        stats = {key: value - before[key] for key, value in detect.stats.items()}
        scans = max(stats["full_scans"] + stats["partial_scans"], 1)
        mean = float(np.mean(times))
        print(
            f"{name:<8} {mean:8.1f} {times[int(len(times) * 0.95)]:7.1f} "
            f"{stats['reads']:6d} {stats['scanned_share'] / scans:5.0%}  "
            f"{corner_fps(mean):.0f}"
        )
    scan = cv2.resize(base, scanned, interpolation=cv2.INTER_AREA)
    pip = cv2.resize(scan, shown, interpolation=cv2.INTER_AREA)
    extra = corner_extra_ms(screen, scan, pip, at)
    print(
        f"corner extra per frame {extra:.2f} ms, so the screen runs at "
        f"{screen_fps(args.screen_ms, extra):.0f} fps before CPU contention/stale-camera waits"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
