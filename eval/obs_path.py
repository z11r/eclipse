"""Proves what reaches the far end of the output path, not what we tried to send.

Sends a generated 1080p pattern through the real VirtualCam. Each frame carries its
number as large black and white blocks, which survive OBS's scaling and encoder. Record
in OBS while it runs, then decode the recording: every recorded frame says exactly which
sent frame it shows, so a repeat or a skip can't hide behind a healthy send count.

    python -m eval.obs_path send --seconds 25 --fps 60
    python -m eval.obs_path verify ~/Movies     (the newest recording there)

No screen content is involved, so the recording holds only this pattern.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eclipse.output.virtual_cam import OUTPUT_HEIGHT, OUTPUT_WIDTH, VirtualCam

BITS = 16
CHECK_BITS = 8
CELL = 64  # pixels at 1080p, so a block is still 42 pixels wide at 720p
MARGIN = 32
BAR_WIDTH = 48
BAR_STEP = 13  # pixels per frame: visible motion at 60 fps, and not a multiple of CELL
VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv"}


def check_of(number: int) -> int:
    """A small checksum, so a frame blended from two others is rejected."""
    return ((number >> 8) ^ (number & 0xFF) ^ 0x5A) & 0xFF


def _bits_of(number: int) -> list[int]:
    word = (number << CHECK_BITS) | check_of(number)
    return [(word >> shift) & 1 for shift in range(BITS + CHECK_BITS - 1, -1, -1)]


def pattern(index: int, width: int = OUTPUT_WIDTH, height: int = OUTPUT_HEIGHT) -> np.ndarray:
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    bar = index * BAR_STEP % (width - BAR_WIDTH)
    image[CELL * 3 :, bar : bar + BAR_WIDTH] = (0, 200, 255)
    for position, bit in enumerate(_bits_of(index % (1 << BITS))):
        row, column = divmod(position, 12)
        left = MARGIN + column * CELL
        top = MARGIN + row * CELL
        image[top : top + CELL, left : left + CELL] = 255 if bit else 0
    return image


def read_number(image: np.ndarray) -> int | None:
    """The frame number stamped on an image of any size, or None if it doesn't check out."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x_scale, y_scale = width / OUTPUT_WIDTH, height / OUTPUT_HEIGHT
    word = 0
    for position in range(BITS + CHECK_BITS):
        row, column = divmod(position, 12)
        center_x = int((MARGIN + column * CELL + CELL / 2) * x_scale)
        center_y = int((MARGIN + row * CELL + CELL / 2) * y_scale)
        reach_x, reach_y = max(int(CELL * x_scale / 4), 1), max(int(CELL * y_scale / 4), 1)
        patch = gray[
            center_y - reach_y : center_y + reach_y, center_x - reach_x : center_x + reach_x
        ]
        word = (word << 1) | int(patch.mean() > 128)
    number, check = word >> CHECK_BITS, word & 0xFF
    return number if check == check_of(number) else None


@dataclass
class PathReport:
    frames: int  # frames in the recording
    undecoded: int  # blended or unreadable frames
    distinct: int  # different sent frames seen
    repeats: int  # recorded frames showing the same sent frame as the one before
    skipped: int  # sent frames that never appeared
    backwards: int  # frames that went back in time
    longest_hold: int  # most recorded frames in a row showing one sent frame
    seconds: float
    fresh_per_second: list[int]

    @property
    def fresh_fps(self) -> float:
        return self.distinct / self.seconds if self.seconds else 0.0


def analyse(numbers: list[int | None], recorded_fps: float) -> PathReport:
    """Turn the numbers decoded from a recording into what a viewer actually saw."""
    known = [(index, number) for index, number in enumerate(numbers) if number is not None]
    if not known:
        return PathReport(len(numbers), len(numbers), 0, 0, 0, 0, 0, 0.0, [])
    # Leave out OBS's first and last second, where start-up and shutdown dominate.
    margin = int(recorded_fps)
    trimmed = known[margin:-margin] if len(known) > margin * 3 else known
    repeats = skipped = backwards = 0
    hold = longest = 1
    fresh_by_second: Counter = Counter()
    previous = None
    for index, number in trimmed:
        if previous is None or number != previous:
            fresh_by_second[int(index / recorded_fps)] += 1
        if previous is not None:
            if number == previous:
                repeats += 1
                hold += 1
                longest = max(longest, hold)
                continue
            hold = 1
            if number < previous:
                backwards += 1
            else:
                skipped += number - previous - 1
        previous = number
    seconds = (trimmed[-1][0] - trimmed[0][0] + 1) / recorded_fps
    whole_seconds = sorted(fresh_by_second)[1:-1] or sorted(fresh_by_second)
    return PathReport(
        frames=len(trimmed),
        undecoded=len(numbers) - len(known),
        distinct=len({number for _, number in trimmed}),
        repeats=repeats,
        skipped=skipped,
        backwards=backwards,
        longest_hold=longest,
        seconds=seconds,
        fresh_per_second=[fresh_by_second[second] for second in whole_seconds],
    )


def send(seconds: float, fps: int) -> None:
    frames = int(seconds * fps)
    print(f"Sending {frames} numbered frames at {fps} fps. Start recording in OBS now.")
    camera = VirtualCam(fps=fps)
    gaps = []
    with camera:
        last = None
        started = time.monotonic()
        for index in range(frames):
            camera.send(pattern(index))
            now = time.monotonic()
            if last is not None:
                gaps.append((now - last) * 1000)
            last = now
        elapsed = time.monotonic() - started
    gaps.sort()
    print(
        f"Sent {frames} in {elapsed:.1f} s = {frames / elapsed:.2f} fps, interval ms "
        f"p50 {gaps[len(gaps) // 2]:.2f} p95 {gaps[int(len(gaps) * 0.95)]:.2f} max {gaps[-1]:.2f}, "
        f"late ticks {camera.stats['late_ticks']}"
    )


def newest_recording(folder: str) -> str | None:
    """The most recent video file in a folder, such as OBS's recording folder."""
    videos = [
        entry
        for entry in Path(folder).expanduser().iterdir()
        if entry.is_file() and entry.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        return None
    return str(max(videos, key=lambda entry: entry.stat().st_mtime))


def verify(path: str) -> int:
    if Path(path).expanduser().is_dir():
        found = newest_recording(path)
        if found is None:
            print(f"No recording in {path}. Press Start Recording in OBS, then Stop, first.")
            return 2
        path = found
    print(f"Checking {Path(path).name}")
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print(f"Can't open {path}")
        return 2
    recorded_fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numbers = []
    while True:
        ok, image = video.read()
        if not ok:
            break
        numbers.append(read_number(image))
    video.release()
    report = analyse(numbers, recorded_fps)
    per_second = report.fresh_per_second or [0]
    print(f"Recording: {width}x{height} at {recorded_fps:.2f} fps, {len(numbers)} frames")
    print(
        f"Measured over {report.seconds:.1f} s: {report.fresh_fps:.2f} fresh frames per second "
        f"(per second: median {statistics.median(per_second)}, min {min(per_second)}, "
        f"max {max(per_second)})"
    )
    print(
        f"Repeats {report.repeats}, sent frames lost {report.skipped}, went backwards "
        f"{report.backwards}, unreadable {report.undecoded}, "
        f"longest hold {report.longest_hold} frames"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="obs_path")
    commands = parser.add_subparsers(dest="command", required=True)
    sender = commands.add_parser("send")
    sender.add_argument("--seconds", type=float, default=25)
    sender.add_argument("--fps", type=int, default=60)
    checker = commands.add_parser("verify")
    checker.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "send":
        send(args.seconds, args.fps)
        return 0
    return verify(args.path)


if __name__ == "__main__":
    sys.exit(main())
