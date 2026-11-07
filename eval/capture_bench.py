"""Measures capture alone, with nothing else in the pipeline.

A capture stage that takes 26 ms can't supply 60 new frames a second, so this settles what
each backend can do before anything downstream is blamed. It runs the small motion strip
so there is always a new picture to capture.

    python -m eval.capture_bench --backend sck --seconds 15
    python -m eval.capture_bench --backend mss --seconds 15

Prints numbers only. Frames stay in memory for one comparison and are never saved.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

from eclipse.capture.screen import ScreenCapture

RETINA_MAX_WIDTH = 6016
SETTLE_SECONDS = 1.0


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(ordered[len(ordered) // 2], 2),
        "p95": round(ordered[int(len(ordered) * 0.95)], 2),
        "max": round(ordered[-1], 2),
    }


def _source(backend: str, fps: int, width: int, retina: bool, collect_dirty: bool):
    if backend == "mss":
        return ScreenCapture(max_width=width, retina=retina)
    from eclipse.capture.sck import ScreenCaptureKitCapture

    return ScreenCaptureKitCapture(width, retina, fps, collect_dirty=collect_dirty)


class DirtyCheck:
    """Compares the system's changed areas with the pixels that really changed."""

    UNITS = {"pixels": 1, "points": 2}

    def __init__(self) -> None:
        self.checked = 0
        self.no_list = 0
        self.gaps = 0
        self.covered = {name: 0 for name in self.UNITS}
        self.area_share = 0.0
        self.changed_share = 0.0

    def note(self, before, frame, dirty) -> None:
        if before is None:
            return
        if frame.frame_id != before.frame_id + 1:
            self.gaps += 1
            return
        if dirty is None:
            self.no_list += 1
            return
        difference = cv2.absdiff(frame.image, before.image).max(axis=2) > 0
        height, width = difference.shape
        self.checked += 1
        self.changed_share += float(difference.mean())
        for name, scale in self.UNITS.items():
            mask = np.zeros_like(difference)
            area = 0
            for x, y, w, h in dirty:
                left, top = x * scale, y * scale
                mask[top : top + h * scale, left : left + w * scale] = True
            area = float(mask.mean())
            if name == "pixels":
                self.area_share += area
            if not (difference & ~mask).any():
                self.covered[name] += 1

    def report(self) -> dict:
        checked = max(self.checked, 1)
        return {
            "pairs_checked": self.checked,
            "pairs_without_list": self.no_list,
            "pairs_with_gap": self.gaps,
            "covered_as_pixels": self.covered["pixels"],
            "covered_as_points": self.covered["points"],
            "mean_dirty_area_share": round(self.area_share / checked, 4),
            "mean_changed_share": round(self.changed_share / checked, 4),
        }


def run(args) -> dict:
    motion = None
    if not args.no_motion:
        motion = subprocess.Popen(
            [sys.executable, "-m", "eval.motion_window", "--seconds", str(args.seconds + 3)],
            stdout=subprocess.DEVNULL,
        )
    source = _source(args.backend, args.fps, args.width, not args.standard, args.check_dirty)
    check = DirtyCheck()
    intervals, ages = [], []
    fresh = frames = 0
    previous = None
    last_rows = None
    last_ts = None
    size = None
    try:
        with source:
            time.sleep(SETTLE_SECONDS)
            cpu_start = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
            started = time.monotonic()
            base = dict(source.stats)
            while (now := time.monotonic()) - started < args.seconds:
                frame = source.read()
                if frame is None:
                    continue
                arrived = time.monotonic()
                frames += 1
                size = frame.image.shape[1], frame.image.shape[0]
                ages.append((arrived - frame.ts) * 1000)
                if last_ts is not None:
                    intervals.append((frame.ts - last_ts) * 1000)
                last_ts = frame.ts
                rows = frame.image[::8]
                if last_rows is None or not np.array_equal(rows, last_rows):
                    fresh += 1
                last_rows = rows
                if args.check_dirty:
                    check.note(previous, frame, source.dirty_by_id.pop(frame.frame_id, None))
                    previous = frame
            elapsed = time.monotonic() - started
            cpu = sum(resource.getrusage(resource.RUSAGE_SELF)[:2]) - cpu_start
            spent = {name: source.stats[name] - base.get(name, 0) for name in source.stats}
    finally:
        if motion is not None:
            motion.terminate()
    result = {
        "backend": args.backend,
        "size": size,
        "seconds": round(elapsed, 1),
        "frames_per_second": round(frames / elapsed, 2),
        "fresh_per_second": round(fresh / elapsed, 2),
        "interval_ms": _percentiles(intervals),
        "ready_age_ms": _percentiles(ages),
        "grab_ms": round(spent.get("grab_seconds", 0) / max(frames, 1) * 1000, 2),
        "convert_ms": round(spent.get("convert_seconds", 0) / max(frames, 1) * 1000, 2),
        "delivery_ms": round(spent.get("delivery_seconds", 0) / max(frames, 1) * 1000, 2),
        "dropped": round(spent.get("dropped", 0)),
        "not_complete": round(spent.get("not_complete", 0)),
        "dirty_frames": round(spent.get("dirty_frames", 0)),
        "cores": round(cpu / elapsed, 2),
    }
    if args.check_dirty:
        result["dirty_check"] = check.report()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capture_bench")
    parser.add_argument("--backend", choices=["sck", "mss"], default="sck")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=RETINA_MAX_WIDTH)
    parser.add_argument("--standard", action="store_true", help="capture at standard resolution")
    parser.add_argument("--check-dirty", action="store_true", help="validate changed areas")
    parser.add_argument("--no-motion", action="store_true", help="don't show the moving strip")
    parser.add_argument("--out", help="also append the JSON result to this file")
    args = parser.parse_args(argv)
    result = run(args)
    line = json.dumps(result)
    print(line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as out:
            out.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
