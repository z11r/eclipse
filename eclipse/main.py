"""Wires the pipeline. Run with: python -m eclipse.main --test-pattern

There is no detection yet, so real screen output is unscanned. That breaks the
fail-closed rule, so it needs --unsafe-passthrough. Remove that flag once the
compositor sits between capture and output.
"""

from __future__ import annotations

import argparse
import time

from eclipse.capture.pattern import TestPattern
from eclipse.capture.screen import ScreenCapture
from eclipse.output.virtual_cam import VirtualCam
from eclipse.platform import screen_capture_allowed


def run(source, sink, max_frames: int | None = None) -> list[float]:
    """Move frames from source to sink. Returns capture-to-send times in ms."""
    latencies: list[float] = []
    with source, sink:
        try:
            while max_frames is None or len(latencies) < max_frames:
                frame = source.read()
                send_started = time.monotonic()
                sink.send(frame.image)
                latencies.append((send_started - frame.ts) * 1000)
        except KeyboardInterrupt:
            pass
    return latencies


def p95(values: list[float]) -> float:
    return sorted(values)[int(len(values) * 0.95)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eclipse")
    parser.add_argument("--test-pattern", action="store_true", help="send a moving pattern")
    parser.add_argument(
        "--unsafe-passthrough",
        action="store_true",
        help="send the real screen with NO redaction; never use while live",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    if args.test_pattern:
        source = TestPattern()
    elif args.unsafe_passthrough:
        if not screen_capture_allowed():
            print(
                "Screen Recording is off for this app, so frames would be blank.\n"
                "Allow it in System Settings > Privacy & Security > Screen Recording, "
                "then restart the app."
            )
            return 1
        print("WARNING: output is NOT redacted. Do not stream this.")
        source = ScreenCapture(max_width=args.max_width)
    else:
        print(
            "Detection isn't built yet, so Eclipse won't send your real screen by default.\n"
            "Use --test-pattern to check OBS, or --unsafe-passthrough for an unredacted test."
        )
        return 2

    latencies = run(source, VirtualCam(fps=args.fps), args.max_frames)
    if latencies:
        print(f"{len(latencies)} frames, capture to send p95 {p95(latencies):.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
