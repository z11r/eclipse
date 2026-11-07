"""One OCR process. Started by eclipse/text_detect/read_ahead.py, never by hand.

Takes (height, width) over a pipe, reads that much BGRA from shared memory, and answers
(True, words) or (False, None). Failure details stay here, because an error message could
hold text from the screen. Exits when the pipe closes, so it can't outlive Eclipse.

Leaves with os._exit: Vision's queues can still be releasing objects, and a normal
interpreter shutdown underneath them crashes the process.
"""

from __future__ import annotations

import mmap
import os
import sys
from multiprocessing.connection import Connection

import numpy as np

from eclipse.platform import AppleVisionReader

_READY = "ready"


def serve(channel: Connection, memory: mmap.mmap) -> None:
    reader = AppleVisionReader()
    reader.read_bgra(np.zeros((64, 64, 4), np.uint8))  # load the models before saying ready
    channel.send(_READY)
    while True:
        try:
            height, width = channel.recv()
        except EOFError:
            return
        pixels = np.ndarray((height, width, 4), np.uint8, memory)
        try:
            words, _ = reader.read_bgra(pixels)
        except Exception:
            channel.send((False, None))
            continue
        channel.send((True, words))


def main() -> None:
    pipe_fd, memory_fd, size = (int(value) for value in sys.argv[1:4])
    os.setpgrp()  # a Ctrl-C in the terminal goes to Eclipse, which shuts workers down itself
    try:
        serve(Connection(pipe_fd), mmap.mmap(memory_fd, size))
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
