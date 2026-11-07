"""The only module that checks which OS we are on."""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import sys

IS_MAC = sys.platform == "darwin"


def screen_capture_kit_available() -> bool:
    """Whether the streaming capture backend can be used on this machine."""
    if not IS_MAC:
        return False
    try:
        import ScreenCaptureKit  # noqa: F401
    except ImportError:
        return False
    return True


def use_full_screen_resolution() -> None:
    """Capture every pixel. By default mss halves a Retina screen, which blurs text."""
    if IS_MAC:
        import mss.darwin

        bounds_ignore_framing, should_be_opaque = 1 << 0, 1 << 1
        mss.darwin.IMAGE_OPTIONS = bounds_ignore_framing | should_be_opaque


def screen_capture_allowed() -> bool:
    """Whether this process may record the screen.

    Without permission, macOS returns blank frames instead of an error.
    """
    if not IS_MAC:
        return True
    graphics = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
    graphics.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
    return bool(graphics.CGPreflightScreenCaptureAccess())


def peak_memory_mb() -> float:
    """The most memory this process has used. macOS counts in bytes, Linux in kilobytes."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if IS_MAC else peak / 1024


def keyboard_listening_allowed() -> bool:
    """Whether this process can see key presses in other apps.

    Without permission, macOS delivers nothing and raises no error, so a hotkey would
    look fine and do nothing.
    """
    if not IS_MAC:
        return True
    graphics = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
    graphics.CGPreflightListenEventAccess.restype = ctypes.c_bool
    services = ctypes.CDLL(ctypes.util.find_library("ApplicationServices"))
    services.AXIsProcessTrusted.restype = ctypes.c_bool
    return bool(graphics.CGPreflightListenEventAccess() or services.AXIsProcessTrusted())


class AppleVisionReader:
    """Apple's built-in OCR, as a reader for the text layer. macOS only.

    Takes BGR images and returns (words, None), the same shape as RapidOCR, with
    each word as (corners, text, confidence) in the image's own pixels. Words, not
    lines, because Vision shrinks wide gaps to one space, so positions can't be
    worked out from a line's character count. It needs Retina-scale frames: on
    low-resolution frames it misses most small text. Any Vision failure raises, so
    it counts as a failed scan.

    Vision allows one read at a time per process, however many threads ask. Reading
    faster takes more processes: see eclipse/text_detect/read_ahead.py.
    """

    FAST = 1  # VNRequestTextRecognitionLevelFast

    def __init__(self) -> None:
        if not IS_MAC:
            raise RuntimeError("Apple OCR needs macOS")

    def __call__(self, image) -> tuple[list, None]:
        import cv2

        return self.read_bgra(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))

    def read_bgra(self, pixels) -> tuple[list, None]:
        """Read a BGRA image. A pixel buffer reads a full frame in 58 ms, a CGImage in 88."""
        import numpy as np
        import objc
        import Quartz
        import Vision
        from Foundation import NSArray, NSString

        height, width = pixels.shape[:2]
        with objc.autorelease_pool():
            status, buffer = Quartz.CVPixelBufferCreate(
                None, width, height, Quartz.kCVPixelFormatType_32BGRA, None, None
            )
            if status != 0:
                raise RuntimeError("Apple OCR couldn't read the frame")
            Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
            try:
                stride = Quartz.CVPixelBufferGetBytesPerRow(buffer)
                memory = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(stride * height)
                target = np.ndarray((height, width, 4), np.uint8, memory, strides=(stride, 4, 1))
                target[:] = pixels
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(self.FAST)
            request.setUsesLanguageCorrection_(False)  # it would "fix" random keys
            request.setMinimumTextHeight_(0.0)  # the default skips small text
            # Nothing Python-backed may reach Vision, not even a str inside an NSArray. It
            # releases these on its own queues, and doing that while the interpreter shuts
            # down crashes the process, including at the end of a normal run.
            request.setRecognitionLanguages_(
                NSArray.arrayWithObject_(NSString.stringWithString_("en-US"))
            )
            handler = Vision.VNImageRequestHandler.alloc().initWithCVPixelBuffer_options_(
                buffer, None
            )
            succeeded, error = handler.performRequests_error_(
                NSArray.arrayWithObject_(request), None
            )
            if not succeeded or error is not None:
                raise RuntimeError("Apple OCR failed")

            def corners(box):  # Vision's boxes run 0 to 1 from the bottom left
                left, right = box.origin.x * width, (box.origin.x + box.size.width) * width
                top = (1 - box.origin.y - box.size.height) * height
                bottom = (1 - box.origin.y) * height
                return [[left, top], [right, top], [right, bottom], [left, bottom]]

            words = []
            for observation in request.results() or []:
                candidate = observation.topCandidates_(1)[0]
                text = str(candidate.string())
                confidence = float(candidate.confidence())
                found = []
                for match in re.finditer(r"\S+", text):
                    # Vision counts in UTF-16 units, as emoji take two.
                    start = len(text[: match.start()].encode("utf-16-le")) // 2
                    length = len(match.group().encode("utf-16-le")) // 2
                    box, _ = candidate.boundingBoxForRange_error_((start, length), None)
                    if box is None:
                        found = None  # can't place the words, so keep the whole line
                        break
                    found.append((corners(box.boundingBox()), match.group(), confidence))
                if found is None:
                    found = [(corners(observation.boundingBox()), text, confidence)]
                words.extend(found)
        return words, None
