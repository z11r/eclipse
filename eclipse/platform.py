"""The only module that checks which OS we are on."""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import sys

IS_MAC = sys.platform == "darwin"


def use_full_screen_resolution() -> None:
    """Capture every pixel. By default mss halves a Retina screen, which blurs text."""
    if IS_MAC:
        import mss.darwin

        bounds_ignore_framing, should_be_opaque = 1 << 0, 1 << 1
        mss.darwin.IMAGE_OPTIONS = bounds_ignore_framing | should_be_opaque


def camera_backend() -> int:
    """The OpenCV camera driver to use.

    On Windows this is Media Foundation. It takes a few seconds to open, but
    DirectShow gets only black frames at 1 fps from cameras behind a vendor camera
    service, such as Acer's.
    """
    import cv2

    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def screen_capture_allowed() -> bool:
    """Whether this process may record the screen.

    Without permission, macOS returns blank frames instead of an error.
    """
    if not IS_MAC:
        return True
    graphics = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
    graphics.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
    return bool(graphics.CGPreflightScreenCaptureAccess())


class AppleVisionReader:
    """Apple's built-in OCR, as a reader for the text layer. macOS only.

    Takes BGR images and returns (words, None), the same shape as RapidOCR, with
    each word as (corners, text, confidence) in the image's own pixels. Words, not
    lines, because Vision shrinks wide gaps to one space, so positions can't be
    worked out from a line's character count. It needs Retina-scale frames: on
    low-resolution frames it misses most small text. Any Vision failure raises, so
    it counts as a failed scan.
    """

    FAST = 1  # VNRequestTextRecognitionLevelFast

    def __init__(self) -> None:
        if not IS_MAC:
            raise RuntimeError("Apple OCR needs macOS")

    def __call__(self, image) -> tuple[list, None]:
        import cv2
        import objc
        import Quartz
        import Vision
        from Foundation import NSData

        height, width = image.shape[:2]
        with objc.autorelease_pool():
            # Hand Vision the pixels directly. Encoding a Retina frame costs 17 ms.
            pixels = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            data = NSData.dataWithBytes_length_(pixels.tobytes(), pixels.size)
            picture = Quartz.CGImageCreate(
                width,
                height,
                8,
                32,
                width * 4,
                Quartz.CGColorSpaceCreateDeviceRGB(),
                Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaNoneSkipFirst,
                Quartz.CGDataProviderCreateWithCFData(data),
                None,
                False,
                Quartz.kCGRenderingIntentDefault,
            )
            if picture is None:
                raise RuntimeError("Apple OCR couldn't read the frame")
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(self.FAST)
            request.setUsesLanguageCorrection_(False)  # it would "fix" random keys
            request.setMinimumTextHeight_(0.0)  # the default skips small text
            request.setRecognitionLanguages_(["en-US"])
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(picture, None)
            succeeded, error = handler.performRequests_error_([request], None)
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
