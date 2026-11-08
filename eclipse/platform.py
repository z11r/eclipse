"""The only module that checks which OS we are on."""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import sys

IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"


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


def stamped_virtual_camera(width: int, height: int):
    """OBS Virtual Camera that stamps each frame with the tick it is due, or None.

    Only on macOS, where Eclipse writes to OBS's camera extension itself. Elsewhere
    pyvirtualcam sends the frames and stamps them as they leave.
    """
    if not IS_MAC:
        return None
    from eclipse.output.obs_mac_camera import ObsMacCamera

    return ObsMacCamera(width, height)


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
    if sys.platform == "win32":
        return _windows_peak_memory() / 1024**2

    import resource  # not on Windows

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if IS_MAC else peak / 1024


def _windows_peak_memory() -> int:
    """Peak working set in bytes, the Windows counterpart of ru_maxrss."""
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize


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
    ACCURATE = 0  # VNRequestTextRecognitionLevelAccurate

    def __init__(self) -> None:
        if not IS_MAC:
            raise RuntimeError("Apple OCR needs macOS")

    def __call__(self, image) -> tuple[list, None]:
        import cv2

        return self.read_bgra(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))

    def read_bgra(self, pixels) -> tuple[list, None]:
        """Fast screen read, plus accurate color and contrast reads of colored areas.

        Keep the observations from each pass: a retry is supplementary and must not
        discard a secret found by the fast pass. Coordinates stay in input pixels.
        Both direct calls and shared-memory OCR workers enter through this method.
        """
        from eclipse.text_detect.styled_text import colour_regions, contrast_bgra

        words, _ = self._read_vision(pixels, self.FAST)
        words = self._refine_secret_boxes(pixels, words)
        for left, top, right, bottom in colour_regions(pixels):
            original = pixels[top:bottom, left:right]
            # A single selected channel may lose some foreground/background
            # combinations. Keep accurate original-color observations as well.
            for crop in (original, contrast_bgra(original)):
                extra, _ = self._read_vision(crop, self.ACCURATE)
                extra = self._refine_secret_boxes(crop, extra)
                for corners, text, confidence in extra:
                    translated = [[x + left, y + top] for x, y in corners]
                    if not any(text == old[1] and translated == old[0] for old in words):
                        words.append((translated, text, confidence))
        return words, None

    def _refine_secret_boxes(self, pixels, words):
        """Retry overly broad email/credential boxes; replace only on confirmation.

        Fast Vision sometimes returns a whole line for a word range, or one
        failed word range makes the observation fall back to its complete line.
        An accurate local read can supply a smaller credential box. Other matched
        secrets must also survive before the original observation is replaced.
        """
        import math
        from eclipse.text_detect.detection import find_matches

        height, width = pixels.shape[:2]
        refined = []
        for polygon, text, confidence in words:
            matches = find_matches(text)
            if not any(category in {'email', 'api_key'} for category, _, _ in matches):
                refined.append((polygon, text, confidence))
                continue
            x1 = max(0, math.floor(min(p[0] for p in polygon)) - 8)
            y1 = max(0, math.floor(min(p[1] for p in polygon)) - 8)
            x2 = min(width, math.ceil(max(p[0] for p in polygon)) + 8)
            y2 = min(height, math.ceil(max(p[1] for p in polygon)) + 8)
            # A normal single-word box needs no second pass. A line-wide box
            # for a short email is suspicious even if Vision returned one word.
            broad = ' ' in text or x2-x1 > max(len(text), 1) * max(y2-y1-16, 1)
            if not broad or x2 <= x1 or y2 <= y1:
                refined.append((polygon, text, confidence))
                continue
            extra, _ = self._read_vision(pixels[y1:y2, x1:x2], self.ACCURATE)
            wanted = {(kind, text[start:end].casefold()) for kind, start, end in matches}
            found = {(kind, value[start:end].casefold())
                     for _, value, _ in extra for kind, start, end in find_matches(value)}
            secret_boxes = [box for box, value, _ in extra
                           if any(kind in {'email', 'api_key'} for kind, _, _ in find_matches(value))]
            narrow = secret_boxes and all(
                max(p[0] for p in box)-min(p[0] for p in box) < (x2-x1-16)*0.9
                for box in secret_boxes
            )
            # Preserve the surrounding text too: configured keyword matching
            # happens downstream, and the reader does not know those keywords.
            original_text = re.sub(r"\s+", "", text).casefold()
            retry_text = re.sub(r"\s+", "", ''.join(value for _, value, _ in extra)).casefold()
            if wanted <= found and narrow and original_text in retry_text:
                refined.extend(([[x+x1, y+y1] for x, y in box], value, score)
                               for box, value, score in extra)
            else:
                refined.append((polygon, text, confidence))
        return refined

    def _read_vision(self, pixels, level) -> tuple[list, None]:
        """One native Vision request; errors propagate to the fail-closed pipeline."""
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
            request.setRecognitionLevel_(level)
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


class WindowsOCRReader:
    """Read local screen text with Windows OCR and return each word's box."""

    def __init__(self) -> None:
        if not IS_WINDOWS:
            raise RuntimeError('Windows OCR requires Windows')

    def __call__(self, image) -> tuple[list, None]:
        import asyncio
        import winocr

        async def read():
            result = await winocr.recognize_cv2(image)
            words = []
            for line in result.lines:
                for word in line.words:
                    box = word.bounding_rect
                    left, top = float(box.x), float(box.y)
                    right, bottom = left + float(box.width), top + float(box.height)
                    corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
                    # Windows OCR exposes no confidence score; don't use this as a threshold.
                    words.append((corners, str(word.text), 1.0))
            return words, None

        return asyncio.run(read())
