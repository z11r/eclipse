"""Screen capture through ScreenCaptureKit, macOS's streaming capture API.

Polling costs about 23 ms a grab on a Retina screen, which caps capture near 40 fps.
This backend is handed each new picture instead, and sends nothing while the screen
is still, so a Frame from here always means the picture changed.

ScreenCaptureKit lends frames from a small pool of surfaces, and holding them stalls the
stream. So every frame is copied into our own memory inside the callback, and the surface
goes straight back. The delay buffer only ever holds those copies. Never keep a sample
buffer or pixel buffer past the callback: a second of delay would need 60 surfaces.

Written against Apple's ScreenCaptureKit documentation. Architecture reference: the
stream-output and frame-status handling in Apple's "Capturing screen content in macOS"
sample. No code was taken from it.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import CoreMedia
import dispatch
import numpy as np
import objc
import Quartz
import ScreenCaptureKit as SCK
from Foundation import NSObject

from eclipse.types import Frame

QUEUE_DEPTH = 5  # surfaces the system may hold for us; separate from the delay buffer
_INBOX_LIMIT = 4  # frames allowed to wait for read() before the oldest is dropped
_READ_WAIT = 0.1  # seconds, so a still screen can't stop the pipeline from shutting down
_START_WAIT = 5.0  # seconds to wait on the system when starting or stopping
_CLOCK_LIMIT = 1.0  # a frame time further than this from now can't be trusted
_QOS_USER_INTERACTIVE = 0x21  # from <sys/qos.h>; the bindings don't export it
# Ask for a touch more than we want. On a 120 Hz screen, refreshes land 16.6 ms or 16.7 ms
# apart, and asking for exactly 1/60 s rejects the short ones: measured 58.2 fps, 19.7 ms
# p95. Asking for 1/61 s takes every second refresh: 60.6 fps, 16.8 ms p95.
_INTERVAL_HEADROOM = 1


class CaptureStopped(RuntimeError):
    """The stream ended. The caller must not treat later frames as coming from it."""


def even_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Shrink to max_width, rounding down to even numbers for the virtual camera."""
    scale = min(1.0, max_width / width)
    return int(width * scale) & ~1, int(height * scale) & ~1


def dirty_rects(
    rects: list[tuple[float, float, float, float]] | None, width: int, height: int
) -> tuple[tuple[int, int, int, int], ...] | None:
    """Clean the system's changed areas into whole pixels, or None if unusable.

    A frame with no usable list says nothing about what changed, so the caller must
    fall back to comparing the whole frame.
    """
    if not rects:
        return None
    cleaned = []
    for x, y, w, h in rects:
        if not (w > 0 and h > 0):
            return None
        left, top = max(int(x), 0), max(int(y), 0)
        right, bottom = min(int(x + w + 0.999), width), min(int(y + h + 0.999), height)
        if right <= left or bottom <= top:
            return None
        cleaned.append((left, top, right - left, bottom - top))
    return tuple(cleaned)


class _StreamDelegate(
    NSObject, protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]
):
    def initWithOwner_(self, owner):
        self = objc.super(_StreamDelegate, self).init()
        if self is not None:
            self.owner = owner
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample, kind):
        if kind != SCK.SCStreamOutputTypeScreen:
            return
        try:
            with objc.autorelease_pool():
                self.owner.on_sample(sample)
        except BaseException as error:  # an exception here would be lost in the system's queue
            self.owner.on_failure(error)

    def stream_didStopWithError_(self, stream, error):
        self.owner.on_failure(RuntimeError("the stream stopped"))


class ScreenCaptureKitCapture:
    """Frames from the main display. Call read() from one thread only."""

    paced = True  # read() waits for the system's next picture, so nothing else need

    def __init__(
        self,
        max_width: int = 1920,
        retina: bool = False,
        fps: int = 60,
        queue_depth: int = QUEUE_DEPTH,
        first_id: int = 0,
        collect_dirty: bool = False,
    ) -> None:
        self._max_width = max_width
        self._retina = retina
        self._fps = fps
        self._queue_depth = queue_depth
        self._next_id = first_id
        self._collect_dirty = collect_dirty  # for capture_bench's check only
        self.dirty_by_id: dict[int, tuple] = {}
        self._inbox: deque[Frame] = deque()
        self._ready = threading.Condition()
        self._failure: BaseException | None = None
        self._stream = None
        self._delegate = None
        self._geometry = None
        self._announced = False
        self.size = (0, 0)
        self.stats = {
            "grab_seconds": 0.0,  # callback time outside the copy
            "convert_seconds": 0.0,  # copying out of the surface and dropping alpha
            "delivery_seconds": 0.0,  # display time to the callback starting
            "source_frames": 0,
            "not_complete": 0,  # the system's idle, blank and suspended notices
            "dropped": 0,  # never read in time, which shows as a gap in frame ids
            "dirty_frames": 0,  # frames that came with usable changed areas
        }

    @property
    def next_id(self) -> int:
        return self._next_id

    def __enter__(self) -> "ScreenCaptureKitCapture":
        display = self._main_display()
        mode = Quartz.CGDisplayCopyDisplayMode(display.displayID())
        pixel_width = Quartz.CGDisplayModeGetPixelWidth(mode)
        pixel_height = Quartz.CGDisplayModeGetPixelHeight(mode)
        if not self._retina:
            pixel_width, pixel_height = display.width(), display.height()
        self.size = even_size(pixel_width, pixel_height, self._max_width)

        config = SCK.SCStreamConfiguration.alloc().init()
        config.setWidth_(self.size[0])
        config.setHeight_(self.size[1])
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self._fps + _INTERVAL_HEADROOM))
        config.setPixelFormat_(Quartz.kCVPixelFormatType_32BGRA)
        config.setQueueDepth_(self._queue_depth)
        config.setShowsCursor_(False)  # as the polling backend does; moves aren't changes
        config.setColorSpaceName_(Quartz.kCGColorSpaceSRGB)  # what OBS assumes

        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        self._delegate = _StreamDelegate.alloc().initWithOwner_(self)
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, self._delegate
        )
        attributes = dispatch.dispatch_queue_attr_make_with_qos_class(
            dispatch.DISPATCH_QUEUE_SERIAL, _QOS_USER_INTERACTIVE, 0
        )
        queue = dispatch.dispatch_queue_create(b"eclipse.capture", attributes)
        added, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._delegate, SCK.SCStreamOutputTypeScreen, queue, None
        )
        if not added or error is not None:
            raise CaptureStopped("couldn't attach to the stream")
        self._call(self._stream.startCaptureWithCompletionHandler_, "start")
        return self

    def __exit__(self, *exc) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                self._call(stream.stopCaptureWithCompletionHandler_, "stop")
            except CaptureStopped:
                pass  # already stopped, which is the state we want

    def read(self) -> Frame | None:
        """The next picture, or None if the screen hasn't changed for a moment."""
        with self._ready:
            if not self._inbox and self._failure is None:
                self._ready.wait(_READ_WAIT)
            if self._inbox:
                return self._inbox.popleft()
            if self._failure is not None:
                raise CaptureStopped("capture stopped") from self._failure
        return None

    def on_failure(self, error: BaseException) -> None:
        with self._ready:
            if self._failure is None:
                self._failure = error
            self._ready.notify_all()

    def on_sample(self, sample) -> None:
        arrived = time.monotonic()
        if not CoreMedia.CMSampleBufferIsValid(sample):
            self.stats["not_complete"] += 1
            return
        attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample, False)
        info = attachments[0] if attachments else {}
        if info.get(SCK.SCStreamFrameInfoStatus) != SCK.SCFrameStatusComplete:
            self.stats["not_complete"] += 1
            return
        pixels = CoreMedia.CMSampleBufferGetImageBuffer(sample)
        if pixels is None:
            self.stats["not_complete"] += 1
            return
        ts = CoreMedia.CMTimeGetSeconds(CoreMedia.CMSampleBufferGetPresentationTimeStamp(sample))
        if not arrived - _CLOCK_LIMIT < ts <= arrived:
            ts = arrived
        rects = self._frame_rects(info) if self._collect_dirty else None
        copying = time.monotonic()
        image = self._copy_pixels(pixels)
        copied = time.monotonic()
        self.accept(image, ts, rects)
        self.stats["delivery_seconds"] += arrived - ts
        self.stats["convert_seconds"] += copied - copying
        self.stats["grab_seconds"] += (copying - arrived) + (time.monotonic() - copied)

    def accept(self, image: np.ndarray, ts: float, dirty) -> None:
        """Queue a copied frame for read(). Frame ids count every picture, read or not."""
        image.flags.writeable = False
        frame = Frame(image, ts, "screen", self._next_id)
        if self._collect_dirty and dirty is not None:
            self.dirty_by_id[self._next_id] = dirty
        if not self._announced:
            self._announced = True
            height, width = image.shape[:2]
            print(f"Capturing {width}x{height} with ScreenCaptureKit")
        self._next_id += 1
        self.stats["source_frames"] += 1
        self.stats["dirty_frames"] += dirty is not None
        with self._ready:
            if len(self._inbox) >= _INBOX_LIMIT:
                self._inbox.popleft()
                self.stats["dropped"] += 1
            self._inbox.append(frame)
            self._ready.notify()

    def _copy_pixels(self, pixels) -> np.ndarray:
        Quartz.CVPixelBufferLockBaseAddress(pixels, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            width = Quartz.CVPixelBufferGetWidth(pixels) & ~1
            height = Quartz.CVPixelBufferGetHeight(pixels) & ~1
            stride = Quartz.CVPixelBufferGetBytesPerRow(pixels)
            memory = Quartz.CVPixelBufferGetBaseAddress(pixels).as_buffer(stride * height)
            view = np.ndarray((height, width, 4), np.uint8, memory, strides=(stride, 4, 1))
            return cv2.cvtColor(view, cv2.COLOR_BGRA2BGR)  # a new array, so the surface is free
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixels, Quartz.kCVPixelBufferLock_ReadOnly)

    def _frame_rects(self, info):
        """Changed areas for this frame, or None if the picture's geometry just moved.

        Not used by the pipeline. Measured on this screen they were always correct but
        covered 72% of the frame when 1.1% had changed, and comparing a whole frame costs
        about a millisecond, so they would save little and add a way to miss text.
        """
        content = info.get(SCK.SCStreamFrameInfoContentRect)
        geometry = (
            tuple(sorted(dict(content).items())) if content else None,
            info.get(SCK.SCStreamFrameInfoScaleFactor),
            info.get(SCK.SCStreamFrameInfoContentScale),
        )
        changed, self._geometry = geometry != self._geometry, geometry
        listed = info.get(SCK.SCStreamFrameInfoDirtyRects)
        if changed or listed is None:
            return None
        rects = []
        for entry in listed:
            ok, rect = Quartz.CGRectMakeWithDictionaryRepresentation(entry, None)
            if not ok:
                return None
            rects.append((rect.origin.x, rect.origin.y, rect.size.width, rect.size.height))
        return dirty_rects(rects, *self.size)

    def _main_display(self):
        Quartz.CGMainDisplayID()  # connects this process to the window server
        done = threading.Event()
        result = {}

        def finished(content, error):
            result["content"], result["error"] = content, error
            done.set()

        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(finished)
        if not done.wait(_START_WAIT) or result.get("content") is None:
            raise CaptureStopped("couldn't list the displays")
        main = Quartz.CGMainDisplayID()
        for display in result["content"].displays():
            if display.displayID() == main:
                return display
        raise CaptureStopped("main display not found")

    def _call(self, method, name: str) -> None:
        done = threading.Event()
        result = {}

        def finished(error):
            result["error"] = error
            done.set()

        method(finished)
        if not done.wait(_START_WAIT) or result.get("error") is not None:
            raise CaptureStopped(f"couldn't {name} the stream")
