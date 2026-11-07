"""OBS Virtual Camera on macOS, written to directly so each frame carries its due time.

pyvirtualcam stamps a frame with the moment it hands it over. OBS's Video Capture
Device, with Use Buffering on, shows frames by that stamp, so a late wake-up in Python
turns into a doubled frame and then a skipped one. Here the caller says which output
tick the frame belongs to, and OBS paces by that. With Use Buffering off, OBS ignores
stamps and shows whatever arrived last, so this changes nothing there.

Speaks the same protocol as pyvirtualcam's macOS backend: find OBS's camera by its
fixed UID, start its second (sink) stream, and put UYVY sample buffers on that
stream's queue. The camera extension forwards them to apps reading the camera.
Imported only through eclipse.platform.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import POINTER, Structure, byref, c_bool, c_int32, c_int64, c_size_t, c_uint32, c_uint64, c_void_p

import numpy as np

from eclipse.output.virtual_cam import bgr_to_uyvy

# Fixed by OBS's camera extension. pyvirtualcam relies on the same value.
OBS_DEVICE_UID = "7626645E-4425-469E-9D8B-97E0FA59AC75"
_DRAIN_SECONDS = 0.1  # how long close() lets the extension take queued frames

# Stamps run this far behind the clock. OBS keeps one timeline per source and only
# starts a new one when a stamp jumps by more than 2 seconds (MAX_TS_VAR in libobs).
# While Eclipse is off, OBS's camera extension sends its placeholder stamped with the
# real time, so without the jump OBS would line our ticks up against the placeholder's
# and leave a random margin for late frames. With it, OBS starts its timeline on our
# first frame, which leaves late frames a whole frame of margin.
_STAMP_OFFSET_SECONDS = 3.0


def _framework(name: str) -> ctypes.CDLL:
    return ctypes.CDLL(f"/System/Library/Frameworks/{name}.framework/{name}")


def _code(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "big")


class _Address(Structure):
    _fields_ = [("selector", c_uint32), ("scope", c_uint32), ("element", c_uint32)]


class _Time(Structure):  # CMTime
    _fields_ = [("value", c_int64), ("timescale", c_int32), ("flags", c_uint32), ("epoch", c_int64)]


class _Timing(Structure):  # CMSampleTimingInfo
    _fields_ = [("duration", _Time), ("presentation", _Time), ("decode", _Time)]


_SYSTEM_OBJECT = 1
_GLOBAL = _code("glob")
_DEVICES = _code("dev#")
_DEVICE_UID = _code("uid ")
_STREAMS = _code("stm#")
_UYVY = _code("2vuy")
_UTF8 = 0x08000100
_SINT32 = 3
_TIME_VALID = 1
_NANOSECONDS = 1_000_000_000

_QueueAltered = ctypes.CFUNCTYPE(None, c_uint32, c_void_p, c_void_p)

_cf = _framework("CoreFoundation")
_cv = _framework("CoreVideo")
_cm = _framework("CoreMedia")
_cmio = _framework("CoreMediaIO")


def _declare(function, restype, *argtypes) -> None:
    function.restype = restype
    function.argtypes = list(argtypes)


_declare(_cf.CFRelease, None, c_void_p)
_declare(_cf.CFStringGetCString, c_bool, c_void_p, ctypes.c_char_p, ctypes.c_long, c_uint32)
_declare(_cf.CFNumberCreate, c_void_p, c_void_p, ctypes.c_long, c_void_p)
_declare(
    _cf.CFDictionaryCreate,
    c_void_p,
    c_void_p,
    POINTER(c_void_p),
    POINTER(c_void_p),
    ctypes.c_long,
    c_void_p,
    c_void_p,
)
_declare(_cmio.CMIOObjectGetPropertyDataSize, c_int32, c_uint32, POINTER(_Address), c_uint32, c_void_p, POINTER(c_uint32))
_declare(
    _cmio.CMIOObjectGetPropertyData,
    c_int32,
    c_uint32,
    POINTER(_Address),
    c_uint32,
    c_void_p,
    c_uint32,
    POINTER(c_uint32),
    c_void_p,
)
_declare(_cmio.CMIOStreamCopyBufferQueue, c_int32, c_uint32, _QueueAltered, c_void_p, POINTER(c_void_p))
_declare(_cmio.CMIODeviceStartStream, c_int32, c_uint32, c_uint32)
_declare(_cmio.CMIODeviceStopStream, c_int32, c_uint32, c_uint32)
_declare(_cv.CVPixelBufferPoolCreate, c_int32, c_void_p, c_void_p, c_void_p, POINTER(c_void_p))
_declare(_cv.CVPixelBufferPoolCreatePixelBuffer, c_int32, c_void_p, c_void_p, POINTER(c_void_p))
_declare(_cv.CVPixelBufferPoolRelease, None, c_void_p)
_declare(_cv.CVPixelBufferLockBaseAddress, c_int32, c_void_p, c_uint64)
_declare(_cv.CVPixelBufferUnlockBaseAddress, c_int32, c_void_p, c_uint64)
_declare(_cv.CVPixelBufferGetBaseAddress, c_void_p, c_void_p)
_declare(_cv.CVPixelBufferGetBytesPerRow, c_size_t, c_void_p)
_declare(_cv.CVPixelBufferRelease, None, c_void_p)
_declare(_cm.CMVideoFormatDescriptionCreate, c_int32, c_void_p, c_uint32, c_int32, c_int32, c_void_p, POINTER(c_void_p))
_declare(
    _cm.CMSampleBufferCreateForImageBuffer,
    c_int32,
    c_void_p,
    c_void_p,
    c_bool,
    c_void_p,
    c_void_p,
    c_void_p,
    POINTER(_Timing),
    POINTER(c_void_p),
)
_declare(_cm.CMSimpleQueueEnqueue, c_int32, c_void_p, c_void_p)
_declare(_cm.CMSimpleQueueGetCount, c_int32, c_void_p)


def _symbol(library: ctypes.CDLL, name: str) -> int:
    """The value of a CFStringRef constant such as kCVPixelBufferWidthKey."""
    return c_void_p.in_dll(library, name).value


def _property(object_id: int, selector: int, item_type) -> list:
    """All items of one CMIO property, such as the device list."""
    address = _Address(selector, _GLOBAL, 0)
    size = c_uint32(0)
    if _cmio.CMIOObjectGetPropertyDataSize(object_id, byref(address), 0, None, byref(size)):
        return []
    count = size.value // ctypes.sizeof(item_type)
    items = (item_type * count)()
    used = c_uint32(0)
    if _cmio.CMIOObjectGetPropertyData(object_id, byref(address), 0, None, size, byref(used), items):
        return []
    return list(items)[: used.value // ctypes.sizeof(item_type)]


def _device_uid(device: int) -> str | None:
    uid = _property(device, _DEVICE_UID, c_void_p)
    if not uid or not uid[0]:
        return None
    text = ctypes.create_string_buffer(256)
    found = _cf.CFStringGetCString(uid[0], text, len(text), _UTF8)
    _cf.CFRelease(uid[0])
    return text.value.decode() if found else None


def _number(value: int) -> int:
    number = c_int32(value)
    return _cf.CFNumberCreate(None, _SINT32, byref(number))


def _dictionary(pairs: list[tuple[int, int]]) -> int:
    keys = (c_void_p * len(pairs))(*[key for key, _ in pairs])
    values = (c_void_p * len(pairs))(*[value for _, value in pairs])
    key_callbacks = ctypes.addressof(c_void_p.in_dll(_cf, "kCFTypeDictionaryKeyCallBacks"))
    value_callbacks = ctypes.addressof(c_void_p.in_dll(_cf, "kCFTypeDictionaryValueCallBacks"))
    return _cf.CFDictionaryCreate(None, keys, values, len(pairs), key_callbacks, value_callbacks)


def _pixel_buffer_pool(width: int, height: int) -> int:
    numbers = [_number(_UYVY), _number(width), _number(height)]
    surface = _dictionary([])
    attributes = _dictionary(
        [
            (_symbol(_cv, "kCVPixelBufferPixelFormatTypeKey"), numbers[0]),
            (_symbol(_cv, "kCVPixelBufferWidthKey"), numbers[1]),
            (_symbol(_cv, "kCVPixelBufferHeightKey"), numbers[2]),
            (_symbol(_cv, "kCVPixelBufferIOSurfacePropertiesKey"), surface),
        ]
    )
    pool = c_void_p()
    status = _cv.CVPixelBufferPoolCreate(None, None, attributes, byref(pool))
    for item in numbers + [surface, attributes]:
        _cf.CFRelease(item)
    if status or not pool.value:
        raise RuntimeError(f"Couldn't make pixel buffers for OBS Virtual Camera (error {status})")
    return pool.value


class ObsMacCamera:
    """Sends frames to OBS Virtual Camera, each stamped with the tick it is due."""

    device = "OBS Virtual Camera"
    prepare = staticmethod(bgr_to_uyvy)  # done before the tick, so only a copy is left after it

    def __init__(self, width: int, height: int) -> None:
        self._width, self._height = width, height
        self._row_bytes = width * 2
        self._device = self._stream = 0
        self._queue = self._pool = self._format = None
        self._queue_altered = _QueueAltered(lambda stream, token, refcon: None)  # kept alive here
        self.stats = {"queue_full": 0, "no_buffer": 0}
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self) -> None:
        devices = [device for device in _property(_SYSTEM_OBJECT, _DEVICES, c_uint32) if _device_uid(device) == OBS_DEVICE_UID]
        if not devices:
            raise RuntimeError(
                "OBS Virtual Camera is not installed. Start Virtual Camera once in OBS 30 or "
                "later, then allow the system extension in System Settings."
            )
        streams = _property(devices[0], _STREAMS, c_uint32)
        if len(streams) < 2:  # the extension's streams are its output, then its input
            raise RuntimeError("OBS Virtual Camera has no input stream")
        self._device, stream = devices[0], streams[1]
        self._pool = _pixel_buffer_pool(self._width, self._height)
        description = c_void_p()
        status = _cm.CMVideoFormatDescriptionCreate(None, _UYVY, self._width, self._height, None, byref(description))
        if status:
            raise RuntimeError(f"Couldn't describe frames for OBS Virtual Camera (error {status})")
        self._format = description.value
        queue = c_void_p()
        status = _cmio.CMIOStreamCopyBufferQueue(stream, self._queue_altered, None, byref(queue))
        if status or not queue.value:
            raise RuntimeError(f"Couldn't reach OBS Virtual Camera's input (error {status})")
        self._queue = queue.value
        status = _cmio.CMIODeviceStartStream(self._device, stream)
        if status:
            raise RuntimeError(f"Couldn't start OBS Virtual Camera (error {status})")
        self._stream = stream

    def send(self, uyvy: np.ndarray, due: float) -> bool:
        """Queue one frame. due is the time.monotonic() tick it belongs to.

        Returns False if the frame was dropped because the camera had no room for it.
        """
        buffer = c_void_p()
        if _cv.CVPixelBufferPoolCreatePixelBuffer(None, self._pool, byref(buffer)) or not buffer.value:
            self.stats["no_buffer"] += 1
            return False
        try:
            self._copy_into(buffer.value, uyvy)
            timing = _Timing()
            # time.monotonic() is mach_absolute_time on macOS, the clock CMIO stamps use.
            stamp = round((due - _STAMP_OFFSET_SECONDS) * _NANOSECONDS)
            timing.presentation = _Time(stamp, _NANOSECONDS, _TIME_VALID, 0)
            sample = c_void_p()
            status = _cm.CMSampleBufferCreateForImageBuffer(
                None, buffer.value, True, None, None, self._format, byref(timing), byref(sample)
            )
        finally:
            _cv.CVPixelBufferRelease(buffer.value)
        if status or not sample.value:
            self.stats["no_buffer"] += 1
            return False
        if _cm.CMSimpleQueueEnqueue(self._queue, sample.value):
            _cf.CFRelease(sample.value)  # the queue only takes ownership when it accepts
            self.stats["queue_full"] += 1
            return False
        return True

    def _copy_into(self, buffer: int, uyvy: np.ndarray) -> None:
        frame = np.ascontiguousarray(uyvy)
        if frame.shape != (self._height, self._row_bytes):
            raise ValueError(f"expected UYVY {self._width}x{self._height}, got {uyvy.shape}")
        _cv.CVPixelBufferLockBaseAddress(buffer, 0)
        try:
            base = _cv.CVPixelBufferGetBaseAddress(buffer)
            stride = _cv.CVPixelBufferGetBytesPerRow(buffer)
            if stride == self._row_bytes:
                ctypes.memmove(base, frame.ctypes.data, frame.nbytes)
            else:
                for row in range(self._height):
                    ctypes.memmove(base + row * stride, frame.ctypes.data + row * self._row_bytes, self._row_bytes)
        finally:
            _cv.CVPixelBufferUnlockBaseAddress(buffer, 0)

    def close(self) -> None:
        if self._stream:
            # Let the extension take what is queued, so the final black reaches OBS.
            give_up = time.monotonic() + _DRAIN_SECONDS
            while _cm.CMSimpleQueueGetCount(self._queue) > 0 and time.monotonic() < give_up:
                time.sleep(0.005)
            _cmio.CMIODeviceStopStream(self._device, self._stream)
            self._stream = 0
        if self._queue:
            _cf.CFRelease(self._queue)  # a Copy call, so this reference is ours
            self._queue = None
        if self._format:
            _cf.CFRelease(self._format)
            self._format = None
        if self._pool:
            _cv.CVPixelBufferPoolRelease(self._pool)
            self._pool = None
