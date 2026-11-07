"""The only module that checks which OS we are on."""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

IS_MAC = sys.platform == "darwin"


def screen_capture_allowed() -> bool:
    """Whether this process may record the screen.

    Without permission, macOS returns blank frames instead of an error.
    """
    if not IS_MAC:
        return True
    graphics = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
    graphics.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
    return bool(graphics.CGPreflightScreenCaptureAccess())
