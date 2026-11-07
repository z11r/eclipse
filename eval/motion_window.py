"""A small always-on-top strip that changes on every screen refresh.

ScreenCaptureKit sends nothing while the screen is still, so a capture benchmark needs
something moving. Click-through, takes no focus, and shows only a moving bar.

    python -m eval.motion_window --seconds 20
"""

from __future__ import annotations

import argparse

import AppKit
import objc
from Foundation import NSObject, NSTimer

WIDTH, HEIGHT = 720, 90
BAR = 48
STEP = 7


class _Strip(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(_Strip, self).initWithFrame_(frame)
        if self is not None:
            self.position = 0
        return self

    def drawRect_(self, rect):
        AppKit.NSColor.blackColor().setFill()
        AppKit.NSRectFill(self.bounds())
        AppKit.NSColor.orangeColor().setFill()
        AppKit.NSRectFill(((self.position, 0), (BAR, HEIGHT)))


class _Driver(NSObject):
    def initWithStrip_seconds_(self, strip, seconds):
        self = objc.super(_Driver, self).init()
        if self is not None:
            self.strip = strip
            self.ticks_left = int(seconds * 120)
        return self

    def tick_(self, timer):
        self.ticks_left -= 1
        if self.ticks_left <= 0:
            AppKit.NSApp().terminate_(None)
        self.strip.position = (self.strip.position + STEP) % (WIDTH - BAR)
        self.strip.setNeedsDisplay_(True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="motion_window")
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args(argv)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        ((40, 40), (WIDTH, HEIGHT)),
        AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    panel.setLevel_(AppKit.NSStatusWindowLevel)
    panel.setIgnoresMouseEvents_(True)
    strip = _Strip.alloc().initWithFrame_(((0, 0), (WIDTH, HEIGHT)))
    panel.setContentView_(strip)
    panel.orderFrontRegardless()
    driver = _Driver.alloc().initWithStrip_seconds_(strip, args.seconds)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1 / 120, driver, "tick:", None, True
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
