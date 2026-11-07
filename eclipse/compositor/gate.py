"""The last check before a picture leaves. Nothing behind it can hold it open.

Panic, resume and settings changes all land here, on whoever's thread asked. The lock
is only ever held for a moment and never across any I/O, so a panic can't end up
waiting for OCR or for the camera.

Asking for a blackout and getting one are different things. panic() records the
request, and black_confirmed only turns true once a black frame has really gone out
after it. A frame already inside the camera call when the request arrives can't be
called back; everything after it is stopped.

Each settings change gets a generation number. A change that hides more raises the
required generation, so anything checked under an older one stops being allowed out,
wherever it is waiting. Resume sets a time floor instead: nothing captured before that
moment may ever appear, however it got into the queue.
"""

from __future__ import annotations

import math
import threading
import time

from eclipse.types import Rendered


class OutputGate:
    def __init__(self) -> None:
        self._changed = threading.Condition(threading.Lock())
        self._panicked = False
        self._panic_count = 0  # panics asked for so far
        self._answered = 0  # the latest of those a black frame has gone out for
        self._closed = False
        self._generation = 0
        self._required = 0
        self._floor = -math.inf
        self._invalidations = 0

    def panic(self) -> None:
        """Black out from the very next frame. Stays on until resume()."""
        with self._changed:
            if not self._panicked:
                self._panicked = True
                self._panic_count += 1

    def resume(self) -> None:
        """End a panic. Only frames captured from now on may go out."""
        with self._changed:
            self._panicked = False
            self._floor = time.monotonic()
            self._invalidations += 1

    def close(self) -> None:
        """Nothing but black from now on, for good."""
        with self._changed:
            self._closed = True

    def advance(self, stricter: bool) -> int:
        """Start a new policy generation, and return it.

        If it hides more than the last one, older output stops being allowed at once.
        """
        with self._changed:
            self._generation += 1
            if stricter:
                self._required = self._generation
                self._invalidations += 1
            return self._generation

    def allows(self, rendered: Rendered) -> bool:
        """Whether this picture may go out right now."""
        with self._changed:
            return not self._panicked and not self._closed and not self._outdated(rendered)

    def outdated(self, rendered: Rendered) -> bool:
        """Whether this picture must never go out, even once a panic ends."""
        with self._changed:
            return self._outdated(rendered)

    def captured_too_early(self, ts: float) -> bool:
        """Whether a frame predates the last resume, so it must not be shown at all."""
        with self._changed:
            return ts < self._floor

    def black_ticket(self) -> int:
        """Call before sending black, and hand this to confirm_black() once it has gone."""
        with self._changed:
            return self._panic_count

    def confirm_black(self, ticket: int) -> None:
        with self._changed:
            self._answered = max(self._answered, ticket)
            self._changed.notify_all()

    def wait_for_black(self, seconds: float) -> bool:
        """Wait a moment for the panic to be answered, and return whether it was."""
        with self._changed:
            self._changed.wait_for(lambda: self._is_black(), seconds)
            return self._is_black()

    @property
    def panicked(self) -> bool:
        with self._changed:
            return self._panicked

    @property
    def closed(self) -> bool:
        with self._changed:
            return self._closed

    @property
    def black_confirmed(self) -> bool:
        with self._changed:
            return self._is_black()

    @property
    def generation(self) -> int:
        with self._changed:
            return self._generation

    @property
    def required_generation(self) -> int:
        with self._changed:
            return self._required

    @property
    def invalidations(self) -> int:
        """Goes up each time older output stops being allowed."""
        with self._changed:
            return self._invalidations

    def _is_black(self) -> bool:
        return self._panicked and self._answered >= self._panic_count

    def _outdated(self, rendered: Rendered) -> bool:
        return rendered.generation < self._required or rendered.ts < self._floor
