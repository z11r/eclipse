"""Chooses a capture backend, and survives losing it.

The streaming backend is preferred but can fail to start, or stop mid-run if the display
changes. Each failure falls back to the polling backend, so the stream keeps going and
every frame still goes through the scanner.
"""

from __future__ import annotations

from typing import Callable

from eclipse.types import Frame


class PreferredCapture:
    """Reads from `preferred`, switching to `make_fallback(next_id)` if it fails."""

    def __init__(self, preferred, make_fallback: Callable[[int], object]) -> None:
        self._active = preferred
        self._make_fallback = make_fallback
        self._fell_back = False

    @property
    def stats(self) -> dict:
        return self._active.stats

    @property
    def paced(self) -> bool:
        return getattr(self._active, "paced", False)

    @property
    def fell_back(self) -> bool:
        return self._fell_back

    def __enter__(self) -> "PreferredCapture":
        try:
            self._active.__enter__()
        except Exception as error:
            self._fall_back(error, 0)
        return self

    def __exit__(self, *exc) -> None:
        self._active.__exit__(*exc)

    def read(self) -> Frame | None:
        try:
            return self._active.read()
        except Exception as error:
            if self._fell_back:
                raise
            failed = self._active
            next_id = getattr(failed, "next_id", 0)
            failed.__exit__(None, None, None)
            self._fall_back(error, next_id)
            return self._active.read()

    def _fall_back(self, error: Exception, next_id: int) -> None:
        # Type only: an error message could hold text from the screen.
        print(f"Streaming capture unavailable ({type(error).__name__}), polling instead")
        self._fell_back = True
        self._active = self._make_fallback(next_id)
        self._active.__enter__()
