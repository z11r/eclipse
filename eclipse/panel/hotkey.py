"""The global panic key. A convenience: the panel's panic button works without it."""

from __future__ import annotations

from typing import Callable

from eclipse.platform import keyboard_listening_allowed

PANIC_KEYS = "<ctrl>+<alt>+b"
VIEW_KEYS = "<ctrl>+<alt>+v"  # with --with-webcam: screen, both, webcam, and round again


def _system_listener(keys: str, on_press: Callable[[], None]):
    from pynput import keyboard

    return keyboard.GlobalHotKeys({keys: on_press})


class PanicHotkey:
    """Runs on_press on a background thread whenever the keys are pressed.

    If it can't start, available stays False, and whoever started it should say so:
    a streamer must not count on a key that does nothing.
    """

    def __init__(
        self,
        on_press: Callable[[], None],
        keys: str = PANIC_KEYS,
        make_listener: Callable = _system_listener,
        allowed: Callable[[], bool] = keyboard_listening_allowed,
    ) -> None:
        self._on_press = on_press
        self._keys = keys
        self._make_listener = make_listener
        self._allowed = allowed
        self._listener = None
        self.available = False

    def start(self) -> bool:
        if not self._allowed():
            return False
        try:
            self._listener = self._make_listener(self._keys, self._on_press)
            self._listener.daemon = True
            self._listener.start()
        except Exception:
            self._listener = None
            return False
        self.available = True
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        self._listener = None
        self.available = False
