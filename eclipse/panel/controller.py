"""Minimal panel/pipeline contract and safe-to-test mock.

Real pipeline implementation will be supplied by main.py callbacks. No OBS,
webcam, or screen operations happen in this mock.
"""
from pathlib import Path
import json
import os
from threading import RLock
from typing import Protocol
from fastapi import HTTPException
from .schemas import (
    ALL_CATEGORIES, TEXT_CATEGORIES, VISUAL_CATEGORIES, CatchLog,
    DetectorHealth, ModeConfig, PanelConfig, PanelState,
)
from .events import EventHistory


class PipelineController(Protocol):
    def get_state(self) -> PanelState: ...
    def set_config(self, config: PanelConfig) -> PanelState: ...
    def get_events(self, after: int = 0) -> CatchLog: ...
    def panic(self) -> PanelState: ...
    def resume(self) -> PanelState: ...
    def set_mode(self, mode: ModeConfig) -> PanelState: ...


class MockPipelineController:
    """Simulates controls without ever claiming real stream protection."""
    def __init__(self, config_path: Path | None = None):
        self._lock = RLock()
        self._config_path = config_path or (Path.home() / ".eclipse" / "config.json")
        self.events = EventHistory()
        self._state = PanelState(
            detector_health={**{c: DetectorHealth.UNAVAILABLE for c in ALL_CATEGORIES}},
            enabled=list(ALL_CATEGORIES),
            keywords=self._load_keywords(),
        )

    def _load_keywords(self) -> list[str]:
        try:
            if not self._config_path.exists():
                return []
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            return PanelConfig(enabled=list(ALL_CATEGORIES),
                               keywords=data.get("keywords", [])).keywords
        except (ValueError, OSError, TypeError):
            return []

    def _save_keywords(self, keywords: list[str]) -> None:
        try:
            self._config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temp = self._config_path.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as stream:
                os.chmod(temp, 0o600)
                json.dump({"keywords": keywords}, stream)
            temp.replace(self._config_path)
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Unable to persist private keywords") from exc

    def get_state(self) -> PanelState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def set_config(self, config: PanelConfig) -> PanelState:
        with self._lock:
            # Persist first so an I/O failure cannot partially apply config.
            self._save_keywords(config.keywords)
            self._state.enabled = list(config.enabled)
            self._state.keywords = list(config.keywords)
            return self._state.model_copy(deep=True)

    def get_events(self, after: int = 0) -> CatchLog:
        return self.events.since(after)

    def panic(self) -> PanelState:
        with self._lock:
            self._state.panic = True
            self._state.message = "Mock panic requested; no real output exists to black out."
            return self._state.model_copy(deep=True)

    def resume(self) -> PanelState:
        with self._lock:
            self._state.panic = False
            self._state.message = "Mock resumed; no real output exists."
            return self._state.model_copy(deep=True)

    def set_mode(self, mode: ModeConfig) -> PanelState:
        with self._lock:
            if mode.mode == "irl" and not self._state.irl_available:
                raise HTTPException(status_code=409, detail="IRL mode not available")
            self._state.mode = mode.mode
            return self._state.model_copy(deep=True)
