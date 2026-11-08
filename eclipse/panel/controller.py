"""The panel's view of the pipeline: one real controller, and a mock for simulation.

LivePipelineController is the single authority over a running pipeline. The mock is
kept for explicit simulation and tests. No OBS, webcam or screen operations happen in
it, and it never claims real stream protection.
"""
from pathlib import Path
import json
import os
from threading import Lock, RLock
from time import monotonic
from typing import Callable, Iterable, Protocol
from fastapi import HTTPException
from .schemas import (
    ALL_CATEGORIES, TEXT_CATEGORIES, VISUAL_CATEGORIES, CatchLog,
    DetectorHealth, ModeConfig, PanelConfig, PanelState, ViewConfig,
)
from .events import EventHistory
from eclipse.capture.dual import drops_a_source

PANIC_WAIT_SECONDS = 0.2  # a panic reports confirmed black if it arrives this soon
APPLY_WAIT_SECONDS = 2.0  # a settings change reports its outcome if it arrives this soon
SCAN_LATE_SECONDS = 5.0  # with no scan finished for this long, the detector is degraded


def default_config_path() -> Path:
    return Path.home() / ".eclipse" / "config.json"


class KeywordStore:
    """Private keywords on disk, written so a failure never leaves half a file."""

    def __init__(self, path: Path | None = None):
        self.path = path or default_config_path()

    def load(self) -> list[str]:
        try:
            if not self.path.exists():
                return []
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return PanelConfig(enabled=list(ALL_CATEGORIES),
                               keywords=data.get("keywords", [])).keywords
        except (ValueError, OSError, TypeError):
            return []

    def stage(self, keywords: list[str]) -> Path:
        """Write the keywords beside the real file, ready for commit()."""
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as stream:
                os.chmod(temp, 0o600)
                json.dump({"keywords": keywords}, stream)
            return temp
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Unable to persist private keywords") from exc

    def commit(self, staged: Path) -> None:
        try:
            staged.replace(self.path)
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Unable to persist private keywords") from exc

    def discard(self, staged: Path) -> None:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass

    def save(self, keywords: list[str]) -> None:
        self.commit(self.stage(keywords))


class PipelineController(Protocol):
    pipeline_mode: str

    def get_state(self) -> PanelState: ...
    def set_config(self, config: PanelConfig) -> PanelState: ...
    def get_events(self, after: int = 0) -> CatchLog: ...
    def panic(self) -> PanelState: ...
    def resume(self) -> PanelState: ...
    def set_mode(self, mode: ModeConfig) -> PanelState: ...
    def set_view(self, view: ViewConfig) -> PanelState: ...


_VIEWS = ["screen", "both", "webcam"]


class MockPipelineController:
    """Simulates controls without ever claiming real stream protection."""
    pipeline_mode = "mock"

    def __init__(self, config_path: Path | None = None):
        self._lock = RLock()
        self._store = KeywordStore(config_path)
        self._config_path = self._store.path
        self.events = EventHistory()
        self._state = PanelState(
            detector_health={**{c: DetectorHealth.UNAVAILABLE for c in ALL_CATEGORIES}},
            enabled=list(ALL_CATEGORIES),
            keywords=self._store.load(),
            view="both",
            views=list(_VIEWS),
        )

    def get_state(self) -> PanelState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def set_view(self, view: ViewConfig) -> PanelState:
        with self._lock:
            self._state.view = view.view
            return self._state.model_copy(deep=True)

    def set_config(self, config: PanelConfig) -> PanelState:
        with self._lock:
            # Persist first so an I/O failure cannot partially apply config.
            self._store.save(config.keywords)
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


class LivePipelineController:
    """The one authority over a running pipeline.

    Panic goes straight to the output gate and takes no lock shared with anything slow,
    so a stuck scan or a settings change can't hold it up. Settings reach the real
    detector between scans, so no frame is ever scanned with half of them.

    configure takes (keywords, enabled). Pass None if this run's detector has none, as
    with the test pattern, and settings changes are refused rather than ignored.

    views is the capture that switches between screen, both and webcam (a DualCapture),
    or None for a run with one source. A switch that drops a source cuts the output at
    once: black goes out until the new view's first frame has been scanned, about the
    delay later, so nothing of the dropped source shows after the switch.
    """
    pipeline_mode = "live"

    def __init__(
        self,
        pipeline,
        configure: Callable[[Iterable[str], Iterable[str]], None] | None,
        config_path: Path | None = None,
        apply_wait: float = APPLY_WAIT_SECONDS,
        views=None,
    ):
        self._pipeline = pipeline
        self._views = views
        self._view_lock = Lock()  # one switch at a time; panic never takes this
        self._gate = pipeline.gate
        self._configure = configure
        self._apply_wait = apply_wait
        self._store = KeywordStore(config_path)
        self._change_lock = Lock()  # one settings change at a time; panic never takes this
        self._lock = Lock()
        self._enabled = list(ALL_CATEGORIES)
        self._keywords = self._store.load()
        self._caught: set[str] = set()
        self.events = EventHistory()
        pipeline.on_found = self.note_detections
        if self._configure is not None:
            self._configure(tuple(self._keywords), tuple(self._enabled))

    def note_detections(self, frame, detections) -> None:
        """Log each category as it appears: never the text, only that one was caught."""
        found = {d.category for d in detections if d.category in ALL_CATEGORIES}
        for category in found - self._caught:
            self.events.record(frame.source, category)
        self._caught = found

    def get_state(self) -> PanelState:
        with self._lock:
            enabled, keywords = list(self._enabled), list(self._keywords)
        pipeline = self._pipeline
        return PanelState(
            panic=self._gate.panicked,
            blackout_confirmed=self._gate.black_confirmed,
            mode="desktop",
            irl_available=False,
            view=self._views.view if self._views is not None else None,
            views=self.views,
            enabled=enabled,
            keywords=keywords,
            held_frames=pipeline.stats.rendered_held,
            detector_health=self._detector_health(),
            output_connected=pipeline.sink_connected,
            output_state=self._output_state(),
            simulated=False,
            message=self._message(),
        )

    def set_config(self, config: PanelConfig) -> PanelState:
        with self._change_lock:
            if self._configure is None:
                raise HTTPException(status_code=409, detail="This run has no configurable detector")
            with self._lock:
                stricter = self._is_stricter(config)
            staged = self._store.stage(config.keywords)
            words, enabled = tuple(config.keywords), tuple(config.enabled)
            change = self._pipeline.change_policy(lambda: self._configure(words, enabled), stricter)
            if change.wait(self._apply_wait) and change.error is not None:
                self._store.discard(staged)
                raise HTTPException(
                    status_code=503,
                    detail="Settings were not applied. The previous detector settings stay on, "
                           "and any cover already in place stays until a change succeeds.",
                )
            with self._lock:
                self._enabled = list(config.enabled)
                self._keywords = list(config.keywords)
            self._store.commit(staged)
            return self.get_state()

    def get_events(self, after: int = 0) -> CatchLog:
        return self.events.since(after)

    def panic(self) -> PanelState:
        self._gate.panic()
        self._gate.wait_for_black(PANIC_WAIT_SECONDS)
        return self.get_state()

    def resume(self) -> PanelState:
        self._gate.resume()
        return self.get_state()

    def set_mode(self, mode: ModeConfig) -> PanelState:
        if mode.mode != "desktop":
            raise HTTPException(status_code=409, detail="IRL mode not available")
        return self.get_state()

    @property
    def views(self) -> list[str]:
        return list(self._views.views) if self._views is not None else []

    def set_view(self, view: ViewConfig) -> PanelState:
        """Switch what the output shows. Takes effect on the next captured frame."""
        if self._views is None:
            raise HTTPException(status_code=409, detail="This run has no webcam to switch to")
        with self._view_lock:
            old = self._views.view
            if view.view != old:
                cut = self._gate.cut if drops_a_source(old, view.view) else None
                self._views.set_view(view.view, cut)
        return self.get_state()

    def next_view(self) -> PanelState:
        """The switch view key: on to the next view, and round again."""
        views = self.views
        if not views:
            return self.get_state()
        with self._view_lock:
            following = views[(views.index(self._views.view) + 1) % len(views)]
        return self.set_view(ViewConfig(view=following))

    def _is_stricter(self, config: PanelConfig) -> bool:
        """Whether this hides anything the current settings don't."""
        known = {word.casefold() for word in self._keywords}
        added_word = any(word.casefold() not in known for word in config.keywords)
        return added_word or not set(config.enabled) <= set(self._enabled)

    def _detector_health(self) -> dict[str, DetectorHealth]:
        pipeline = self._pipeline
        if not pipeline.running or self._configure is None:
            text = DetectorHealth.UNAVAILABLE  # with no real detector, nothing is protected
        elif pipeline.policy_error is not None or self._scanner_is_struggling():
            text = DetectorHealth.DEGRADED
        else:
            text = DetectorHealth.READY
        health = {category: text for category in TEXT_CATEGORIES}
        # The visual layer doesn't exist yet, so never report it as protecting anything.
        health.update({category: DetectorHealth.UNAVAILABLE for category in VISUAL_CATEGORIES})
        return health

    def _scanner_is_struggling(self) -> bool:
        pipeline = self._pipeline
        if pipeline.last_output in ("held", "blackout"):
            return True
        last = pipeline.last_scan_at
        return last is not None and monotonic() - last > SCAN_LATE_SECONDS

    def _output_state(self) -> str:
        pipeline = self._pipeline
        if pipeline.output_error is not None:
            return "error"
        if self._gate.black_confirmed:
            return "black"
        if self._gate.panicked:
            return "covered"
        if not pipeline.running:
            return "black" if pipeline.stats.final_black_sent else "stopped"
        if pipeline.last_output in ("scanned", "cached"):
            return "protected" if self._configure is not None else "unverified"
        return "covered" if pipeline.last_output is not None else "stopped"

    def _message(self) -> str:
        pipeline = self._pipeline
        if pipeline.output_error is not None:
            return "The output failed, so nothing is reaching the virtual camera."
        if pipeline.policy_error is not None:
            return "The last settings change failed. Cover stays on until a change succeeds."
        if self._gate.panicked:
            if self._gate.black_confirmed:
                return "Blackout is on the output."
            return "Blackout requested. A black frame has not been confirmed yet."
        if not pipeline.running:
            return "The pipeline is not running."
        if self._configure is None:
            return "Test run with a stand-in detector. Real screen text is not being protected."
        if pipeline.last_output in ("scanned", "cached"):
            shown = self._views.view if self._views is not None else "screen"
            protected = {"screen": "Screen", "both": "Screen and webcam", "webcam": "Webcam"}
            return f"{protected[shown]} text protection is running. IRL protection is planned."
        return "The output is covered until a frame has been checked."
