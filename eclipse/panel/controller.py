"""Pipeline contract and a mock implementation for panel development.

This mock never captures screens/cameras or sends frames to OBS.
It MUST NOT be used as proof that output is protected.
"""
from threading import Lock
from typing import Protocol

from .schemas import DetectorInfo, DetectorState, PipelineState, PipelineStatus


class PipelineController(Protocol):
    def get_status(self) -> PipelineStatus: ...


class MockPipelineController:
    def __init__(self) -> None:
        self._lock = Lock()
        self._status = PipelineStatus(
            state=PipelineState.STOPPED,
            output_connected=False,
            mode="mock",
            message="Mock only: no capture or OBS output is running.",
            detectors={
                name: DetectorInfo(enabled=False, state=DetectorState.UNAVAILABLE)
                for name in ("email", "phone", "two_factor", "api_key", "payment_card", "visual")
            },
        )

    def get_status(self) -> PipelineStatus:
        with self._lock:
            return self._status.model_copy(deep=True)
