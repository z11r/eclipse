from enum import Enum
from pydantic import BaseModel, Field


class PipelineState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    PROTECTED = "protected"
    DEGRADED = "degraded"
    BLACKOUT = "blackout"
    ERROR = "error"


class DetectorState(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class DetectorInfo(BaseModel):
    enabled: bool
    state: DetectorState


class PipelineStatus(BaseModel):
    state: PipelineState
    output_connected: bool = False
    mode: str = "mock"
    detectors: dict[str, DetectorInfo] = Field(default_factory=dict)
    message: str = ""


class HealthResponse(BaseModel):
    service: str = "eclipse-panel"
    ok: bool = True
    pipeline_mode: str = "mock"
