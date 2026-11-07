"""HTTP schemas for the local Eclipse panel.

These mirror Max's proposed handoff contract, without claiming the mock
controls a real OBS pipeline. Shared types.py additions await team approval.
"""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

TEXT_CATEGORIES = ("two_fa", "email", "phone", "api_key", "card", "keyword")
VISUAL_CATEGORIES = (
    "street_sign", "house_number", "licence_plate", "mail_label",
    "window_view", "map", "storefront",
)
ALL_CATEGORIES = TEXT_CATEGORIES + VISUAL_CATEGORIES
Category = Literal[
    "two_fa", "email", "phone", "api_key", "card", "keyword",
    "street_sign", "house_number", "licence_plate", "mail_label",
    "window_view", "map", "storefront",
]
Source = Literal["screen", "webcam", "irl"]
Mode = Literal["desktop", "irl"]


class PanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: list[Category]
    keywords: list[str] = Field(max_length=100)

    @field_validator("enabled")
    @classmethod
    def unique_enabled(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate enabled categories")
        return value

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, value: list[str]) -> list[str]:
        if any(not k.strip() or len(k) > 120 or any(ord(c) < 32 for c in k) for k in value):
            raise ValueError("Keywords must be 1–120 printable characters")
        if len({k.casefold() for k in value}) != len(value):
            raise ValueError("Duplicate keywords")
        return value


class ModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Mode


class DetectorHealth(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class PanelState(BaseModel):
    panic: bool = False
    mode: Mode = "desktop"
    irl_available: bool = False
    enabled: list[Category] = Field(default_factory=lambda: list(ALL_CATEGORIES))
    keywords: list[str] = Field(default_factory=list)
    held_frames: int = 0
    detector_health: dict[str, DetectorHealth] = Field(default_factory=dict)
    output_connected: bool = False
    output_state: Literal["stopped", "protected", "covered", "black", "error"] = "stopped"
    simulated: bool = True
    message: str = "Mock only: no capture or OBS output is running."


class CatchEvent(BaseModel):
    seq: int = Field(ge=1)
    time: float = Field(ge=0)
    source: Source
    category: Category
    count: int = Field(ge=1)


class CatchLog(BaseModel):
    events: list[CatchEvent]
    last: int = Field(ge=0)


class HealthResponse(BaseModel):
    service: str = "eclipse-panel"
    ok: bool = True
    pipeline_mode: str = "mock"
