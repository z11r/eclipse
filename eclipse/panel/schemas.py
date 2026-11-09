"""HTTP schemas for the local Eclipse panel.

These mirror Max's proposed handoff contract. `simulated` says whether a real
pipeline is behind them. Shared types.py additions await team approval.
"""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

TEXT_CATEGORIES = ("two_fa", "email", "phone", "api_key", "card", "address", "keyword")
VISUAL_CATEGORIES = (
    "street_sign", "house_number", "licence_plate", "mail_label",
    "window_view", "map", "storefront",
)
ALL_CATEGORIES = TEXT_CATEGORIES + VISUAL_CATEGORIES
Category = Literal[
    "two_fa", "email", "phone", "api_key", "card", "address", "keyword",
    "street_sign", "house_number", "licence_plate", "mail_label",
    "window_view", "map", "storefront",
]
Source = Literal["screen", "webcam", "irl"]
Mode = Literal["desktop", "irl"]
View = Literal["screen", "both", "webcam"]  # what goes out: see eclipse/capture/dual.py
CoverStyle = Literal["black", "blur", "blur_text"]  # how covers look: see eclipse/compositor/compositor.py


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


class ViewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    view: View


class CoverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    style: CoverStyle


class DetectorHealth(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class PanelState(BaseModel):
    panic: bool = False  # a blackout has been requested
    blackout_confirmed: bool = False  # a black frame has gone out since it was requested
    mode: Mode = "desktop"
    irl_available: bool = False
    # What the output shows, and what it can be switched to. None and [] when the run
    # has a single source, as without --with-webcam.
    view: View | None = None
    views: list[View] = Field(default_factory=list)
    cover_style: CoverStyle = "black"
    enabled: list[Category] = Field(default_factory=lambda: list(ALL_CATEGORIES))
    keywords: list[str] = Field(default_factory=list)
    config_status: Literal["applied", "pending", "failed"] = "applied"
    config_message: str = "Settings applied."
    held_frames: int = 0
    detector_health: dict[str, DetectorHealth] = Field(default_factory=dict)
    output_connected: bool = False
    # unverified: frames are flowing, but no real detector is behind them.
    output_state: Literal[
        "stopped", "protected", "covered", "black", "error", "unverified"
    ] = "stopped"
    simulated: bool = True
    message: str = "Mock only: no capture or OBS output is running."


class CatchEvent(BaseModel):
    incident_id: int = Field(ge=1)
    seq: int = Field(ge=1)
    time: float = Field(ge=0)
    source: Source
    category: Category
    count: int = Field(ge=1)


class CatchLog(BaseModel):
    session_id: str
    events: list[CatchEvent]
    last: int = Field(ge=0)


class HealthResponse(BaseModel):
    service: str = "eclipse-panel"
    ok: bool = True
    pipeline_mode: str = "mock"
