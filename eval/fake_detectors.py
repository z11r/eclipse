"""Stand-in detectors that behave in known ways, to prove the eval reports the truth.

If the eval can't report a leak from a detector built to leak, a score of 0 means
nothing.
"""

from __future__ import annotations

from eclipse.types import Detection, Frame
from eval.truth import box_at, is_on_screen

TTL_MS = 500


def _detection(secret: dict, frame: Frame) -> Detection:
    x, y, w, h = box_at(secret, frame.frame_id)
    return Detection(x, y, w, h, secret["category"], 1.0, TTL_MS, frame.frame_id)


def perfect(labels: dict):
    """Finds everything on every frame. Should score 0 leaks."""

    def detect(frame: Frame) -> list[Detection]:
        return [
            _detection(secret, frame)
            for secret in labels["secrets"]
            if is_on_screen(secret, frame.frame_id)
        ]

    return detect


def misses_one(labels: dict):
    """Never finds the email. Should leak every frame the email is on screen."""
    others = {**labels, "secrets": [s for s in labels["secrets"] if s["id"] != "email"]}
    return perfect(others)


def first_sighting_only(labels: dict):
    """Reports a secret once, then stops, like OCR that skips unchanged areas.

    The compositor keeps nothing between frames, so this should leak still secrets
    from the next frame on.
    """
    seen: set[str] = set()

    def detect(frame: Frame) -> list[Detection]:
        found = []
        for secret in labels["secrets"]:
            if is_on_screen(secret, frame.frame_id) and secret["id"] not in seen:
                seen.add(secret["id"])
                found.append(_detection(secret, frame))
        return found

    return detect


def half_box(labels: dict):
    """Covers only the left half of each secret. Should leak every secret."""
    whole = perfect(labels)

    def detect(frame: Frame) -> list[Detection]:
        return [
            Detection(d.x, d.y, d.w // 2, d.h, d.category, d.confidence, d.ttl_ms, d.frame_id)
            for d in whole(frame)
        ]

    return detect


def crashes(labels: dict):
    """Fails on every frame. Should send black and leak nothing."""

    def detect(frame: Frame) -> list[Detection]:
        raise RuntimeError("detector crashed")

    return detect


def over_blurs(labels: dict):
    """Finds everything but also covers the top of the screen. Should raise false blur."""
    whole = perfect(labels)

    def detect(frame: Frame) -> list[Detection]:
        extra = Detection(0, 0, 900, 40, "keyword", 0.5, TTL_MS, frame.frame_id)
        return [*whole(frame), extra]

    return detect


FAKE_DETECTORS = {
    "perfect": perfect,
    "misses-one": misses_one,
    "first-sighting-only": first_sighting_only,
    "half-box": half_box,
    "crashes": crashes,
    "over-blurs": over_blurs,
}
