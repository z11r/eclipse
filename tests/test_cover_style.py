"""Blur covers: a different look, never a different set of covered pixels."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eclipse.compositor.compositor import BLACK, BLUR, Compositor
from eclipse.panel.app import create_app
from eclipse.panel.controller import LivePipelineController, MockPipelineController
from eclipse.panel.schemas import CoverConfig
from eclipse.types import Detection, Frame


def gradient_frame(frame_id=0):
    rows, columns = np.mgrid[0:100, 0:200]
    image = np.stack([rows * 2, columns, 255 - columns], axis=-1).astype(np.uint8)
    return Frame(image, 0.0, "screen", frame_id)


def box(x=50, y=30, w=40, h=20, frame_id=0):
    return Detection(x, y, w, h, "api_key", 0.9, 500, frame_id)


def test_blur_hides_the_same_area_and_leaves_the_rest_alone():
    frame = gradient_frame()
    black = Compositor(padding=0).process(frame, lambda f: [box()])
    blurred = Compositor(padding=0, style=BLUR).process(frame, lambda f: [box()])
    outside = np.ones(frame.image.shape[:2], bool)
    outside[30:50, 50:90] = False
    assert (blurred[outside] == frame.image[outside]).all()
    assert (black[outside] == blurred[outside]).all()
    assert not (blurred[30:50, 50:90] == 0).all()  # it really is a different look


def test_blur_carries_nothing_from_the_pixels_under_it():
    plain, secret = gradient_frame(), gradient_frame()
    rng = np.random.default_rng(1)
    secret.image[30:50, 50:90] = rng.integers(0, 255, (20, 40, 3), dtype=np.uint8)
    one = Compositor(padding=0, style=BLUR).process(plain, lambda f: [box()])
    other = Compositor(padding=0, style=BLUR).process(secret, lambda f: [box()])
    assert (one == other).all()


def test_a_ring_that_crosses_another_cover_never_sees_its_secret():
    plain, secret = gradient_frame(), gradient_frame()
    secret.image[30:50, 92:120] = 255  # under the second cover, inside the first's ring
    covers = lambda f: [box(), box(x=92, w=28)]
    one = Compositor(padding=0, style=BLUR).process(plain, covers)
    other = Compositor(padding=0, style=BLUR).process(secret, covers)
    assert (one == other).all()


def test_a_cover_over_the_whole_frame_is_black():
    frame = gradient_frame()
    output = Compositor(padding=0, style=BLUR).process(frame, lambda f: [box(0, 0, 200, 100)])
    assert (output == 0).all()


def test_style_switches_live_and_rejects_unknown_styles():
    compositor = Compositor(padding=0)
    compositor.style = BLUR
    assert compositor.style == BLUR
    with pytest.raises(ValueError):
        compositor.style = "pink"
    assert compositor.style == BLUR


def test_mock_panel_round_trips_the_style(tmp_path):
    controller = MockPipelineController(config_path=tmp_path / "config.json")
    with TestClient(create_app(controller), base_url="http://127.0.0.1:8000") as client:
        client.headers["Authorization"] = "Bearer " + client.app.state.panel_token
        assert client.get("/api/state").json()["cover_style"] == "black"
        assert client.put("/api/cover", json={"style": "blur"}).json()["cover_style"] == "blur"
        assert client.put("/api/cover", json={"style": "pink"}).status_code == 422
        del client.headers["Authorization"]
        assert client.put("/api/cover", json={"style": "black"}).status_code == 401


def _live(compositor):
    gate = SimpleNamespace(panicked=False, black_confirmed=False)
    stats = SimpleNamespace(rendered_held=0, final_black_sent=False)
    pipeline = SimpleNamespace(
        gate=gate, stats=stats, running=True, policy_error=None, output_error=None,
        last_output="scanned", last_scan_at=None, sink_connected=True,
    )
    return LivePipelineController(pipeline, None, compositor=compositor)


def test_live_controller_changes_the_compositor_it_was_given(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    compositor = Compositor()
    controller = _live(compositor)
    assert controller.set_cover(CoverConfig(style=BLUR)).cover_style == BLUR
    assert compositor.style == BLUR
    controller.set_cover(CoverConfig(style=BLACK))
    assert compositor.style == BLACK
    with pytest.raises(HTTPException):
        _live(None).set_cover(CoverConfig(style=BLUR))


def test_blur_looks_like_blurred_text_even_on_a_plain_background():
    image = np.full((100, 200, 3), 30, np.uint8)  # one flat dark colour around the cover
    frame = Frame(image, 0.0, "screen", 0)
    output = Compositor(padding=0, style=BLUR).process(frame, lambda f: [box()])
    region = output[30:50, 50:90].astype(int)
    assert region.max() - region.min() > 40  # visible smudges, not just the background
    assert (np.abs(region - 30) > 20).any()


def test_blurred_words_stay_put_while_the_cover_moves():
    image = np.full((100, 200, 3), 30, np.uint8)
    frame = Frame(image, 0.0, "screen", 0)
    here = Compositor(padding=0, style=BLUR).process(frame, lambda f: [box(x=50)])
    there = Compositor(padding=0, style=BLUR).process(frame, lambda f: [box(x=110)])
    assert (here[30:50, 50:90] == there[30:50, 110:150]).all()
