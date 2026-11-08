"""Switching live between the screen, both, and the webcam, and a slow camera never
holding the screen back."""

import threading
import time

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eclipse.capture.dual import DualCapture, DualFrame, drops_a_source
from eclipse.compositor.delay import DelayedPipeline
from eclipse.compositor.gate import OutputGate
from eclipse.compositor.pip import PictureInPictureCompositor, PictureInPictureDetector
from eclipse.panel.app import create_app
from eclipse.panel.controller import LivePipelineController, MockPipelineController
from eclipse.panel.schemas import ViewConfig
from eclipse.types import Detection, Frame

from tests.test_pip import Recorder, camera_image, dual_frame, screen_image

SCREEN, CAMERA = 200, 90


class Endless:
    """A live source: the same picture, new each read, at its own pace."""

    def __init__(self, image, source, every=0.004):
        self.image = image
        self.source = source
        self.every = every
        self.next_id = 0
        self.paced = True
        self.stats = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        time.sleep(self.every)
        frame = Frame(self.image, time.monotonic(), self.source, self.next_id)
        self.next_id += 1
        return frame


class Still(Endless):
    """A screen that sends one picture and then nothing, as ScreenCaptureKit does."""

    def read(self):
        if self.next_id:
            time.sleep(self.every)
            return None
        return super().read()


class Slow(Recorder):
    """A camera detector that waits for permission before each scan after the first."""

    def __init__(self, boxes=()):
        super().__init__(boxes)
        self.go = threading.Event()

    def __call__(self, frame):
        if self.calls:
            assert self.go.wait(5)
            self.go.clear()
        return super().__call__(frame)


def wait_until(check, seconds=5.0):
    give_up = time.monotonic() + seconds
    while time.monotonic() < give_up:
        if check():
            return True
        time.sleep(0.005)
    return False


# A slow camera never holds the screen back.


def test_a_slow_camera_scan_leaves_the_screen_at_full_rate_with_the_last_scanned_picture():
    webcam = Slow([(20, 40, 40, 20)])
    detect = PictureInPictureDetector(Recorder(), webcam)
    compositor = PictureInPictureCompositor(detect, padding=0)
    first = compositor.render(dual_frame(0, webcam_id=1, camera_value=90), detect)
    assert first.state == "scanned"

    # Camera picture 2 is being scanned; the frame goes out without waiting for it, with
    # picture 1 in the corner and picture 1's covers.
    started = time.monotonic()
    second = compositor.render(dual_frame(1, webcam_id=2, camera_value=60), detect)
    assert time.monotonic() - started < 1
    assert (second.image[175, 235] == 90).all()
    assert (second.image[190:200, 240:260] == 0).all()
    assert detect.stats["webcam_behind"] == 1

    webcam.go.set()
    assert wait_until(lambda: detect._job is not None and detect._job.future.done())
    third = compositor.render(dual_frame(2, webcam_id=2, camera_value=60), detect)
    assert (third.image[175, 235] == 60).all()  # caught up


def test_the_screen_waits_once_the_corner_falls_too_far_behind():
    webcam = Slow()
    detect = PictureInPictureDetector(Recorder(), webcam)
    detect(dual_frame(0, webcam_id=1, webcam_ts=10.0))
    waits = detect.stats["webcam_waits"]  # the first picture is always waited for
    threading.Timer(0.2, webcam.go.set).start()
    started = time.monotonic()
    detect(dual_frame(1, webcam_id=2, webcam_ts=11.0))  # a second newer: too stale to show
    assert time.monotonic() - started >= 0.15
    assert detect.stats["webcam_waits"] == waits + 1
    assert detect.shown_camera(dual_frame(1, webcam_id=2)) is not None


def test_a_scan_still_running_when_results_are_forgotten_is_never_shown():
    webcam = Slow([(20, 40, 40, 20)])
    detect = PictureInPictureDetector(Recorder(), webcam)
    detect(dual_frame(0, webcam_id=1))
    detect(dual_frame(1, webcam_id=2))  # picture 2's scan starts and is still running
    detect.forget()  # a cut, or stricter settings
    webcam.boxes = []  # what picture 2's scan would find, if it counted
    threading.Timer(0.1, webcam.go.set).start()
    threading.Timer(0.4, webcam.go.set).start()
    found = detect(dual_frame(2, webcam_id=3))
    # Nothing found before the forget was used: picture 3 was scanned afresh.
    assert len(webcam.calls) == 3 and webcam.calls[-1].frame_id == 3
    assert found == []


def test_the_compositor_refuses_a_frame_without_its_scanned_camera_picture():
    detect = PictureInPictureDetector(Recorder(), Recorder())
    compositor = PictureInPictureCompositor(detect, padding=0)
    rendered = compositor.render(dual_frame(0, webcam_id=1), lambda frame: [])
    assert rendered.state == "blackout"


def test_each_view_goes_to_its_own_detector():
    screen, corner, alone = Recorder(), Recorder(), Recorder()
    detect = PictureInPictureDetector(screen, corner, alone)
    detect(Frame(screen_image(), 0.0, "screen", 0))
    detect(Frame(camera_image(), 0.0, "webcam", 1))
    detect(dual_frame(2, webcam_id=0))
    assert [f.frame_id for f in screen.calls] == [0, 2]
    assert [f.frame_id for f in alone.calls] == [1]
    assert [f.frame_id for f in corner.calls] == [0]  # the camera's own picture id


# The capture switches views.


def read_until(capture, check, seconds=2.0):
    give_up = time.monotonic() + seconds
    while time.monotonic() < give_up:
        frame = capture.read()
        if frame is not None and check(frame):
            return frame
    return None


def test_each_view_makes_its_own_kind_of_frame():
    capture = DualCapture(
        Endless(screen_image(SCREEN), "screen"), Endless(camera_image(CAMERA), "webcam"),
        fps=200,
    )
    with capture:
        assert isinstance(capture.read() or read_until(capture, bool), DualFrame)
        capture.set_view("screen")
        frame = read_until(capture, lambda f: True)
        assert type(frame) is Frame and frame.source == "screen"
        assert (frame.image == SCREEN).all()
        capture.set_view("webcam")
        frame = read_until(capture, lambda f: True)
        assert type(frame) is Frame and frame.source == "webcam"
        assert frame.image.shape == camera_image().shape  # full size, not the corner's
    assert capture.stats["view_switches"] == 2


def test_a_switch_to_a_still_screen_still_sends_a_frame_after_the_cut():
    gate = OutputGate()
    capture = DualCapture(
        Still(screen_image(SCREEN), "screen"), Endless(camera_image(CAMERA), "webcam"),
        fps=200,
    )
    with capture:
        assert read_until(capture, lambda f: isinstance(f, DualFrame))
        capture.set_view("screen", gate.cut)
        frame = read_until(capture, lambda f: True)
    assert frame is not None and frame.source == "screen"
    assert not gate.captured_too_early(frame.ts)


def test_only_a_switch_that_drops_a_source_cuts():
    assert drops_a_source("both", "screen") and drops_a_source("both", "webcam")
    assert drops_a_source("screen", "webcam") and drops_a_source("webcam", "screen")
    assert not drops_a_source("screen", "both") and not drops_a_source("webcam", "both")


def test_the_cut_stops_frames_from_the_same_clock_tick():
    gate = OutputGate()
    before = time.monotonic()
    floor = gate.cut()
    assert gate.captured_too_early(before)
    assert gate.captured_too_early(time.monotonic()) is (time.monotonic() < floor)
    assert not gate.panicked


# Switching a running pipeline.


class Sink:
    """Keeps what each picture showed: (corner value, top-left value), black as (0, 0)."""

    def __init__(self):
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        # The corner the camera goes in on a 320x240 screen, and a spot it never covers.
        self.sent.append((int(image[-40, -60, 0]), int(image[10, 10, 0])))


class Running:
    def __init__(self, tmp_path, view="both"):
        self.capture = DualCapture(
            Endless(screen_image(SCREEN), "screen"),
            Endless(camera_image(CAMERA), "webcam"),
            fps=200,
            view=view,
        )
        detect = PictureInPictureDetector(Recorder(), Recorder(), Recorder())
        self.sink = Sink()
        self.pipeline = DelayedPipeline(
            self.capture, self.sink, detect, PictureInPictureCompositor(detect, padding=0),
            delay=0.05, fps=200,
        )
        self.controller = LivePipelineController(
            self.pipeline, detect.configure, tmp_path / "config.json", views=self.capture
        )
        self.thread = threading.Thread(target=self.pipeline.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.pipeline._stop.set()
        self.thread.join(timeout=10)


def test_switching_away_from_the_camera_never_shows_it_again(tmp_path):
    with Running(tmp_path) as run:
        assert wait_until(lambda: run.sink.sent[-1:] == [(CAMERA, SCREEN)])
        state = run.controller.set_view(ViewConfig(view="screen"))
        after = len(run.sink.sent)
        assert state.view == "screen" and state.views == ["screen", "both", "webcam"]
        assert wait_until(lambda: run.sink.sent[-1:] == [(SCREEN, SCREEN)])
        later = set(run.sink.sent[after:])
    # Between the switch and the first screen-only frame, only black.
    assert later <= {(0, 0), (SCREEN, SCREEN)}


def test_switching_to_the_camera_alone_shows_it_full_size(tmp_path):
    with Running(tmp_path, view="screen") as run:
        assert wait_until(lambda: run.sink.sent[-1:] == [(SCREEN, SCREEN)])
        run.controller.set_view(ViewConfig(view="webcam"))
        after = len(run.sink.sent)
        assert wait_until(lambda: run.sink.sent[-1:] == [(CAMERA, CAMERA)])
        later = set(run.sink.sent[after:])
    assert later <= {(0, 0), (CAMERA, CAMERA)}  # never the screen again


def test_the_switch_key_goes_round_every_view(tmp_path):
    with Running(tmp_path, view="screen") as run:
        seen = [run.controller.get_state().view]
        for _ in range(3):
            seen.append(run.controller.next_view().view)
        assert wait_until(lambda: len(run.sink.sent) > 3)
    assert seen == ["screen", "both", "webcam", "screen"]


def test_switching_needs_a_webcam_run(tmp_path):
    pipeline = DelayedPipeline(
        Endless(screen_image(), "screen"), Sink(), Recorder(),
        PictureInPictureCompositor(PictureInPictureDetector(Recorder(), Recorder())),
    )
    controller = LivePipelineController(pipeline, None, tmp_path / "config.json")
    with pytest.raises(HTTPException) as refused:
        controller.set_view(ViewConfig(view="webcam"))
    assert refused.value.status_code == 409
    assert controller.get_state().view is None and controller.get_state().views == []


def test_the_panel_api_switches_views(tmp_path):
    controller = MockPipelineController(config_path=tmp_path / "config.json")
    with TestClient(create_app(controller), base_url="http://127.0.0.1:8000") as client:
        assert client.get("/api/state").json()["views"] == ["screen", "both", "webcam"]
        assert client.put("/api/view", json={"view": "webcam"}).json()["view"] == "webcam"
        assert client.put("/api/view", json={"view": "irl"}).status_code == 422
        assert client.get("/api/state").json()["view"] == "webcam"
