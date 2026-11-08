"""Screen and webcam together: each scanned on its own, the camera shown in a corner."""

import threading
import time

import numpy as np
import pytest

from eclipse import main as app
from eclipse.capture.dual import DualCapture, DualFrame, pip_layout
from eclipse.compositor.compositor import FILL
from eclipse.compositor.pip import PictureInPictureCompositor, PictureInPictureDetector
from eclipse.types import Detection, Frame


class Feed:
    """A source handing out frames from a list, one per read, then waiting."""

    def __init__(self, images, source="screen", paced=False, finish=False, fail_at=None):
        self.images = list(images)
        self.source = source
        self.paced = paced
        self.finish = finish
        self.fail_at = fail_at
        self.finished = False
        self.read_thread = None
        self.next_id = 0
        self.stats = {"grab_seconds": 0.0}

    def __enter__(self):
        self.read_thread = threading.current_thread()
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        assert threading.current_thread() is self.read_thread  # opened and read on one thread
        if self.fail_at is not None and self.next_id == self.fail_at:
            raise RuntimeError("camera unplugged")
        if self.next_id >= len(self.images):
            if self.finish:
                self.finished = True
            time.sleep(0.005)
            return None
        frame = Frame(self.images[self.next_id], time.monotonic(), self.source, self.next_id)
        self.next_id += 1
        return frame


def screen_image(value=200, height=240, width=320):
    return np.full((height, width, 3), value, dtype=np.uint8)


def camera_image(value=90, height=120, width=160):
    return np.full((height, width, 3), value, dtype=np.uint8)


def read_until(capture, count, seconds=2.0):
    frames = []
    give_up = time.monotonic() + seconds
    while len(frames) < count and time.monotonic() < give_up:
        frame = capture.read()
        if frame is not None:
            frames.append(frame)
    return frames


def test_layout_puts_the_camera_in_the_bottom_right_at_its_own_shape():
    at, shown, scanned = pip_layout((1080, 1920, 3), (720, 1280, 3), 0.25)
    assert shown == (480, 270)
    assert scanned == (960, 540)  # twice the shown size, never more than the camera has
    left, top = at
    assert left + shown[0] < 1920 and top + shown[1] < 1080
    assert left > 1920 / 2 and top > 1080 / 2


def test_layout_never_scans_above_the_cameras_own_size():
    _, shown, scanned = pip_layout((1080, 1920, 3), (480, 640, 3), 0.5)
    assert shown == (960, 720)
    assert scanned == (640, 480)


def test_each_input_is_read_on_its_own_thread_and_both_reach_every_frame():
    screen = Feed([screen_image(v) for v in (10, 20, 30)])
    camera = Feed([camera_image()], source="webcam")
    with DualCapture(screen, camera, fps=200) as capture:
        frames = read_until(capture, 3)
    assert screen.read_thread is not camera.read_thread
    # Only the newest of each input counts: anything older is already out of date.
    assert [frame.frame_id for frame in frames] == list(range(len(frames)))
    assert all(isinstance(frame, DualFrame) for frame in frames)
    assert all(frame.webcam is not None and frame.pip is not None for frame in frames)
    # The screen pixels are passed through untouched and the camera is not pasted yet.
    values = [int(frame.image[0, 0, 0]) for frame in frames]
    assert values == sorted(set(values)) and values[-1] == 30
    # One camera picture, resized once and shared.
    assert len({id(frame.pip) for frame in frames}) == 1
    assert {frame.webcam_id for frame in frames} == {0}


def test_a_still_screen_still_sends_each_new_camera_picture():
    screen = Feed([screen_image()], paced=True)
    camera = Feed([camera_image(v) for v in (40, 50, 60)], source="webcam")
    with DualCapture(screen, camera, fps=200) as capture:
        frames = read_until(capture, 3, seconds=0.5)
    ids = [frame.webcam_id for frame in frames]
    assert len(frames) >= 1 and ids == sorted(set(ids)) and ids[-1] == 2
    assert {id(frame.image) for frame in frames} == {id(screen.images[0])}


def test_a_camera_that_fails_stops_the_capture():
    screen = Feed([screen_image()] * 1000)
    camera = Feed([camera_image()] * 1000, source="webcam", fail_at=2)
    with DualCapture(screen, camera, fps=200) as capture:
        with pytest.raises(RuntimeError):
            for _ in range(200):
                capture.read()


def test_a_clip_standing_in_for_the_screen_finishes_the_capture():
    screen = Feed([screen_image()], finish=True)
    camera = Feed([camera_image()], source="webcam")
    with DualCapture(screen, camera, fps=200) as capture:
        read_until(capture, 1)
        give_up = time.monotonic() + 2
        while not capture.finished and time.monotonic() < give_up:
            capture.read()
    assert capture.finished


class Recorder:
    """A detector reporting fixed boxes, in whatever frame it is handed."""

    def __init__(self, boxes=(), fail=False):
        self.boxes = list(boxes)
        self.fail = fail
        self.calls = []
        self.threads = set()
        self.reused_read = False
        self.configured = []

    def __call__(self, frame):
        self.calls.append(frame)
        self.threads.add(threading.current_thread().name)
        if self.fail:
            raise RuntimeError("scan failed")
        return [
            Detection(x, y, w, h, "api_key", 1.0, 500, frame.frame_id)
            for x, y, w, h in self.boxes
        ]

    def configure(self, keywords=(), enabled=None):
        self.configured.append((keywords, enabled))


def dual_frame(frame_id, webcam_id, screen=None, webcam_ts=0.0, camera_value=90):
    # A 320x240 screen with a 160x120 camera shown at 80x60 in the corner at (230, 170).
    return DualFrame(
        screen if screen is not None else screen_image(),
        0.0,
        "screen",
        frame_id,
        webcam=camera_image(camera_value),
        pip=camera_image(camera_value, height=60, width=80),
        pip_at=(230, 170),
        webcam_id=webcam_id,
        webcam_ts=webcam_ts,
    )


def test_the_camera_is_scanned_once_per_picture_and_off_the_scan_thread():
    screen, webcam = Recorder(), Recorder([(20, 40, 40, 20)])
    detect = PictureInPictureDetector(screen, webcam)
    first = detect(dual_frame(0, webcam_id=7))
    second = detect(dual_frame(1, webcam_id=7))
    assert len(screen.calls) == 2
    assert len(webcam.calls) == 1  # the repeat reused the camera's covers
    assert webcam.calls[0].source == "webcam"
    assert webcam.threads == {"eclipse-webcam-scan"}
    # Mapped from the scanned camera's pixels into the corner, at half the size.
    assert [(b.x, b.y, b.w, b.h) for b in first] == [(240, 190, 20, 10)]
    assert [(b.x, b.y, b.w, b.h, b.frame_id) for b in second] == [(240, 190, 20, 10, 1)]
    assert detect.stats["webcam_repeats"] == 1
    detect(dual_frame(2, webcam_id=8))  # a new picture: scanned beside, not waited for
    give_up = time.monotonic() + 5
    while len(webcam.calls) < 2 and time.monotonic() < give_up:
        time.sleep(0.005)
    assert len(webcam.calls) == 2


def test_covers_hidden_behind_the_camera_are_dropped_and_others_kept():
    screen = Recorder([(240, 180, 20, 10), (10, 10, 50, 10), (220, 160, 30, 30)])
    detect = PictureInPictureDetector(screen, Recorder())
    found = detect(dual_frame(0, webcam_id=0))
    # The first sits wholly under the camera; the last only partly, so it stays.
    assert [(b.x, b.y) for b in found] == [(10, 10), (220, 160)]


def test_a_failed_camera_scan_fails_the_frame_and_is_retried():
    webcam = Recorder(fail=True)
    detect = PictureInPictureDetector(Recorder(), webcam)
    with pytest.raises(RuntimeError):
        detect(dual_frame(0, webcam_id=3))
    webcam.fail = False
    detect(dual_frame(1, webcam_id=3))
    assert len(webcam.calls) == 2


def test_new_settings_read_the_camera_again():
    screen, webcam = Recorder(), Recorder()
    detect = PictureInPictureDetector(screen, webcam)
    detect(dual_frame(0, webcam_id=1))
    detect.configure(["secret"], ["api_key"])
    detect(dual_frame(1, webcam_id=1))
    assert len(webcam.calls) == 2
    assert screen.configured == webcam.configured == [(("secret",), ("api_key",))]


def test_a_plain_frame_goes_to_the_screen_alone():
    screen, webcam = Recorder([(1, 2, 3, 4)]), Recorder()
    detect = PictureInPictureDetector(screen, webcam)
    found = detect(Frame(screen_image(), 0.0, "screen", 5))
    assert [(b.x, b.y) for b in found] == [(1, 2)] and not webcam.calls


def test_the_camera_is_pasted_and_its_covers_land_on_it():
    frame = dual_frame(0, webcam_id=0)
    detect = PictureInPictureDetector(Recorder(), Recorder([(20, 40, 40, 20)]))
    rendered = PictureInPictureCompositor(detect, padding=0).render(frame, detect)
    image = rendered.image
    assert rendered.state == "scanned"
    assert (image[175, 235] == 90).all()  # camera pixels in the corner
    assert (image[10, 10] == 200).all()  # the screen elsewhere
    assert (image[190:200, 240:260] == FILL).all()  # the camera's cover
    assert (frame.image == 200).all()  # the screen frame itself is left alone


def test_main_runs_screen_and_webcam_together(monkeypatch):
    sent = []

    class Sink:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def send(self, image):
            sent.append(image)

    screens = [screen_image(v) for v in range(10, 60)]
    monkeypatch.setattr(app, "screen_capture_allowed", lambda: True)
    monkeypatch.setattr(app, "screen_capture_kit_available", lambda: False)
    monkeypatch.setattr(app, "ScreenCapture", lambda max_width, retina, first_id=0: Feed(screens))
    monkeypatch.setattr(app, "WebcamCapture", lambda index: Feed([camera_image()] * 50, "webcam"))
    monkeypatch.setattr(app, "TextDetector", lambda **kwargs: Recorder())
    monkeypatch.setattr(app, "VirtualCam", Sink)
    assert app.main(["--with-webcam", "0", "--max-frames", "5", "--delay", "0.05"]) == 0
    shown = [image for image in sent if image.any()]
    assert shown and all(image.shape == (240, 320, 3) for image in shown)
    left, top = pip_layout((240, 320), (120, 160), app.PIP_SIZE)[0]
    assert (shown[0][top + 1, left + 1] == 90).all()  # the camera made it into the corner
