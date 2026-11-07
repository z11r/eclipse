"""Panic and settings, checked against the pixels that really reached the output."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eclipse import main as app
from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.compositor.gate import OutputGate
from eclipse.output import virtual_cam
from eclipse.panel.app import create_app, create_simulation_app
from eclipse.panel.controller import LivePipelineController, MockPipelineController
from eclipse.panel.hotkey import PanicHotkey
from eclipse.panel.schemas import ALL_CATEGORIES, PanelConfig
from eclipse.text_detect import TextDetector
from eclipse.types import Detection, Frame

HEIGHT, WIDTH = 60, 80
SECRET = (20, 30, 20, 60)  # top, bottom, left, right of the fake word
KEYWORD = "FAKE_DEMO_STREET"
ALL = list(ALL_CATEGORIES)


def wait_until(condition, seconds=3.0):
    give_up_at = time.monotonic() + seconds
    while time.monotonic() < give_up_at:
        if condition():
            return True
        time.sleep(0.002)
    return condition()


class Source:
    """Frames with their id in the top left corner, and a fake word drawn on them."""

    def __init__(self, interval=0.002):
        self.next_id = 0
        self.interval = interval
        self.read_times = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self):
        time.sleep(self.interval)
        image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
        image[0, 0] = (self.next_id // 256, self.next_id % 256, 200)
        top, bottom, left, right = SECRET
        image[top:bottom, left:right] = 250
        now = time.monotonic()
        self.read_times[self.next_id] = now
        frame = Frame(image, now, "screen", self.next_id)
        self.next_id += 1
        return frame


def stamped_id(image):
    return int(image[0, 0, 0]) * 256 + int(image[0, 0, 1])


def is_black(image):
    return not image.any()


class Sink:
    """Records every picture as it goes out, and when."""

    def __init__(self, interval=0.002):
        self.interval = interval
        self.sent = []
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def send(self, image):
        with self.lock:
            self.sent.append((time.monotonic(), image))
        time.sleep(self.interval)

    def after(self, moment):
        with self.lock:
            return [image for sent_at, image in self.sent if sent_at >= moment]

    def pictures(self):
        return [image for _, image in list(self.sent)]


class KeywordDetector:
    """Hides the fake word only once it's a keyword, as a real detector would."""

    def __init__(self, delay=0.0):
        self.keywords = ()
        self.enabled = ()
        self.delay = delay
        self.fail_next_configure = False
        self.generations = []

    def configure(self, keywords=(), enabled=None):
        if self.fail_next_configure:
            self.fail_next_configure = False
            raise ValueError("this configuration can't be used")
        self.keywords, self.enabled = tuple(keywords), tuple(enabled or ())

    def __call__(self, frame):
        time.sleep(self.delay)
        if KEYWORD not in self.keywords:
            return []
        top, bottom, left, right = SECRET
        return [Detection(left, top, right - left, bottom - top, "keyword", 1.0, 0, frame.frame_id)]


def word_visible(image):
    top, bottom, left, right = SECRET
    return bool((image[top:bottom, left:right] == 250).any())


class Live:
    """A real pipeline and controller, running on their own thread."""

    def __init__(self, tmp_path, detect=None, delay=0.03, fps=200, source=None, sink=None, **kw):
        self.source = source or Source()
        self.sink = sink or Sink()
        self.detect = detect if detect is not None else KeywordDetector()
        self.pipeline = DelayedPipeline(
            self.source, self.sink, self.detect, Compositor(padding=0), delay, fps, **kw
        )
        configure = getattr(self.detect, "configure", None)
        self.controller = LivePipelineController(
            self.pipeline, configure, tmp_path / "config.json"
        )
        self.gate = self.pipeline.gate
        self.result = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self.result["stats"] = self.pipeline.run()
        except BaseException as error:
            self.result["error"] = error

    def __enter__(self):
        self.thread.start()
        assert wait_until(lambda: len(self.sink.sent) > 3), "nothing reached the output"
        return self

    def __exit__(self, *exc):
        self.pipeline._stop.set()
        self.thread.join(timeout=10)

    def config(self, keywords=(), enabled=None):
        return PanelConfig(enabled=list(enabled or ALL), keywords=list(keywords))


def test_panic_blacks_the_next_picture_while_ocr_is_stuck(tmp_path):
    release = threading.Event()

    class Stuck:
        def __init__(self):
            self.calls = 0

        def __call__(self, frame):
            self.calls += 1
            if self.calls > 20:
                release.wait(10)
            return []

    with Live(tmp_path, Stuck()) as live:
        assert wait_until(lambda: live.detect.calls > 20)
        asked = time.monotonic()
        state = live.controller.panic()
        answered = time.monotonic() - asked
        assert answered < 0.3
        assert state.panic and state.blackout_confirmed and state.output_state == "black"
        late = time.monotonic()
        assert wait_until(lambda: len(live.sink.after(late)) > 3)
        assert all(is_black(image) for image in live.sink.after(late))
        release.set()


def test_panic_is_not_reported_as_black_until_a_black_frame_went_out(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    class Blocking(Sink):
        def __init__(self):
            super().__init__()
            self.block = False

        def send(self, image):
            if self.block:
                entered.set()
                finish.wait(5)
            super().send(image)

    sink = Blocking()
    with Live(tmp_path, sink=sink) as live:
        sink.block = True
        assert entered.wait(2)
        state = live.controller.panic()
        assert state.panic is True
        assert state.blackout_confirmed is False
        assert state.output_state == "covered"
        assert "not been confirmed" in state.message
        sink.block = False
        finish.set()
        assert wait_until(lambda: live.controller.get_state().blackout_confirmed)
        assert live.controller.get_state().output_state == "black"


def test_panic_while_a_frame_is_being_prepared_replaces_it_with_black(monkeypatch, tmp_path):
    fitting, go_on = threading.Event(), threading.Event()
    sent = []

    class Camera:
        device = "fake"

        def __init__(self, **kwargs):
            pass

        def send(self, image):
            sent.append(image.copy())

        def close(self):
            pass

    def slow_fit(image, width, height):
        if image.any() and fitting.is_set() is False and len(sent) > 3:
            fitting.set()
            go_on.wait(5)
        return np.full((height, width, 3), image.max(), np.uint8)

    monkeypatch.setattr(virtual_cam.pyvirtualcam, "Camera", Camera)
    monkeypatch.setattr(virtual_cam, "fit", slow_fit)
    cam = virtual_cam.VirtualCam(fps=500)
    source = Source()
    pipeline = DelayedPipeline(source, cam, lambda frame: [], Compositor(), 0.02, 200)
    thread = threading.Thread(target=pipeline.run, daemon=True)
    thread.start()
    assert fitting.wait(5)
    count = len(sent)
    pipeline.gate.panic()
    go_on.set()
    assert wait_until(lambda: len(sent) > count)
    assert not sent[count].any()  # the frame being prepared never left
    assert wait_until(lambda: pipeline.gate.black_confirmed)
    pipeline._stop.set()
    thread.join(timeout=5)


def test_panic_before_any_frame_exists_still_reaches_the_output(tmp_path):
    class Nothing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            time.sleep(0.01)

    sink = Sink()
    pipeline = DelayedPipeline(Nothing(), sink, lambda frame: [], Compositor(), 0.02, 100)
    thread = threading.Thread(target=pipeline.run, daemon=True)
    thread.start()
    pipeline.gate.panic()
    assert pipeline.gate.wait_for_black(2)
    assert is_black(sink.pictures()[0])
    pipeline._stop.set()
    thread.join(timeout=5)


def test_resume_never_shows_anything_captured_before_it(tmp_path):
    with Live(tmp_path, delay=0.05) as live:
        live.controller.panic()
        time.sleep(0.1)  # frames pile up in the queue
        before = live.source.next_id
        resumed_at = time.monotonic()
        live.controller.resume()
        assert wait_until(lambda: any(not is_black(i) for i in live.sink.after(resumed_at)))
        pictures = [image for image in live.sink.after(resumed_at) if not is_black(image)]
        assert all(stamped_id(image) >= before - 1 for image in pictures)
        for image in pictures:
            assert live.source.read_times[stamped_id(image)] >= resumed_at
        assert live.pipeline.stats.invalidated > 0


def test_after_resume_a_failed_scan_cannot_fall_back_on_a_picture_from_before(tmp_path):
    class FailsAfterResume:
        def __init__(self):
            self.failing = False

        def __call__(self, frame):
            if self.failing:
                raise RuntimeError("ocr died")
            return []

    detect = FailsAfterResume()
    with Live(tmp_path, detect) as live:
        live.controller.panic()
        detect.failing = True
        resumed_at = time.monotonic()
        held_before = live.pipeline.stats.rendered_held
        live.controller.resume()
        assert wait_until(lambda: len(live.sink.after(resumed_at)) > 10)
        assert all(is_black(image) for image in live.sink.after(resumed_at))
        # The scanner has let go of the old picture, rather than only being refused it.
        assert wait_until(lambda: live.pipeline.stats.rendered_blackout > 0)
        settled = live.pipeline.stats.rendered_held
        assert settled - held_before <= 2  # scans that were under way as it resumed
        time.sleep(0.05)
        assert live.pipeline.stats.rendered_held == settled


def test_a_new_keyword_covers_the_output_at_once_and_stays_covered_until_checked(tmp_path):
    with Live(tmp_path, delay=0.08) as live:
        assert wait_until(lambda: any(word_visible(i) for i in live.sink.pictures()))
        changed_at = time.monotonic()
        state = live.controller.set_config(live.config([KEYWORD]))
        assert live.detect.keywords == (KEYWORD,)
        assert state.keywords == [KEYWORD]
        assert wait_until(lambda: len(live.sink.after(changed_at + 0.15)) > 3)
        # Older results were buffered for 80 ms, and none of them may be sent now.
        assert not any(word_visible(image) for image in live.sink.after(changed_at + 0.002))
        late = live.sink.after(changed_at + 0.15)
        assert any(not is_black(image) for image in late)
        assert live.pipeline.stats.invalidated > 0


def test_a_stricter_change_covers_output_even_while_the_scanner_is_busy(tmp_path):
    detect = KeywordDetector(delay=0.05)
    with Live(tmp_path, detect, delay=0.02, fps=100) as live:
        assert wait_until(lambda: any(word_visible(i) for i in live.sink.pictures()))
        changed_at = time.monotonic()
        live.controller.set_config(live.config([KEYWORD]))
        assert wait_until(lambda: len(live.sink.after(changed_at)) > 10)
        assert not any(word_visible(image) for image in live.sink.after(changed_at + 0.002))


def test_a_change_that_only_removes_keeps_the_picture_up(tmp_path):
    with Live(tmp_path) as live:
        live.controller.set_config(live.config([KEYWORD]))
        changed_at = time.monotonic()  # now the new keyword is in force, the cover can lift
        assert wait_until(lambda: any(not is_black(i) for i in live.sink.after(changed_at)))
        generation = live.gate.required_generation
        removed_at = time.monotonic()
        live.controller.set_config(live.config([]))
        assert live.gate.required_generation == generation
        assert live.detect.keywords == ()
        assert wait_until(lambda: len(live.sink.after(removed_at)) > 10)
        assert not any(is_black(image) for image in live.sink.after(removed_at))


def test_a_failed_configuration_stays_covered_and_keeps_the_old_detector(tmp_path):
    with Live(tmp_path) as live:
        live.controller.set_config(live.config(["FIRST_WORD"]))
        live.detect.fail_next_configure = True
        before = live.controller.get_state()
        failed_at = time.monotonic()
        with pytest.raises(HTTPException) as caught:
            live.controller.set_config(live.config([KEYWORD]))
        assert caught.value.status_code == 503
        assert KEYWORD not in caught.value.detail  # nothing private in the error
        assert live.detect.keywords == ("FIRST_WORD",)
        state = live.controller.get_state()
        assert state.keywords == before.keywords
        assert "failed" in state.message
        assert wait_until(lambda: len(live.sink.after(failed_at + 0.1)) > 5)
        assert all(is_black(image) for image in live.sink.after(failed_at + 0.05))
        # Saving settings that work lifts the cover.
        lifted_at = time.monotonic()
        live.controller.set_config(live.config(["FIRST_WORD"]))
        assert wait_until(lambda: any(not is_black(i) for i in live.sink.after(lifted_at)))


def test_a_failed_configuration_does_not_change_the_saved_file(tmp_path):
    with Live(tmp_path) as live:
        live.controller.set_config(live.config(["FIRST_WORD"]))
        saved = (tmp_path / "config.json").read_text()
        live.detect.fail_next_configure = True
        with pytest.raises(HTTPException):
            live.controller.set_config(live.config([KEYWORD]))
        assert (tmp_path / "config.json").read_text() == saved
        assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


def test_settings_that_cannot_be_saved_change_nothing(tmp_path):
    with Live(tmp_path) as live:
        blocked = tmp_path / "blocked"
        blocked.write_text("a file where the folder should go")
        live.controller._store.path = blocked / "config.json"
        generation = live.gate.generation
        with pytest.raises(HTTPException):
            live.controller.set_config(live.config([KEYWORD]))
        assert live.detect.keywords == ()
        assert live.gate.generation == generation
        assert live.controller.get_state().keywords == []


def test_settings_reach_a_real_text_detector_and_change_its_results(tmp_path):
    class Reader:
        def __call__(self, image):
            box = [[20, 20], [60, 20], [60, 30], [20, 30]]
            return [(box, KEYWORD, 0.9)], None

    detect = TextDetector(reader=Reader())
    with Live(tmp_path, detect) as live:
        assert wait_until(lambda: any(word_visible(i) for i in live.sink.pictures()))
        changed_at = time.monotonic()
        live.controller.set_config(live.config([KEYWORD]))
        assert wait_until(lambda: any(not is_black(i) for i in live.sink.after(changed_at + 0.1)))
        shown = [image for image in live.sink.after(changed_at + 0.002) if not is_black(image)]
        assert shown and not any(word_visible(image) for image in shown)


def test_saved_keywords_are_on_from_the_first_frame(tmp_path):
    (tmp_path / "config.json").write_text('{"keywords": ["%s"]}' % KEYWORD)
    with Live(tmp_path) as live:
        assert live.detect.keywords == (KEYWORD,)
        assert not any(word_visible(image) for image in live.sink.pictures())


def test_turning_a_category_off_reaches_the_detector(tmp_path):
    detect = TextDetector(reader=lambda image: ([], None))
    with Live(tmp_path, detect) as live:
        live.controller.set_config(live.config([], enabled=["email", "street_sign"]))
        assert detect._detector._enabled == frozenset({"email"})


def test_a_run_with_no_real_detector_refuses_settings_rather_than_ignoring_them(tmp_path):
    with Live(tmp_path, lambda frame: []) as live:
        with pytest.raises(HTTPException) as caught:
            live.controller.set_config(live.config([KEYWORD]))
        assert caught.value.status_code == 409
        state = live.controller.get_state()
        assert set(state.detector_health.values()) == {"unavailable"}
        assert state.output_state == "unverified"  # never "protected" with no real detector
        assert "not being protected" in state.message


def test_state_reports_real_output_and_only_the_layers_that_exist(tmp_path):
    with Live(tmp_path) as live:
        state = live.controller.get_state()
        assert state.simulated is False
        assert state.output_connected is True
        assert state.output_state == "protected"
        assert state.irl_available is False
        assert state.detector_health["email"] == "ready"
        assert state.detector_health["street_sign"] == "unavailable"
        assert "planned" in state.message


def test_a_failing_detector_is_reported_as_degraded_and_covered(tmp_path):
    class Fails:
        failing = False

        def __call__(self, frame):
            if self.failing:
                raise RuntimeError("ocr died")
            return []

        def configure(self, keywords=(), enabled=None):
            pass

    detect = Fails()
    with Live(tmp_path, detect) as live:
        detect.failing = True
        assert wait_until(lambda: live.controller.get_state().output_state == "covered")
        state = live.controller.get_state()
        assert state.detector_health["email"] == "degraded"
        assert state.held_frames > 0


def test_catches_reach_the_log_as_categories_with_no_text(tmp_path):
    def finds(frame):
        return [
            Detection(1, 1, 5, 5, "email", 1.0, 0, frame.frame_id),
            Detection(9, 9, 5, 5, "unread", 1.0, 0, frame.frame_id),
        ]

    with Live(tmp_path, finds) as live:
        log = live.controller.get_events()
        assert [event.category for event in log.events] == ["email"]
        assert log.events[0].source == "screen"
        assert log.events[0].count == 1  # one catch, not one per frame


def test_a_broken_event_log_does_not_touch_the_stream(tmp_path):
    def finds(frame):
        return [Detection(20, 20, 40, 10, "email", 1.0, 0, frame.frame_id)]

    with Live(tmp_path, finds) as live:
        def broken(*args):
            raise RuntimeError("log full")

        live.pipeline.on_found = broken
        sends = len(live.sink.sent)
        assert wait_until(lambda: len(live.sink.sent) > sends + 20)
        assert live.pipeline.stats.event_errors > 0
        assert live.pipeline.last_output in ("scanned", "cached")


def test_the_routes_drive_the_real_pipeline(tmp_path):
    with Live(tmp_path) as live:
        with TestClient(create_app(live.controller), base_url="http://127.0.0.1:8000") as client:
            assert client.get("/api/health").json()["pipeline_mode"] == "live"
            assert client.get("/api/state").json()["simulated"] is False
            panicked = client.post("/api/panic").json()
            assert panicked["panic"] is True
            assert panicked["blackout_confirmed"] is True
            assert client.post("/api/resume").json()["panic"] is False
            response = client.put(
                "/api/config", json={"enabled": ALL, "keywords": [KEYWORD]}
            )
            assert response.status_code == 200
            assert live.detect.keywords == (KEYWORD,)
            assert client.put("/api/mode", json={"mode": "irl"}).status_code == 409


def test_the_mock_is_only_there_when_asked_for(tmp_path):
    simulated = create_simulation_app()
    assert isinstance(simulated.state.controller, MockPipelineController)
    with pytest.raises(TypeError):
        create_app()
    with TestClient(simulated, base_url="http://127.0.0.1:8000") as client:
        assert client.get("/api/health").json()["pipeline_mode"] == "mock"


def test_main_connects_the_real_controller(monkeypatch):
    built = {}

    class Recording(app.LivePipelineController):
        def __init__(self, pipeline, configure):
            super().__init__(pipeline, configure, Path("/nonexistent/config.json"))
            built["controller"] = self

    def controls(controller, panel, hotkey):
        built.update(panel=panel, hotkey=hotkey)
        return None, None

    class FakeCam:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def send(self, image):
            pass

    monkeypatch.setattr(app, "LivePipelineController", Recording)
    monkeypatch.setattr(app, "start_controls", controls)
    monkeypatch.setattr(app, "VirtualCam", FakeCam)
    assert app.main(["--test-pattern", "--max-frames", "3", "--delay", "0.02"]) == 0
    assert isinstance(built["controller"], LivePipelineController)
    assert built["panel"] is False and built["hotkey"] is False  # a bounded run is a measurement


def test_a_stream_gets_the_panel_and_the_key_unless_told_otherwise(monkeypatch):
    calls = []

    class Stop(Exception):
        pass

    def controls(controller, panel, hotkey):
        calls.append((panel, hotkey))
        raise Stop

    monkeypatch.setattr(app, "start_controls", controls)
    monkeypatch.setattr(app, "VirtualCam", lambda fps: None)
    with pytest.raises(Stop):
        app.main(["--test-pattern"])
    with pytest.raises(Stop):
        app.main(["--test-pattern", "--no-panel", "--no-hotkey"])
    assert calls == [(True, True), (False, False)]


class FakeListener:
    def __init__(self, keys, on_press):
        self.keys, self.on_press = keys, on_press
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


def test_the_panic_key_presses_the_same_panic(tmp_path):
    with Live(tmp_path) as live:
        made = []

        def listener(keys, on_press):
            made.append(FakeListener(keys, on_press))
            return made[0]

        key = PanicHotkey(live.controller.panic, make_listener=listener, allowed=lambda: True)
        assert key.start() is True and key.available
        made[0].on_press()
        assert live.gate.panicked
        assert wait_until(lambda: live.gate.black_confirmed)
        key.stop()
        assert not made[0].running


def test_a_panic_key_that_cannot_start_says_so():
    def broken(keys, on_press):
        raise OSError("no listener here")

    assert PanicHotkey(lambda: None, make_listener=broken, allowed=lambda: True).start() is False
    never = PanicHotkey(lambda: None, make_listener=FakeListener, allowed=lambda: False)
    assert never.start() is False
    assert never.available is False


def test_black_sent_for_one_panic_does_not_answer_a_later_one():
    gate = OutputGate()
    gate.panic()
    ticket = gate.black_ticket()
    gate.resume()
    gate.panic()
    gate.confirm_black(ticket)
    assert gate.panicked
    assert not gate.black_confirmed


def test_a_new_panic_needs_a_new_black_frame():
    gate = OutputGate()
    gate.panic()
    gate.confirm_black(gate.black_ticket())
    assert gate.black_confirmed
    gate.resume()
    assert not gate.black_confirmed
    gate.panic()
    assert not gate.black_confirmed


def test_pending_settings_keep_confirmed_state_and_file_until_acknowledged(tmp_path, monkeypatch):
    from eclipse.compositor.delay import PolicyChange

    live = Live(tmp_path)
    live.controller._apply_wait = 0
    live.controller._store.save([])
    changes = []

    def queue(apply, stricter):
        change = PolicyChange(apply, 1)
        changes.append(change)
        return change

    monkeypatch.setattr(live.pipeline, 'change_policy', queue)
    state = live.controller.set_config(live.config([KEYWORD]))
    assert state.config_status == 'pending'
    assert state.keywords == []
    assert live.controller._store.load() == []
    with pytest.raises(HTTPException) as caught:
        live.controller.set_config(live.config(['OTHER']))
    assert caught.value.status_code == 409
    # Pending settings do not prevent the authoritative panic request.
    assert live.controller.panic().panic is True
    changes[0].apply()
    changes[0].finish()
    state = live.controller.get_state()
    assert state.config_status == 'applied'
    assert state.keywords == [KEYWORD]
    assert live.controller._store.load() == [KEYWORD]
    assert state.panic is True


def test_late_settings_failure_preserves_previous_settings_and_allows_retry(tmp_path, monkeypatch):
    from eclipse.compositor.delay import PolicyChange

    live = Live(tmp_path)
    live.controller._apply_wait = 0
    live.controller._store.save([])
    change = PolicyChange(lambda: None, 1)
    monkeypatch.setattr(live.pipeline, 'change_policy', lambda *args: change)
    assert live.controller.set_config(live.config([KEYWORD])).config_status == 'pending'
    change.finish(ValueError(KEYWORD))
    state = live.controller.get_state()
    assert state.config_status == 'failed'
    assert KEYWORD not in state.config_message
    assert state.keywords == []
    assert live.controller._store.load() == []
    assert not (tmp_path / 'config.tmp').exists()
    replacement = PolicyChange(lambda: None, 2)
    replacement.finish()
    monkeypatch.setattr(live.pipeline, 'change_policy', lambda *args: replacement)
    assert live.controller.set_config(live.config([])).config_status == 'applied'


def test_persistence_failure_reports_applied_but_unsaved_settings(tmp_path, monkeypatch):
    live = Live(tmp_path)

    def fail_commit(staged):
        raise HTTPException(status_code=503, detail='disk unavailable')

    monkeypatch.setattr(live.controller._store, 'commit', fail_commit)
    with pytest.raises(HTTPException):
        live.controller.set_config(live.config([KEYWORD]))
    state = live.controller.get_state()
    assert state.config_status == 'failed'
    assert state.keywords == [KEYWORD]
    assert live.detect.keywords == (KEYWORD,)
    assert 'Settings applied' in state.config_message
    assert 'could not be saved' in state.config_message
