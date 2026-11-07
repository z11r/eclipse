import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from eclipse.compositor.compositor import Compositor
from eclipse.compositor.delay import DelayedPipeline
from eclipse.platform import AppleVisionReader
from eclipse.text_detect import TextDetector
from eclipse.text_detect.detection import DetectionError
from eclipse.text_detect.read_ahead import (
    OcrPool,
    ReadAhead,
    ReadError,
    bands_to_read,
    changed_rows,
    words_in_area,
)
from eclipse.types import Frame

needs_mac = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")


def word(x, y, text="word", width=40, height=20):
    return ([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], text, 1.0)


class FakeJob:
    def __init__(self, words=(), failed=False, gate=None):
        self.words, self.failed, self.gate = list(words), failed, gate

    def result(self):
        if self.gate is not None:
            self.gate.wait(2)
        if self.failed:
            raise ReadError("read failed")
        return self.words


class FakePool:
    backlog = 0
    size = 4

    def __init__(self, words=(), failed=False):
        self.stats = {"jobs": 0, "failed_reads": 0, "wait_seconds": 0.0, "work_seconds": 0.0}
        self.images = []
        self.words = words
        self.failed = failed

    def submit(self, image, urgent=False):
        self.images.append(image)
        self.urgent = urgent
        return FakeJob(self.words, self.failed)


def frame(image, frame_id=0):
    return Frame(image, time.monotonic(), "screen", frame_id)


def blank(height=400, width=300):
    return np.full((height, width, 3), 20, np.uint8)


def test_changed_rows_notices_one_level_in_one_channel_of_one_pixel():
    before = blank()
    now = before.copy()
    now[123, 45, 2] += 1
    assert changed_rows(now, before).tolist() == [123]


def test_changed_rows_is_empty_for_identical_frames():
    assert changed_rows(blank(), blank()).size == 0


def test_bands_cover_every_changed_row_with_margin():
    assert bands_to_read(np.array([100]), 400, 10) == [(90, 111)]


def test_bands_clip_to_the_frame():
    assert bands_to_read(np.array([2, 398]), 400, 10) == [(0, 13), (388, 400)]


def test_nearby_changes_share_one_band():
    assert bands_to_read(np.array([100, 115]), 400, 10) == [(90, 126)]


def test_no_change_needs_no_reading():
    assert bands_to_read(np.array([], dtype=int), 400, 10) == []


def test_words_are_moved_to_the_areas_own_coordinates():
    assert words_in_area([word(100, 10)], 200, 205, 90, 30, 100) == [
        ([[10, 5], [50, 5], [50, 25], [10, 25]], "word", 1.0)
    ]


def test_words_outside_the_area_are_left_out():
    words = [word(0, 0, "left"), word(500, 0, "right"), word(100, 300, "below")]
    assert words_in_area(words, 0, 0, 90, 40, 100) == []


def test_a_word_that_only_touches_the_area_is_kept():
    assert len(words_in_area([word(80, 0)], 0, 0, 100, 40, 100)) == 1


def test_first_frame_is_read_whole_and_a_still_frame_not_at_all():
    pool = FakePool()
    reader = ReadAhead(pool)
    reader.prepare(frame(blank(), 0))
    assert pool.images[0].shape == blank().shape
    reader.prepare(frame(blank(), 1))
    assert len(pool.images) == 1


def test_a_change_of_frame_size_is_read_whole():
    pool = FakePool()
    reader = ReadAhead(pool)
    reader.prepare(frame(blank(100, 100), 0))
    reader.prepare(frame(blank(200, 100), 1))
    assert pool.images[1].shape[:2] == (200, 100)


def test_only_the_changed_band_is_read_and_at_full_width():
    pool = FakePool()
    reader = ReadAhead(pool, margin=10)
    before = blank()
    now = before.copy()
    now[200, 5] = 255
    reader.prepare(frame(before, 0))
    reader.prepare(frame(now, 1))
    assert pool.images[1].shape == (21, 300, 3)


def test_an_area_inside_a_read_band_is_answered_without_reading_again():
    pool = FakePool([word(50, 5)])
    reader = ReadAhead(pool)
    image = blank()
    reader.prepare(frame(image, 0))
    reader.begin_frame(0)
    words, _ = reader.read_area(image[0:40, 40:140], 0, 40)
    assert [corners[0] for corners, _, _ in words] == [[10, 5]]
    assert len(pool.images) == 1
    assert reader.stats["hits"] == 1


def test_an_area_nobody_read_is_read_on_the_spot():
    pool = FakePool([word(1, 1)])
    reader = ReadAhead(pool, margin=10)
    before = blank()
    now = before.copy()
    now[300, 5] = 255
    reader.prepare(frame(before, 0))
    reader.prepare(frame(now, 1))
    reader.begin_frame(1)
    area = now[0:100, 0:100]
    words, _ = reader.read_area(area, 0, 0)
    assert len(words) == 1
    assert pool.images[-1] is area
    assert reader.stats["misses"] == 1


def test_an_unprepared_frame_is_still_read():
    pool = FakePool([word(1, 1)])
    reader = ReadAhead(pool)
    reader.begin_frame(5)
    image = blank()
    assert len(reader.read_area(image, 0, 0)[0]) == 1


def test_one_frames_answer_is_never_used_for_another():
    pool = FakePool()
    reader = ReadAhead(pool, margin=10)
    first, second = blank(), blank()
    second[100, 5] = 255
    reader.prepare(frame(first, 0))
    reader.prepare(frame(second, 1))
    reader.begin_frame(1)  # frame 0 was dropped
    assert 0 not in reader._read
    reader.begin_frame(1)
    assert reader._bands == []  # collected once only


def test_a_failed_read_ahead_raises_rather_than_reporting_no_text():
    reader = ReadAhead(FakePool(failed=True))
    image = blank()
    reader.prepare(frame(image, 0))
    reader.begin_frame(0)
    with pytest.raises(ReadError):
        reader.read_area(image, 0, 0)


def test_a_failed_read_holds_the_frame_back():
    detect = TextDetector(reader=ReadAhead(FakePool(failed=True)))
    image = blank()
    shown = frame(image, 0)
    detect.prepare(shown)
    with pytest.raises(DetectionError):
        detect(shown)
    assert not Compositor().process(shown, detect).any()  # black, not the raw frame


def test_prepare_failing_never_lets_a_frame_skip_the_scan():
    class Source:
        def __init__(self):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            self.count += 1
            return frame(blank(), self.count)

    class Detect:
        def __init__(self):
            self.scanned = 0

        def prepare(self, frame):
            raise RuntimeError("no head start today")

        def __call__(self, frame):
            self.scanned += 1
            return []

    class Sink:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def send(self, image):
            time.sleep(0.001)

    detect = Detect()
    stats = DelayedPipeline(
        Source(), Sink(), detect, Compositor(padding=0), 0.01, 200, 12
    ).run()
    assert detect.scanned == 12
    assert stats.prepare_failures == 12


def test_a_frame_no_reader_can_reach_is_shed_without_reading_it():
    pool = FakePool()
    pool.backlog = 12
    reader = ReadAhead(pool)
    assert reader.prepare(frame(blank(), 0)) is False
    assert pool.images == []
    assert reader.stats["shed"] == 1


def test_a_shed_frame_cannot_hide_a_change_from_the_next_read():
    pool = FakePool()
    reader = ReadAhead(pool, margin=10)
    start, changed = blank(), blank()
    changed[200, 5] = 255
    assert reader.prepare(frame(start, 0))
    pool.backlog = 99
    assert reader.prepare(frame(changed, 1)) is False  # the change arrives on a shed frame
    pool.backlog = 0
    assert reader.prepare(frame(changed.copy(), 2))  # identical to frame 1
    assert pool.images[-1].shape[0] == 21  # still read, as it's new since the last taken frame


def test_a_refused_frame_is_dropped_and_never_shown_unscanned():
    class Source:
        def __init__(self):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            self.count += 1
            return frame(blank(), self.count)

    class Detect:
        def __init__(self):
            self.scanned = []

        def prepare(self, frame):
            return frame.frame_id % 2 == 0

        def __call__(self, frame):
            self.scanned.append(frame.frame_id)
            return []

    class Sink:
        def __init__(self):
            self.sent = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def send(self, image):
            self.sent += 1
            time.sleep(0.001)

    detect, sink = Detect(), Sink()
    stats = DelayedPipeline(Source(), sink, detect, Compositor(padding=0), 0.01, 200, 20).run()
    assert all(frame_id % 2 == 0 for frame_id in detect.scanned)
    assert stats.shed == 10
    assert stats.shown == stats.scanned == 10
    assert stats.dropped == 10


def test_text_detector_passes_on_a_refusal():
    pool = FakePool()
    pool.backlog = 99
    detect = TextDetector(reader=ReadAhead(pool))
    assert detect.prepare(frame(blank(), 0)) is False


def test_a_detector_without_read_ahead_never_refuses():
    assert TextDetector(reader=lambda image: ([], None)).prepare(frame(blank(), 0)) is True


def test_mostly_changed_frames_are_read_whole():
    pool = FakePool()
    reader = ReadAhead(pool, margin=10)
    before, now = blank(400, 300), blank(400, 300)
    now[100:300] = 200  # half the rows; the top and bottom stay still
    reader.prepare(frame(before, 0))
    reader.prepare(frame(now, 1))
    assert pool.images[-1].shape[0] == 400
    reader.begin_frame(1)
    reader.read_area(now, 0, 0)  # what the detector asks for on a full scan
    assert reader.stats["hits"] == 1 and reader.stats["misses"] == 0


def test_a_small_change_is_still_read_as_a_band():
    pool = FakePool()
    reader = ReadAhead(pool, margin=10)
    before, now = blank(400, 300), blank(400, 300)
    now[200, 5] = 255
    reader.prepare(frame(before, 0))
    reader.prepare(frame(now, 1))
    assert pool.images[-1].shape[0] == 21


def test_reads_the_scan_waits_on_go_first():
    from queue import PriorityQueue
    from itertools import count

    pool = OcrPool.__new__(OcrPool)
    pool.stats = {"failed_reads": 0}
    pool._alive, pool._alive_lock = 1, threading.Lock()
    pool._jobs, pool._order = PriorityQueue(), count()
    ahead = [pool.submit(blank()) for _ in range(3)]
    urgent = pool.submit(blank(), urgent=True)
    order = [pool._jobs.get()[2] for _ in range(4)]
    assert order[0] is urgent
    assert order[1:] == ahead  # reading ahead keeps its own order


def test_reading_on_the_spot_is_urgent():
    pool = FakePool()
    reader = ReadAhead(pool)
    reader.begin_frame(0)
    reader.read_area(blank(), 0, 0)
    assert pool.urgent is True


def test_a_slow_read_makes_the_scan_wait_rather_than_skip_ahead():
    gate = threading.Event()

    class Pool(FakePool):
        def submit(self, image):
            return FakeJob([word(1, 1)], gate=gate)

    reader = ReadAhead(Pool())
    image = blank()
    reader.prepare(frame(image, 0))
    reader.begin_frame(0)
    answered = []
    thread = threading.Thread(target=lambda: answered.append(reader.read_area(image, 0, 0)))
    thread.start()
    time.sleep(0.03)
    assert not answered  # still waiting on this frame's own answer
    gate.set()
    thread.join(2)
    assert answered


def text_page(text, y=300):
    image = np.full((900, 1600, 3), 30, dtype=np.uint8)
    cv2.putText(image, text, (200, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (240, 240, 240), 3)
    return image


@pytest.fixture(scope="module")
def pool():
    pool = OcrPool(2)
    yield pool
    pool.close()


@needs_mac
class TestRealWorkers:
    def test_reads_the_same_words_as_reading_in_process(self, pool):
        image = text_page("alex.fake@example.com")
        direct, _ = AppleVisionReader()(image)
        ahead = pool.submit(image).result()
        assert [text for _, text, _ in ahead] == [text for _, text, _ in direct]
        for (corners, _, _), (others, _, _) in zip(ahead, direct):
            assert np.allclose(corners, others, atol=0.5)

    def test_many_reads_finish_side_by_side(self, pool):
        jobs = [pool.submit(text_page(f"line {index}")) for index in range(8)]
        assert all(job.result() for job in jobs)

    def test_finds_the_same_secret_as_direct_reads_after_a_change(self, pool):
        ahead = TextDetector(reader=ReadAhead(pool), pixel_scale=2)
        direct = TextDetector(reader=AppleVisionReader(), pixel_scale=2)
        pages = [text_page("nothing here"), text_page("alex.fake@example.com")]
        found = {}
        for index, image in enumerate(pages):
            shown = Frame(image, index * 0.03, "screen", index)
            ahead.prepare(shown)
            found["ahead"] = ahead(shown)
            found["direct"] = direct(shown)
        assert {d.category for d in found["ahead"]} == {"email"}
        one, other = found["ahead"][0], found["direct"][0]
        assert abs(one.x - other.x) <= 2 and abs(one.w - other.w) <= 4
        assert abs(one.y - other.y) <= 2 and abs(one.h - other.h) <= 4


@needs_mac
def test_a_worker_dying_fails_reads_instead_of_stalling_or_passing_them():
    pool = OcrPool(1)
    try:
        worker = pool._workers[0]
        os.kill(worker._process.pid, signal.SIGKILL)
        worker._process.wait(timeout=5)
        job = pool.submit(text_page("hello"))
        with pytest.raises(ReadError):
            job.result()
        assert pool.stats["failed_reads"] >= 1
    finally:
        pool.close()


@needs_mac
def test_losing_one_worker_leaves_the_others_reading():
    pool = OcrPool(2)
    try:
        victim = pool._workers[0]
        os.kill(victim._process.pid, signal.SIGKILL)
        victim._process.wait(timeout=5)
        image = text_page("still reading")
        for _ in range(40):  # the dead worker's feeder takes one job, fails it and stops
            if pool.alive == 1:
                break
            try:
                pool.submit(image).result()
            except ReadError:
                pass
        assert pool.alive == 1
        results = [pool.submit(image).result() for _ in range(6)]
        assert all(results)
    finally:
        pool.close()


def test_a_worker_that_dies_during_startup_reports_a_read_error(monkeypatch):
    from eclipse.text_detect import read_ahead

    real_popen = read_ahead.subprocess.Popen

    def dies_at_once(arguments, **options):
        return real_popen([sys.executable, "-c", "pass"], **options)

    monkeypatch.setattr(read_ahead.subprocess, "Popen", dies_at_once)
    with pytest.raises(ReadError):
        OcrPool(1)


def test_losing_every_worker_fails_reads_at_once():
    pool = OcrPool.__new__(OcrPool)
    pool.stats = {"failed_reads": 0}
    pool._alive = 0
    pool._alive_lock = threading.Lock()
    pool._jobs = None
    with pytest.raises(ReadError):
        pool.submit(blank()).result()


def test_jobs_for_dropped_frames_are_cancelled():
    jobs = []

    class Pool(FakePool):
        def submit(self, image):
            job = super().submit(image)
            job.cancelled = False
            jobs.append(job)
            return job

    reader = ReadAhead(Pool())
    reader.prepare(frame(blank(), 0))
    second = blank()
    second[10, 10] = 200
    reader.prepare(frame(second, 1))
    reader.begin_frame(1)
    assert jobs[0].cancelled
    assert not jobs[1].cancelled


@needs_mac
def test_a_cancelled_job_is_skipped_rather_than_read(pool):
    busy = [pool.submit(text_page("keep the workers busy")) for _ in range(2)]
    skipped = pool.submit(text_page("nobody wants this"))
    skipped.cancelled = True
    before = pool.stats["cancelled"]
    assert all(job.result() for job in busy)
    with pytest.raises(ReadError):
        skipped.result()
    assert pool.stats["cancelled"] == before + 1


@needs_mac
def test_a_read_frame_is_released_once_read(pool):
    job = pool.submit(text_page("let go of me"))
    job.result()
    assert job.image is None


@needs_mac
def test_oversized_images_fail_instead_of_writing_past_shared_memory(monkeypatch):
    from eclipse.text_detect import read_ahead

    pool = OcrPool(1)
    try:
        monkeypatch.setattr(read_ahead, "MAX_FRAME_PIXELS", 10)
        with pytest.raises(ReadError):
            pool.submit(blank(100, 100)).result()
    finally:
        pool.close()
