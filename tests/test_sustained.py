"""Short runs of the sustained test. The long one is python -m eval.sustained_motion."""

from eval.scroll_scene import FAKE_EMAIL, FAKE_PHONE
from eval.sustained_motion import FAKE_NOTE, failures, run


def test_nothing_leaks_while_everything_goes_wrong_at_once(tmp_path):
    report = run(seconds=5, fps=60, delay=0.15, config_dir=tmp_path)
    assert failures(report) == [], report
    assert report["pictures_checked"] > 100
    assert report["black_sends"] > 0
    assert report["panic_reported_black"] is True
    assert report["panic_to_black_ms"] < 100
    assert report["ocr_failures_held"] > 0
    assert report["invalidated"] > 0


def test_the_check_catches_secrets_the_detector_cannot_see(tmp_path):
    report = run(
        seconds=2, delay=0.05, config_dir=tmp_path, blind_to=(FAKE_EMAIL, FAKE_PHONE, FAKE_NOTE)
    )
    assert "exposed_secret_pixels" in failures(report)
    assert report["exposed_secret_pixels"] > 1000


def test_the_check_catches_a_keyword_that_never_took_effect(tmp_path):
    report = run(seconds=2, delay=0.05, config_dir=tmp_path, add_keyword=False)
    assert "exposed_keyword_pixels" in failures(report)
