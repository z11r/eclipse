import sys

import numpy as np
import pytest

from eclipse.text_detect.detection import find_matches

needs_mac = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS and Menlo")

stress = pytest.importorskip("eval.stress_60") if sys.platform == "darwin" else None


def tiny_mask():
    mask = np.zeros((4, 6), bool)
    mask[1:3, 1:5] = True
    return mask


@needs_mac
def test_all_glyphs_showing_when_nothing_covers_them():
    output = np.full((100, 100, 3), 200, np.uint8)
    assert stress.uncovered(output, 10, 10, tiny_mask()) == (8, 8)


@needs_mac
def test_nothing_showing_under_a_black_cover():
    output = np.full((100, 100, 3), 200, np.uint8)
    output[8:20, 8:20] = 0
    assert stress.uncovered(output, 10, 10, tiny_mask()) == (0, 8)


@needs_mac
def test_a_partly_covered_secret_counts_only_the_showing_part():
    output = np.full((100, 100, 3), 200, np.uint8)
    output[:, :13] = 0  # covers the two left columns of glyphs
    assert stress.uncovered(output, 10, 10, tiny_mask()) == (4, 8)


@needs_mac
def test_a_secret_off_screen_counts_for_nothing():
    output = np.full((100, 100, 3), 200, np.uint8)
    assert stress.uncovered(output, 200, 10, tiny_mask()) == (0, 0)


@needs_mac
def test_a_secret_cut_by_the_frame_edge_counts_its_visible_part():
    output = np.full((100, 100, 3), 200, np.uint8)
    assert stress.uncovered(output, -3, 10, tiny_mask()) == (4, 4)


@needs_mac
def test_rows_hidden_behind_a_title_bar_are_not_on_screen():
    output = np.full((100, 100, 3), 200, np.uint8)
    assert stress.uncovered(output, 10, 10, tiny_mask(), top_limit=12) == (4, 4)


@needs_mac
def test_every_planted_value_is_something_the_rules_would_flag():
    for _, prefix, value in stress.SECRETS:
        assert find_matches(prefix + value), value


@needs_mac
@pytest.mark.parametrize("name", list(getattr(stress, "SCENARIOS", {})))
def test_scenarios_make_retina_frames_and_say_where_secrets_are(name):
    scenario = stress.SCENARIOS[name](240)
    planted = 0
    for index in (0, 37, 110, 239):
        image, secrets = scenario.frame(index)
        assert image.shape == (stress.HEIGHT, stress.WIDTH, 3)
        for _, x, y, mask, top_limit in secrets:
            showing, on_screen = stress.uncovered(image, x, y, mask, top_limit)
            planted += on_screen
            assert showing == on_screen  # the mask lands exactly on visible text
    if name != "typing":
        assert planted > 0


@needs_mac
def test_video_needs_nearly_the_whole_frame_read_every_frame():
    from eclipse.text_detect.read_ahead import bands_to_read, changed_rows

    scenario = stress.Video(4)
    first, _ = scenario.frame(1)
    second, secrets = scenario.frame(2)
    assert secrets
    bands = bands_to_read(changed_rows(second, first), stress.HEIGHT, 96)
    assert sum(bottom - top for top, bottom in bands) > 0.9 * stress.HEIGHT


@needs_mac
def test_typed_secret_only_counts_once_all_of_it_is_on_screen():
    scenario = stress.Typing(1200)
    assert scenario.frame(10)[1] == []
    assert scenario.frame(499)[1]


@needs_mac
def test_harness_catches_a_pipeline_that_covers_nothing(tmp_path):
    """A leak detector that can't fail proves nothing, so make it fail."""
    scenario = stress.Notification(60)
    result = stress.run_one(
        scenario, lambda frame: [], 0.05, 60, False, str(tmp_path / "run.log")
    )
    assert result["frames_with_secret"] > 0
    assert result["frames_leaking"] == result["frames_with_secret"]
    assert result["leaked_share"] == 1.0
    assert set(result["leaks_by_secret"]) == {"two_fa", "email"}
