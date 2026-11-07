import numpy as np

from eclipse import platform
from eclipse.output import virtual_cam
from eclipse.output.virtual_cam import bgr_to_uyvy


def libyuv_uyvy(bgr):
    """What pyvirtualcam sends for a BGR frame: libyuv's BT.601 video range UYVY."""
    b, g, r = (bgr[..., channel].astype(np.int32) for channel in range(3))
    luma = (66 * r + 129 * g + 25 * b + 0x1080) >> 8
    pair_b, pair_g, pair_r = ((c[:, 0::2] + c[:, 1::2] + 1) >> 1 for c in (b, g, r))
    blue = (112 * pair_b - 74 * pair_g - 38 * pair_r + 0x8080) >> 8
    red = (112 * pair_r - 94 * pair_g - 18 * pair_b + 0x8080) >> 8
    height, width = luma.shape
    out = np.stack([blue, luma[:, 0::2], red, luma[:, 1::2]], axis=-1)
    return out.reshape(height, width * 2)


def test_uyvy_matches_what_pyvirtualcam_sent():
    image = np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    difference = np.abs(bgr_to_uyvy(image).astype(int) - libyuv_uyvy(image))
    assert difference.max() <= 1


def test_uyvy_black_and_white_are_video_range():
    black = bgr_to_uyvy(np.zeros((2, 4, 3), np.uint8))
    white = bgr_to_uyvy(np.full((2, 4, 3), 255, np.uint8))
    assert black.shape == (2, 8)
    assert list(black[0, :4]) == [128, 16, 128, 16]  # zeros would show as green
    assert list(white[0, :4]) == [128, 235, 128, 235]


def test_panic_black_is_real_black_on_the_stamped_camera(monkeypatch):
    sent = []

    class Camera:
        device = "fake"
        prepare = staticmethod(bgr_to_uyvy)

        def send(self, image, due):
            sent.append(image.copy())

        def close(self):
            pass

    monkeypatch.setattr(virtual_cam, "open_camera", lambda width, height, fps: Camera())
    with virtual_cam.VirtualCam(fps=500) as cam:
        cam.send(np.full((64, 64, 3), 200, np.uint8), lambda: False)
    assert set(np.unique(sent[0][:, 1::2])) == {16} and set(np.unique(sent[0][:, 0::2])) == {128}


def test_only_macos_gets_the_stamped_camera(monkeypatch):
    monkeypatch.setattr(platform, "IS_MAC", False)
    assert platform.stamped_virtual_camera(1920, 1080) is None
