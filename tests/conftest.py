import pytest


@pytest.fixture(autouse=True)
def no_real_obs_camera(monkeypatch):
    """No test reaches OBS. VirtualCam falls back to pyvirtualcam, which tests fake."""
    from eclipse.output import virtual_cam

    monkeypatch.setattr(virtual_cam, "stamped_virtual_camera", lambda width, height: None)
