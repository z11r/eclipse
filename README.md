# Eclipse

Other tools catch your API key. Eclipse catches the things that let people find your house.

Eclipse is a desktop app that sits between your screen and webcam and OBS. It hides anything that leaks your secrets or shows where you live before a frame goes out. Built for natHacks 2026 (Standard Division, Defense & Security). See [SPEC.md](SPEC.md) for the full plan.

> **Status:** in development. The app does not run yet; this README covers setup only and will grow with the build.

## Setup

You need Python 3.11 and OBS 30 or later.

macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS, once per laptop

1. In System Settings → Privacy & Security, allow Screen Recording, Camera and Accessibility for whatever runs Python (Terminal, iTerm or VS Code).
2. In OBS, start and stop the virtual camera once, and approve the camera extension when macOS asks.

## Running

Run Eclipse as a module from the repo root:

```bash
python -m eclipse.main
```

Don't run `python eclipse/main.py`. That puts the package folder on Python's import path, where our `platform.py` hides the standard library module of the same name.

To debug with the same input every time, play a recorded clip through the live pipeline instead of the screen. It plays at the clip's own speed and stops when the clip ends:

```bash
python -m eclipse.main --clip eval/clips/planted_secrets.mp4
```

Add `--loop` to replay it, and `--clip-source webcam` or `--clip-source irl` to tag its frames as that source.

To stream the screen and a webcam together, add the camera's index. It goes in the bottom right corner, a quarter of the screen's width (change that with `--pip-size 0.3`):

```bash
python -m eclipse.main --with-webcam 0
```

The screen and the camera are each scanned on their own, side by side, and joined only after both are covered. A face moving in the corner never makes the screen be read again, and a camera picture that hasn't changed since the last frame isn't read at all. A video file path in place of the index stands in for the camera, looped, and it works with `--clip` too.

With `--with-webcam`, you can switch live between three views: `screen`, `both` (the default) and `webcam` (the camera alone, full size). Start on another one with `--view screen`. To switch:

- Press Ctrl+Alt+V to step through screen, both, webcam and round again.
- Or, for the panel, call `PUT /api/view` with `{"view": "screen" | "both" | "webcam"}`. It returns the panel state. `GET /api/state` reports `view` (the current one) and `views` (the ones this run can switch to). A run without a webcam reports `null` and `[]`, and `PUT /api/view` returns 409.

Both inputs keep running in every view, so a switch takes effect on the next frame. A switch that adds a source (screen → both) just shows up after the usual delay. A switch that drops one (both → screen, or anything → webcam) cuts the output at once: black goes out for about the delay (1 s by default) until the new view's first checked frame arrives, so nothing of the dropped source is shown after the switch.

If the camera's scan falls behind, the screen doesn't wait for it. The corner keeps the last camera picture that was checked, with its covers, so only the corner slows down. The screen waits only if that picture would be more than half a second behind.

Run the tests with:

```bash
python -m pytest
```

## Layout

```text
eclipse/    the app: capture, text_detect, visual_detect, track, compositor, output, panel
eval/       test clips, ground-truth labels and the eval harness
tests/      unit tests
```

Test clips use planted fake secrets and clues only. Never commit a real secret or address, even for testing.

## Credits

- Secret rules are inspired by [gitleaks](https://github.com/gitleaks/gitleaks) patterns.
- OCR by [RapidOCR](https://github.com/RapidAI/RapidOCR).
- Virtual camera output through [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) and [OBS](https://obsproject.com/).
