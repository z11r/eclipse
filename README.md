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
