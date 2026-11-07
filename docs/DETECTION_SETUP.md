# Detection setup

In VS Code, open Terminal > New Terminal in this project folder.
Install Python 3.11 if it is not installed, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-detection.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Use from the pipeline

```python
from detection import Detector, DetectionError, apply_covers

detector = Detector(keywords=['fake name'])  # Create once per source.

# Inside the pipeline, for each BGR uint8 image:
try:
    boxes = detector.detect(frame)
    safe_frame = apply_covers(frame, boxes)
except DetectionError:
    safe_frame = frame * 0
```

- Input: NumPy image with shape (height, width, 3), BGR color order.
- Output: `(x, y, width, height, category)` boxes, with 10px padding.
- Categories: email, phone, 2fa, api_key, card, keyword.
- `detector.configure(keywords=[...], enabled=[...])` replaces settings.
- `detector.reset()` clears cached results on resume or source changes.
- For a single source, `from detection import detect` also works.

## What still belongs in the pipeline

Capture, a minimum 1-second delay, F8 blackout, buffer clearing, and OBS output.
Never release an unscanned frame: black out or hold the last safe frame if OCR
fails, hangs, or misses its deadline. Keep confirmed covers on buffered frames.
Logs must contain categories/timestamps only, never OCR text or screenshots.

## Current limits

Only pixel-identical frames reuse OCR results. Any change triggers a full scan;
changed-region scanning is not implemented. Animated screens may be slow.
OCR can miss secrets; regex rules do not recognize every token format.
Phone rules mostly target North America. Ordinary 4-8 digit numbers can be
flagged as 2FA. Split OCR text may cause an entire row to be covered.

Tests use fake OCR. Real OCR accuracy and latency still need laptop testing.
Use fake secrets only. The pinned OCR package uses local inference and bundled
models; dependency installation requires internet. Running detection.py by itself
does nothing: import its functions from another script.
