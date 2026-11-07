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
from eclipse.text_detect.detection import Detector, DetectionError, apply_covers

detector = Detector(keywords=['fake name'])  # Create once per source.

# Inside the pipeline, for each BGR uint8 image:
try:
    boxes = detector.detect(frame)
    safe_frame = apply_covers(frame, boxes)
except DetectionError:
    safe_frame = frame * 0
```

- Input: NumPy image with shape (height, width, 3), BGR color order.
- Output: `(x, y, width, height, category)` boxes. The compositor adds the padding.
- Categories: email, phone, 2fa, api_key, card, keyword.
- `detector.configure(keywords=[...], enabled=[...])` replaces settings.
- `detector.reset()` clears cached results on resume or source changes.
- The pipeline uses `TextDetector` in `eclipse/text_detect/__init__.py`, which takes a
  `Frame` and returns `Detection`s.

## What still belongs in the pipeline

Capture, a minimum 1-second delay, F8 blackout, buffer clearing, and OBS output.
Never release an unscanned frame: black out or hold the last safe frame if OCR
fails, hangs, or misses its deadline. Keep confirmed covers on buffered frames.
Logs must contain categories/timestamps only, never OCR text or screenshots.

## Current limits

Only rows that changed are read again, and text read earlier keeps being reported
until its pixels change. Vertical scrolling moves known text instead of reading it
again, and rows coming onto the screen stay black ("unread") until they can be read.
Video and big window changes still need full scans and will be slow.
OCR can miss secrets; regex rules do not recognize every token format.
Phone rules mostly target North America. Bare 6-8 digit numbers can be flagged as
2FA; 4-5 digit codes are only caught next to a word like "code", so a bare 4-digit
code is missed. Box widths are estimated from character counts.

Tests use fake OCR. Real OCR accuracy and latency still need laptop testing.
Use fake secrets only. The pinned OCR package uses local inference and bundled
models; dependency installation requires internet. Running detection.py by itself
does nothing: import its functions from another script.
