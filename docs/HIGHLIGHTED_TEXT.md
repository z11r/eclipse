# Styled text recognition with Apple OCR

Apple OCR now keeps its fast pass and adds accurate, contrast-normalized reads
of regions containing strongly colored pixels. This includes candidate regions
where the fast pass returned no text. Original fast results remain available.

Candidate selection uses pixel saturation and brightness, not OCR confidence.
Nearby colored pixels are grouped, with surrounding context for recognition.
At most eight crop requests are used; larger candidate sets become one enclosing
crop so candidates are not silently dropped. The contrast pass selects the color
channel with greatest variation, normalizes it and favors dark text on light.
Crop dimensions remain unchanged, and boxes are translated to the input's pixel
coordinates. A failed retry fails the scan rather than returning a clean result.
Both direct Apple reads and shared-memory OCR workers use the same path.

This is a candidate improvement, not proof that all styled text is recognized.
Saturation thresholds can miss faint colors; colored backgrounds can trigger large
regions, and accurate OCR adds latency. Underlines are retained. Native macOS
accuracy and latency cannot be established by tests running on Linux.

## Local validation

Run from the repository root with the environment active:

```bash
python -m pytest -q tests/test_styled_text.py tests/test_apple_ocr.py
python -m eclipse.main --ocr apple
```

Use the current authorized panel link. Show the same fake email in Notes as plain
white text, amber text, an underlined link, and selected text. Test multiple font
sizes on dark and light backgrounds. Check the full email, including punctuation
and its last characters, is covered in the actual camera output. Repeat after
scrolling, editing and moving the window. Compare OCR times and visible motion
against the previous branch revision under the same workload. Check existing
fake keys, codes, cards, phone numbers and keywords remain covered, and verify
panic/resume. Do not use real private information in screenshots or test logs.

The macOS-only styled email tests exercise native Vision, rules, and cover
placement. Non-macOS tests check region selection, contrast, geometry, retention
of fast results, and propagation of retry failures; they do not prove OCR accuracy.

## Ordinary text regression

OCR word groups are also tried with spaces removed to recover split secrets.
That synthetic no-space form no longer runs the broad 60-character generic
API-key heuristic: it could classify a normal sentence as one long token.
The heuristic still runs on original OCR observations and space-preserving
groups. Structured key formats and split emails still run on both forms.
This intentionally trades detection of an unlabelled generic token split across
multiple OCR words for fewer prose false positives; known key formats remain
covered. Restart Eclipse after updating to clear previously retained covers.
