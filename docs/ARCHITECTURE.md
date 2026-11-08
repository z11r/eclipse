# Architecture

This describes what Eclipse does today. Each module's docstring has the details.

## Pipeline

```text
capture ──> waiting to scan ──> text layer ──> compositor ──> delay buffer ──> output gate ──> virtual camera ──> OBS
                  │                  ▲                                               ▲
                  └──> OCR workers ──┘ (Apple OCR only)                   panel and panic key
```

Every frame is scanned by the text layer and drawn by the compositor before the delay buffer hands it to the output. Output never receives a raw frame, so there is no path for an unscanned frame to go out. If a scan fails or runs out of time, the last safe frame stays up.

## Modules

| Module | What it does |
| --- | --- |
| `capture/` | On macOS, `sck.py` uses ScreenCaptureKit, which hands over each new picture and sends nothing while the screen is still. If it can't start or stops mid-run, `preferred.py` falls back to polling with `mss` (`screen.py`). `webcam.py` reads a camera through OpenCV (`--webcam INDEX`) and always uses RapidOCR. Pick the real camera, not OBS Virtual Camera, or the output loops back in. If the camera stops sending frames, the run stops. `clip.py` reads a recorded clip, and `pattern.py` draws a moving test pattern with a fake secret. |
| `text_detect/` | Reads text with OCR and finds secrets in it. See [Text layer](#text-layer). |
| `compositor/compositor.py` | Draws a solid black box over each detection, padded by at least 10 px. A `held` detection is filled from the last safe picture it drew instead, so a pane not yet checked shows what already went out there, not black. If the scan failed, it returns the last safe frame instead. It keeps no boxes between frames: the text layer reports everything to cover on every frame. |
| `compositor/delay.py` | Runs capture, scanning and output side by side. Output sends each frame `--delay` seconds (1.0 by default) after capture, so a slow scan uses up some of the delay instead of freezing the picture. A frame that can't be scanned in time is dropped. With `--look-ahead` (0.3 s by default, at most 30% of the delay), a scanned frame waits that long before the compositor draws it, so the text layer can use the frames scanned after it. |
| `compositor/gate.py` | The last check before a picture leaves. Panic, resume and settings changes land here, so they take effect at the output however far behind the scanner is. Panic blacks out everything, including frames already in the buffer. After resume, nothing captured before it is shown. A setting that hides more blocks anything scanned under the old setting. On shutdown the gate closes and the last frame sent is black. |
| `output/virtual_cam.py` | Fits each frame to the camera size and sends it to OBS Virtual Camera on a fixed 60 Hz grid. Each frame is prepared before its tick and stamped with the tick, not with when it left. On macOS `obs_mac_camera.py` writes UYVY frames straight to OBS's camera extension so it can set that stamp. Elsewhere `pyvirtualcam` sends them and stamps them as they leave. OBS paces by the stamps only when the Video Capture Device has Use Buffering ticked. Without it, OBS shows whatever arrived last at each of its own ticks, and frames landing near a tick are doubled or skipped. |
| `panel/` | A local page at http://127.0.0.1:8000 with category toggles, a keyword list, a catch log, and panic and resume buttons. Keywords are saved in `~/.eclipse/config.json`. The catch log records categories and times only, never screen text. `hotkey.py` adds a global panic key, Control+Option+B on a Mac. |
| `types.py` | `Frame`, `Detection` and `Rendered`, the shared contract between modules. |
| `platform.py` | The only module that checks which OS it is on: permission checks, ScreenCaptureKit and Retina settings, the camera driver (Media Foundation on Windows), peak memory, and Apple's OCR. |
| `main.py` | Wires the pipeline and writes numbers to `eclipse-run.log` every 2 seconds. Never screen text. |

## Text layer

`Detector` in `text_detect/detection.py` handles each frame in order:

1. It finds the rows that changed and reads only those. Past 60% of the screen, it reads the whole frame.
2. When an area scrolls vertically, it moves the text it already knows instead of reading it again. Rows scrolling into view stay black until they've been read, but only where they show something: background-coloured pixels can't hide anything, so the black covers each incoming line's text, not the pane's width. Incoming rows are read once 96 have come in, or as soon as a whole line has come in past the one the edge cuts (at most every other frame). The line the edge cuts through is measured from the pixels: it runs from the edge to the first blank rows, and is nothing when the edge falls between lines. It stays unread until it is whole.
   A window dragged straight up or down looks like a scroll at first. What it uncovers stays still in the next frame, though, while text scrolling in moves on with the scroll, so from then on it is followed as a window move: nothing is held black, and what it uncovers is read where it shows.
   While a pane scrolls, and after it stops until its window moves, a cover whose text the pane's top or bottom edge cuts through isn't dropped by a read that finds nothing there: no read can see a cut line whole.
   The scrolling area is the pane whose pixels the shift explains, not every pixel that changed. Other things change in the same frames, like OBS's preview of the screen or an audio meter, and would otherwise stretch the area, so the black incoming rows landed on them instead of on the pane's edge. Blocks of pixels the shift explains are joined across blank lines but not across wide gaps to the side. Groups beside the pane join it only if the shift explains them almost exactly. When the whole change gives no shift, each part kept apart by unchanged columns is tried alone.
   Rows scrolling in are only black while that frame waits to be drawn, though. `text_detect/incoming.py` follows each frame's black rows through the frames scanned after it. Every step must reproduce their pixels exactly: unchanged, or moved by a scroll whose shift explains every pixel of the pane. Once a later frame has read them and they're not part of a line its pane's edge cuts, the frame goes out with those rows as they are, plus that later frame's covers moved back by the scroll and kept inside the pane. A later frame at rest also clears rows it shows as read, cut line included, as frames at rest always show it: earlier frames then show the same pixels or fewer, slightly sooner. Rows still unresolved don't go black: the pane is held, showing the last safe picture of it, so a scroll can pause briefly or step instead. This shows nothing OCR found. Rows where the read found text that looks like a secret (an @, five or more digits, or a long token mixing digits or cases) but no rule matched stay black, since OCR may have misread a real one.
   When a window is dragged in any direction, it finds the shift that makes the window's pixels match the previous frame exactly and moves known text and covers by it. Text moves only if every pixel of its new place matches. Anything else, and every pixel the shift doesn't explain, is read again. A cover moved this way gets no extra margin while it waits for a read, so it doesn't swell as text slides under another window's edge.
3. It runs the rules on what it read: email, phone, 2FA code, API key, card number (with a Luhn check) and the user's keywords. Every rule is a regex in `detection.py`.
4. `covers.py` keeps one cover on each secret from its first sighting until it is really gone: its pixels changed and a new read found nothing there, it left the screen, or the same text turned up somewhere else. Time alone never ends a cover. Covers keep a hash of the text, never the text.

There are two OCR engines:

- **RapidOCR** (default) runs locally on both OSes. It returns lines, so a secret's box is estimated from where it sits in the line.
- **Apple Vision** (`--ocr apple`, macOS only) is much faster and gives real word positions, but finds fewer secrets than RapidOCR. It needs full Retina frames: at standard size it misses most small text. Vision reads one area at a time per process, so `read_ahead.py` starts reading changed areas in worker processes (`ocr_worker.py`) as soon as each frame is captured.

`find_emails.py`, `find_phone.py`, `find_cards.py` and `luhn.py` are standalone helpers. The pipeline doesn't call them.

## Not built yet

- IRL feed input
- The visual layer for location clues: `visual_detect/` is empty
- The IRL tracker: `track/` is empty

## Known limits

Eclipse reduces risk. It doesn't guarantee that nothing leaks.

- OCR can miss text, and the rules don't recognize every token format.
- Covers can still fail on a live screen in cases we haven't pinned down yet.
- Video and big window changes need full scans, which take the longest.
- Phone rules mostly match North American numbers.
- A bare 6 to 8 digit number counts as a 2FA code. A 4 or 5 digit code is only caught next to a word like "code", so a bare 4-digit code is missed.

## Testing

| Command | What it checks |
| --- | --- |
| `python -m pytest` | Unit tests |
| `python -m eval.run_eval` | Leaks and latency on the clips in `eval/clips`. Exits 1 if anything leaks. |
| `python -m eval.stress_60` | Hard screen activity at 60 fps through real Apple OCR, counting leaked pixels directly |
| `python -m eval.sustained_motion --seconds 60` | The whole pipeline over time, with failed reads, panic and resume, and new keywords. Uses fake OCR. |
| `python -m eval.scroll_compare` | How covers behave during a scripted scroll |
| `python -m eval.motion_flicker` | Window drags and scrolls through real Apple OCR: secret pixels left showing, and how much cover shapes change around their text from frame to frame. `--look-ahead 18` draws frames as the pipeline's look-ahead does. |
| `python -m eval.capture_bench` | Capture speed alone, per backend |
| `python -m eval.obs_path send`, then `verify` | Which frames really reach an OBS recording. `--plan` runs output paths back to back in one recording. |

Test clips and scenes use fake secrets only.
