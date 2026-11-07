# Eclipse — natHacks 2026 Spec

Nov 3, 2026 · Max

> Copy of the live spec doc as of Nov 5, 2026. The live doc is the source of truth: https://claude.ai/artifact/Fdu2MP9u9S7DAzQ7qF5V7z

## Overview

Eclipse is a desktop app that sits between your screen and webcam and OBS. It hides anything that leaks your secrets or shows where you live before a single frame goes out. We are entering the Standard Division, Defense & Security theme.

**Pitch line:** Other tools catch your API key. Eclipse catches the things that let people find your house.

**Problem:** Streamers and anyone sharing a screen leak 2FA codes, emails, API keys and location clues, such as a street sign through the window or a package label on the desk. Once a frame is broadcast, it can be clipped and archived. There is no undo.

**Users:**

- Primary: small and mid-size Twitch and YouTube streamers, VTubers and people who code on stream
- IRL streamers who broadcast from a phone while out in public (basic support this weekend)
- Secondary: anyone sharing their screen in meetings, classes or demos

## Goals and metrics

The goal is a live, end-to-end demo into OBS that leaks zero planted secrets on our test clips, with numbers to prove it.

**Goals**

- Screen and webcam pass through Eclipse into OBS as a virtual camera
- The text layer hides planted secrets with no leaked frames on the test set
- The visual layer flags location clues on the webcam and screen within a couple of seconds
- Basic IRL mode keeps street signs, house numbers and plates blurred in a moving walking clip
- A measured leak rate and latency we can show on a slide

**Metrics** (proposed targets; we tune them Saturday once the harness runs)

| Metric | How we measure it | Target |
| --- | --- | --- |
| Text leak rate | Frames in the test clips where a planted secret is readable in the output | 0 |
| Added latency, text layer | p95 time from capture to output, per frame | Under 150 ms |
| Location-clue recall | Planted clues caught in the test clips | Most of them; set Saturday |
| IRL clue coverage | Share of frames where a clue in the walking clips is readable but not blurred | Close to 0; set Sunday |
| False blur rate | Blurred areas with nothing sensitive in them | Low enough to stream with |
| Time to clue blur | Seconds from a clue appearing to it being blurred | About 2 s or less (IRL: before it airs, thanks to the delay) |

## Features

The MVP is six desktop features. Basic IRL mode comes next, then the stretch goals; each starts only once the step before it runs end to end.

**MVP (must have by Sunday 6 PM)**

- [ ] **Pass-through:** screen capture and webcam go out as a virtual camera that OBS picks up
- [ ] **Text layer:** OCR plus rules for 2FA codes, emails, phone numbers, API keys and tokens, card numbers (Luhn check) and a custom keyword list (real name, street, username)
- [ ] **Fail closed:** a frame that hasn't been scanned yet never goes out unblurred; we hold the last safe frame or blur the changed area
- [ ] **Visual layer:** every 1–2 s, a frame goes to a vision model that returns boxes for location clues: street signs, house numbers, licence plates, mail or package labels, views through windows, open maps
- [ ] **Panic hotkey:** one key blacks out the whole output
- [ ] **Control panel:** toggles per category, a keyword list and a live log of what was caught and when

**Basic IRL mode (after the MVP, target Sunday 1 PM)**

IRL mode runs the visual layer on a moving phone camera instead of a desk webcam.

- [ ] **Feed input:** capture reads a recorded walking clip or a live SRT or RTMP feed, the kind a phone streaming app sends to OBS
- [ ] **Clue detection:** the visual layer looks for street signs, house numbers, licence plates and storefront names
- [ ] **Tracking:** an OpenCV tracker keeps each blur on its clue between vision-model calls as the camera moves
- [ ] **Delay buffer:** output is held a fixed 2–3 s so a clue is caught before it airs
- [ ] **Generous blur:** boxes are padded and held longer than on desktop, since a missed frame outdoors costs more than an extra blur

Out of scope this weekend: a phone app, a cloud relay deployment, and judging which outdoor clues actually matter (we blur every one found).

**Stretch goals (in priority order)**

1. **Room scan before going live:** sweep the webcam around the room and get a dox-risk report listing each clue found
2. **Window rules:** always blur chosen apps, such as Discord DMs or a password manager
3. **VOD scan:** run a past recording through the same pipeline and list the timestamps that leaked

## Architecture

Eclipse runs two detection layers side by side. A fast text layer checks every frame. A slower visual layer checks sampled frames for location clues.

```text
Screen ──┐                    ┌─> Text layer (OCR + secret rules, every frame, inline) ──┐
Webcam ──┼─> Capture ─────────┤                                                          ├─> Compositor ─> Output ─> OBS
IRL feed ┘  (frame stream     └─> Visual layer (model boxes + tracker, every 1–2 s, async)┘  (applies blur,  (virtual camera)
             per source)                                                                       fails closed)
                                                                                                   ↕
Eval harness: test clips replace live input                                                  Control panel (toggles and log)
```

No frame reaches OBS until the text layer has scanned it; if the scan falls behind, the compositor sends the last safe frame. Visual-layer boxes stay blurred until the clue has been gone for its `ttl_ms`.

## Modules and interfaces

Eight modules talk only through two shared types, `Frame` and `Detection`, defined in `types.py`. That shared contract keeps parallel work from drifting apart.

| Module | Input → output | Key notes |
| --- | --- | --- |
| `capture` | Screen, webcam, IRL clip or live SRT/RTMP feed → `Frame` stream | `mss` for screen; OpenCV (with ffmpeg) for webcam, clips and feeds; tag each frame with time and source |
| `text_detect` | `Frame` → `list[Detection]` | OCR only the areas that changed since the last frame; rules in `rules.yaml` |
| `visual_detect` | Sampled `Frame` → `list[Detection]` (async) | Every 1–2 s; the vision model returns JSON boxes; never blocks the frame loop |
| `track` | Visual detections + new frames → moved `Detection` boxes | IRL only: an OpenCV tracker (CSRT or KCF) follows each clue between model calls; if it loses one, it grows the box and holds it for `ttl_ms` |
| `compositor` | `Frame` + active detections → redacted frame | Applies blur; enforces fail closed; keeps each blur for its `ttl_ms` so it doesn't flicker; holds IRL output in the delay buffer |
| `output` | Redacted frame → virtual camera | `pyvirtualcam`; OBS adds it as a Video Capture Device |
| `panel` | Config and log ↔ user | Local web page: category toggles, keyword list, catch log, panic button, IRL mode switch |
| `eval` | Test clips + ground-truth JSON → report | Leak rate, p95 latency, recall, false blur rate, IRL coverage; one command, run by every agent |

## Tech stack and repo layout

The whole build is Python 3.11 and runs on both Windows and macOS, so every teammate and every agent works in one language. Every OS difference lives in `platform.py`; no other module checks which OS it is on.

| Layer | Choice | Fallback |
| --- | --- | --- |
| Screen capture | `mss` (both OSes) | Downscale frames first (Mac Retina captures are 2x size); `dxcam` on Windows if `mss` is too slow |
| Webcam | `opencv-python` | — |
| OCR | RapidOCR (ONNX, runs locally on both OSes) | Built-in OS OCR: `winocr` on Windows, `ocrmac` on macOS |
| Secret rules | Our own regex set, inspired by gitleaks patterns (credited in README), plus Luhn checks | — |
| Vision model | A hosted vision API that returns JSON boxes (pick Friday by speed and credits) | A small local detector for plates and signs, or a coarse blur of the whole webcam tile |
| Panic hotkey | `pynput` (not `keyboard`, which needs admin rights on macOS) | The panic button in the panel |
| Output | `pyvirtualcam` with OBS Virtual Camera | Show frames in an OpenCV window and add it in OBS with Window Capture, if the Mac virtual camera fails |
| Panel | FastAPI and one HTML page | — |
| Tests and eval | `pytest` and our `eval` command | — |

**macOS setup, once per laptop:** in System Settings → Privacy & Security, allow Screen Recording, Camera and Accessibility for whatever runs Python (Terminal, iTerm or VS Code). Install OBS 30 or later, start and stop its virtual camera once, and approve the camera extension when macOS asks.

```text
eclipse/
  README.md          # required by natHacks: what it is, how to run, credits
  eclipse/
    types.py         # Frame, Detection (hand-written first)
    platform.py      # the only OS checks: OCR fallback, hotkey, permission checks
    capture/  text_detect/  visual_detect/  track/  compositor/  output/  panel/
    rules.yaml       # text patterns and keyword list
    main.py          # wires the pipeline
  eval/
    clips/           # test recordings with planted fake secrets and clues
    labels/          # ground truth per clip
    run_eval.py
  tests/
```

## Timeline

**64-hour plan** (5 phases, each ends at a gate; sleep in shifts):

| Phase | When | Work | Gate to pass |
| --- | --- | --- | --- |
| Skeleton | Fri 7 PM–2 AM | Contract + eval; frames into OBS | Frames in OBS on every laptop |
| Text layer | Sat 9 AM–6 PM | OCR + secret rules, tuned on clips | Text leaks = 0 on test clips |
| Visual layer | Sat 6 PM–Sun 1 PM | Vision model boxes on webcam, screen, IRL | A clue blurs live, end to end |
| Polish | Sun 1–6 PM | Panel + hotkey; merge, tune, test | Feature freeze: no new stretch goals |
| Pitch | Sun 6 PM–Mon 11 AM | Rehearse demo; write-ups | Submit, Mon 11 AM |

A phase doesn't start until the gate before it passes. If the visual layer misses its gate at Sunday 1 PM, ship the coarse fallback (blur the whole webcam tile while a clue is present) and move to Polish.

## Demo and rules

**5-minute demo script.** A split screen runs the whole time: the raw screen on the left, what viewers see on the right.

| Time | Beat |
| --- | --- |
| 0:00 | Hook: a short pre-recorded clip of a "stream" leaking an address, then "this is what Eclipse catches" |
| 0:30 | The problem and who has it |
| 1:00 | Live: a fake 2FA notification pops up, then an `.env` file opens, then an email inbox; all blurred on the right |
| 2:00 | Live: turn the webcam to the props, a package with a fake label and a printed street sign; both blurred |
| 2:30 | IRL: a walking clip runs through the same pipeline; the street sign and house number stay blurred as the camera moves |
| 3:00 | Panic hotkey, then the eval numbers: leak rate, latency, recall |
| 3:45 | How it differs from Censr and Sensitive Guard, and what's next: OBS plugin, room scan, Eclipse on the relay servers IRL streams already pass through |
| 4:30 | Close and the pitch line |

**Rules we follow** (from the Hacker Handbook)

- All code is written between Nov 6, 7 PM and Nov 9, 11 AM. Planning before that is allowed.
- Submission by Nov 9, 11 AM: summary (250 words max), about the project (600 words max), repo with README, 5-minute presentation and demo.
