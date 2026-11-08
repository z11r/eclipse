# Webcam OCR experiment

Built on main b2866b1, without the shipping-label or generic-document model branches.
This change adds webcam OCR selection and an optional smaller PIP scan. Default
camera OCR remains RapidOCR and default scan sizing preserves the existing detail.

## Try native Apple OCR first

Run on macOS from the project root with the normal environment active:

```bash
python -m eclipse.main --webcam 0 --webcam-ocr apple
```

For the screen and camera together:

```bash
python -m eclipse.main --ocr apple --with-webcam 0 --webcam-ocr apple
```

`--ocr` selects the screen reader. `--webcam-ocr` independently selects the camera
reader, also for recorded clips tagged `--clip-source webcam`. Apple camera OCR
uses one dedicated worker process, isolated even when screen Apple OCR runs in
the main process (`--ocr-workers 0`). Camera-only and PIP detectors share this
worker, with serialized, individually paired crop jobs. Rules/configuration and
camera noise filtering stay with the ordered per-source detectors. There is no
extra OCR retry and no full-frame camera read-ahead that would react to all sensor
noise. Worker results use normal camera pixel coordinates (pixel_scale=1).

The worker protocol, native pixel buffer and shutdown behavior reuse the existing
OcrPool implementation. It gets closed on normal shutdown and startup failures.
Read errors propagate into the existing held-frame/blackout behavior. Camera
frames are not newly saved or uploaded, and no raw OCR text is logged.

Unsupported platforms reject Apple selection before opening the camera. A failed
worker does not silently switch engines. Return to the original behavior with:

```bash
python -m eclipse.main --webcam 0 --webcam-ocr rapid --webcam-scan capture
```

## Optional PIP scan-size experiment

```bash
python -m eclipse.main --ocr apple --with-webcam 0 --webcam-ocr apple --webcam-scan output
```

The existing layout bases camera OCR size on the captured screen. Retina captures
can be much larger than the 1920x1080 virtual-camera output. `output` applies the
same aspect-preserving output scale before choosing a scan at twice the final
inset size, capped by the real camera resolution. Inset placement is unchanged;
only its scan resolution changes, and cover coordinates use the existing mapping.
The camera-only view still uses the full-size camera detector.

Example: 6016x3384 capture, 1280x720 camera and a quarter-width inset:

| Setting | Camera OCR input | Pixels |
| --- | --- | --- |
| `capture` (default, existing behavior) | 1280x720 | 921,600 |
| `output` (experimental) | 960x540 | 518,400 |

**This is opt-in because smaller input can miss text.** One cloud RapidOCR test
used nine generated moving fake-key images from eval.webcam_cost.scenes, with the
first as warmup. At capture size, median/p95 times were 94.5/104.0 ms and the key
was explicitly recognized in 6/8 measured frames. At output size they were
60.5/97.7 ms and 2/8 frames. That is not adequate evidence to change the default.
The original size also missed two frames, so neither result establishes reliable
protection. This was a tiny synthetic recognition test, not a real-world accuracy
dataset or a measurement of OBS output. There is no measured Apple speed claim
from the Linux cloud environment.

## Local test procedure

1. Compare RapidOCR and Apple on the same fake webcam scene at `capture` sizing.
   Try talking/head motion and fake email, phone, password, card and code text on
   paper or another screen. Check recognition as well as visible smoothness.
2. Test camera-only and PIP views, blackout/resume, keyword/category changes and
   normal shutdown. Closing the page must not stop protection.
3. Inspect eclipse-run.log for scan p95/max, fresh output, camera scans/waits and
   repeats. Native camera reads add `camera-worker jobs/failed/work-ms` numbers.
4. Try `--webcam-scan output` only as a separate experiment; compare small-text
   misses before using it. A 60 FPS delivery rate made of repeated frames is not
   smooth fresh motion.

The existing synthetic webcam cost helper accepts both options:

```bash
python -m eval.webcam_cost --webcam-ocr rapid --screen 6016x3384 --webcam-scan capture
python -m eval.webcam_cost --webcam-ocr apple --screen 6016x3384 --webcam-scan capture
```

It opens the camera briefly, creates synthetic motion from one image in memory,
and prints costs for still/talking/leaning/fake-card scenes. Its rate estimates
exclude competition for CPU and waits for stale camera imagery; validate the live
OBS path separately. No new model download or Python dependency is required.

## Verification

Tests cover crop/result isolation, bounded camera submissions, failure propagation,
reader cleanup during startup and shutdown, main's camera selection, sharing one
native reader across views, default/candidate sizing, letterboxing, and existing
PIP/compositor/panic behavior. Native Vision and real camera/OBS checks require a
Mac. The native-only tests include synthetic email recognition and the text rules
on fake email/phone/password/code/card data; these are skipped in the cloud.

The existing panel-view API test lacks authorization and fails on the base branch;
it was observed while checking views and left unchanged. Shipping-label detection
and document models are not enabled or modified by this work.
