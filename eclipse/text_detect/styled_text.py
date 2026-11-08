"""Pixel-only regions for a second OCR pass; never stores recognized text."""
import cv2
import numpy as np


def colour_regions(pixels):
    """Group saturated pixels with surrounding context for styled text recognition.

    These are candidate regions, not proof that text exists. If many candidates
    appear, one enclosing crop bounds request count without silently dropping any.
    """
    bgr = pixels[:, :, :3]
    high, low = bgr.max(axis=2), bgr.min(axis=2)
    mask = ((high.astype(np.int16) - low > 40) & (high > 80)).astype(np.uint8)
    if not mask.any():
        return []
    joined = cv2.dilate(mask, np.ones((9, 25), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    height, width = mask.shape
    regions = []
    for x, y, w, h, area in stats[1:count]:
        if np.count_nonzero(mask[y:y+h, x:x+w]) < 8:
            continue
        regions.append((max(0, x-24), max(0, y-16), min(width, x+w+24), min(height, y+h+16)))
    if len(regions) > 8:
        regions = [(min(r[0] for r in regions), min(r[1] for r in regions),
                    max(r[2] for r in regions), max(r[3] for r in regions))]
    return regions


def contrast_bgra(pixels):
    """Preserve geometry, choose the channel with most contrast, normalize polarity."""
    bgr = pixels[:, :, :3]
    channel = max(cv2.split(bgr), key=lambda part: float(part.std()))
    gray = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    if np.median(gray) < 128:
        gray = 255 - gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
