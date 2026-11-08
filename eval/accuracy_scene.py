"""Detection accuracy on generated screens, through real OCR, judged on the final pixels.

Each screen holds fake secrets and harmless look-alikes in a known place, drawn in a
theme: dark, light, selected, colored, at normal and small sizes, in a monospace and a
proportional font. Every glyph of every value is known, so what the output still shows is
counted directly, and so is harmless text a cover hid.

    python -m eval.accuracy_scene            # Apple OCR, every theme, JSON per screen
    python -m eval.accuracy_scene --summary  # one line per case: misses, false covers

Reported apart, as they mean different things:
- missed: a secret glyph still showing.
- false: a harmless line with any glyph covered.
- unneeded: black pixels away from every secret and its padding.
- uncertain: covers from a shape whose check failed, like a card with a misread digit.

Every value here is fake. Nothing is written but numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.types import Frame

FONTS = {
    "menlo": "/System/Library/Fonts/Menlo.ttc",
    "helvetica": "/System/Library/Fonts/Helvetica.ttc",
    "sf": "/System/Library/Fonts/SFNS.ttf",  # the system font, as Notes uses
}
SIZES = {"normal": 26, "small": 20}  # pixels on a Retina screen: 13 pt and 10 pt
# (background, text, selection behind the line or None), BGR order is handled when drawn.
THEMES = {
    "dark": ((30, 30, 34), (225, 225, 225), None),
    "light": ((255, 255, 255), (20, 20, 20), None),
    "selected": ((255, 255, 255), (20, 20, 20), (179, 215, 255)),
    "selected_dark": ((30, 30, 34), (240, 240, 240), (38, 79, 120)),
    "colored": ((30, 30, 34), (206, 145, 90), None),
}
WIDTH, LINE, LEFT, TOP = 2400, 64, 80, 40
CONTRAST = 40  # a pixel this far from the background is part of a glyph

# (case, label drawn before the value, value that must be hidden). All fake.
SECRETS = [
    ("spaced_password", "password = ", "hunter2"),
    ("short_password", "pwd=", "Abc1"),
    ("quoted_passphrase", 'password: "', "correct horse battery staple"),
    ("env_password", "DB_PASSWORD=", "s3cr3tpass"),
    ("labelled_intl_phone", "Phone: ", "+44 20 7946 0958"),
    ("bare_intl_phone", "", "+81 3-1234-5678"),
    ("corrupted_card", "Card: ", "4242 4242 4242 4243"),
    ("labelled_code", "verification code ", "482913"),
    ("email", "contact: ", "alex.fake@example.com"),
    # Fields after the value must stay readable: (case, label, value, text after it).
    ("json_password", '{"password": "', "s3cr3t pass", '", "user": "bob"}'),
    ("pwd_then_user", "pwd=", "1234", ";user=bob"),
]
# Text that shares one OCR word with a secret, so a word-level cover must hide it too.
SAME_WORD = {"pwd_then_user_after"}
# A label on one line and its value on the next, as in forms and wrapped text:
# (case, label line, value line, whether the value is secret).
STACKED = [
    ("password_below", "Password:", "hunter2", True),
    ("sha_below", "Commit SHA:", "0123456789abcdef0123456789abcdef01234567", False),
]
STACKED_STEP = 1.45  # line height, in text sizes, of a value under its label
# Harmless lines that must stay fully visible.
HARMLESS = [
    ("order_number", "Order number: 482913"),
    ("download_count", "Downloads: 12345678"),
    ("commit_sha", "Commit SHA: 0123456789abcdef0123456789abcdef01234567"),
    ("credential_reference", 'password = os.getenv("DB_PASSWORD")'),
    ("template_reference", "password=${DB_PASSWORD}"),
    ("prose", "The meeting is on Friday at 10:30."),
    ("test_summary", "Pass: 3 Fail: 0"),
]


@dataclass(frozen=True)
class Planted:
    case: str
    secret: bool
    mask: np.ndarray  # full-screen, True on the glyphs that matter


def render(font_name: str, size_name: str, theme: str) -> tuple[np.ndarray, list[Planted]]:
    """One screen: each secret line, each harmless line, then labels over their values.

    Only the value's glyphs count as the secret. Text after a value (", "user": "bob"}"),
    every harmless line and every label line must stay visible.
    """
    font = ImageFont.truetype(FONTS[font_name], SIZES[size_name])
    background, text, selection = THEMES[theme]
    # (case, y, text before, value, text after, secret)
    rows = []
    y = TOP
    for case, label, value, *after in SECRETS:
        closing = after[0] if after else ('"' if label.endswith('"') else "")
        rows.append((case, y, label, value, closing, True))
        y += LINE
    for case, value in HARMLESS:
        rows.append((case, y, "", value, "", False))
        y += LINE
    for case, label, value, secret in STACKED:
        rows.append((f"{case}_label", y, "", label, "", False))
        rows.append((case, y + int(STACKED_STEP * SIZES[size_name]), "", value, "", secret))
        y += 2 * LINE
    height = y + TOP
    canvas = Image.new("RGB", (WIDTH, height), background)
    draw = ImageDraw.Draw(canvas)
    planted = []
    for case, y, label, value, closing, secret in rows:
        whole = label + value + closing
        if selection is not None:
            draw.rectangle(
                (LEFT - 4, y - 6, LEFT + int(font.getlength(whole)) + 4, y + SIZES[size_name] + 8),
                fill=selection,
            )
        draw.text((LEFT, y), whole, font=font, fill=text)
        # The value's own glyphs, drawn alone where they sit in the line.
        layer = Image.new("L", (WIDTH, height), 0)
        x = LEFT + font.getlength(label)
        ImageDraw.Draw(layer).text((x, y), value, font=font, fill=255)
        planted.append(Planted(case, secret, np.array(layer) > 96))
        if secret and closing.strip('"'):
            # What follows the value is ordinary text and must stay readable.
            tail = Image.new("L", (WIDTH, height), 0)
            x_after = LEFT + font.getlength(label + value + closing[:2])
            ImageDraw.Draw(tail).text((x_after, y), closing[2:], font=font, fill=255)
            planted.append(Planted(f"{case}_after", False, np.array(tail) > 96))
    image = np.array(canvas)[:, :, ::-1].copy()
    return image, planted


CHAR = max(SIZES.values())  # a character's height, for how far OCR boxes overrun glyphs


def judge(image: np.ndarray, output: np.ndarray, planted: list[Planted], padding: int) -> dict:
    """Secret glyphs showing, and harmless glyphs covered, split by why.

    Harmless glyphs a secret's padded box reaches are padding: covers grow by the
    compositor's padding on every side, and OCR's box by up to a character beside the
    value, so a label right above or beside a value loses its edge. Text in the same OCR word as a secret is word-level: word boxes can't be split.
    Anything else covered is false: a rule matched text it shouldn't have.
    """
    showing = (output == image).all(axis=2)
    black = (output == 0).all(axis=2) & ~(image == 0).all(axis=2)
    allowed = np.zeros(black.shape, bool)
    padded = np.zeros(black.shape, bool)
    cases = {}
    for item in planted:
        if not item.secret:
            continue
        ys, xs = np.nonzero(item.mask)
        reach = padding + LINE  # a cover may run a line's height past its glyphs
        allowed[
            max(ys.min() - reach, 0) : ys.max() + reach, max(xs.min() - reach, 0) : xs.max() + reach
        ] = True
        # OCR's word box runs past the glyphs, by a quote or comma beside the value.
        side, edge = padding + int(0.6 * CHAR), padding + int(0.25 * CHAR)
        padded[
            max(ys.min() - edge, 0) : ys.max() + edge + 1,
            max(xs.min() - side, 0) : xs.max() + side + 1,
        ] = True
        cases[item.case] = {
            "secret": True, "glyphs": int(item.mask.sum()),
            "showing": int((item.mask & showing).sum()),
        }
    for item in planted:
        if item.secret:
            continue
        covered = item.mask & ~showing
        cases[item.case] = {
            "secret": False, "glyphs": int(item.mask.sum()), "covered": int(covered.sum()),
            "padding": int((covered & padded).sum()),
            "word_level": item.case in SAME_WORD,
        }
    false = sorted(
        c for c, r in cases.items()
        if not r["secret"] and not r["word_level"] and r["covered"] > r["padding"]
    )
    return {
        "cases": cases,
        "missed": sorted(c for c, r in cases.items() if r["secret"] and r["showing"]),
        "false": false,
        "padding_covered_px": sum(r["padding"] for r in cases.values() if not r["secret"]),
        "word_level_covered_px": sum(
            r["covered"] for r in cases.values() if not r["secret"] and r["word_level"]
        ),
        "unneeded_black_px": int((black & ~allowed).sum()),
    }


def run_screen(
    font_name: str, size_name: str, theme: str, reader=None, rescue: bool = False
) -> dict:
    from eclipse.text_detect import TextDetector

    if reader is None:
        from eclipse.platform import AppleVisionReader

        reader = AppleVisionReader()
    image, planted = render(font_name, size_name, theme)
    image.flags.writeable = False
    careful = None
    if rescue:
        from eclipse.platform import AppleVisionReader

        careful = AppleVisionReader(accurate=True)
    detect = TextDetector(reader=reader, pixel_scale=2, rescue=careful)
    padding = PADDING * 2  # as the live pipeline pads Apple OCR covers
    output = Compositor(padding=padding).render(Frame(image, 0.0, "screen", 0), detect).image
    result = judge(image, output, planted, padding)
    result["uncertain"] = int(detect.stats.get("uncertain", 0))
    result["rescues"] = int(detect.stats.get("rescues", 0))
    result["rescue_ms"] = round(detect.stats.get("rescue_seconds", 0.0) * 1000, 1)
    result.update(font=font_name, size=size_name, theme=theme)
    return result


def screens():
    for font_name in FONTS:
        for size_name in SIZES:
            for theme in THEMES:
                yield font_name, size_name, theme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true", help="totals per case")
    parser.add_argument(
        "--rescue", action="store_true", help="reread doubtful lines with accurate OCR"
    )
    args = parser.parse_args(argv)
    results = [run_screen(*screen, rescue=args.rescue) for screen in screens()]
    if args.summary:
        missed, false = {}, {}
        for result in results:
            tag = f"{result['font']}/{result['size']}/{result['theme']}"
            for case in result["missed"]:
                missed.setdefault(case, []).append(tag)
            for case in result["false"]:
                false.setdefault(case, []).append(tag)
        print(json.dumps({
            "screens": len(results),
            "missed": {case: len(tags) for case, tags in sorted(missed.items())},
            "false": {case: len(tags) for case, tags in sorted(false.items())},
            "missed_on": missed,
            "showing_glyph_px": sum(
                r["showing"] for res in results for r in res["cases"].values() if r["secret"]
            ),
            "false_covered_px": sum(
                r["covered"] - r["padding"] for res in results for r in res["cases"].values()
                if not r["secret"] and not r["word_level"]
            ),
            "padding_covered_px": sum(r["padding_covered_px"] for r in results),
            "word_level_covered_px": sum(r["word_level_covered_px"] for r in results),
            "unneeded_black_px": sum(r["unneeded_black_px"] for r in results),
            "uncertain_covers": sum(r["uncertain"] for r in results),
            "rescues": sum(r["rescues"] for r in results),
            "rescue_ms": round(sum(r["rescue_ms"] for r in results), 1),
        }))
    else:
        for result in results:
            print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
