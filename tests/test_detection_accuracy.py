"""Detection accuracy: which text counts, the whole value, and what the output still shows.

Three levels, from the rules to the pixels:
A. Text only: categories and complete value spans, against harmless look-alikes.
B. The detector on OCR-shaped words: grouping, thin punctuation, alternative readings.
C. Real Apple OCR on generated screens, judged on the final rendered pixels (macOS only).

Every value here is fake. Keys are joined from pieces so secret scanners skip them.
"""

import sys

import numpy as np
import pytest

from eclipse.compositor.compositor import PADDING, Compositor
from eclipse.text_detect import TextDetector
from eclipse.text_detect.detection import Detector, find_matches
from eclipse.text_detect.rules import UNREAD_MARK, find_spans
from eclipse.types import Frame

# --- A. text only ----------------------------------------------------------------------


def covered(text, category):
    """Positions of text inside some span of this category."""
    return {i for c, a, b in find_matches(text) if c == category for i in range(a, b)}


def assert_value_covered(text, category, value, label=""):
    start = text.index(value, len(label))
    missing = set(range(start, start + len(value))) - covered(text, category)
    assert not missing, f"{text!r}: {category} leaves {''.join(text[i] for i in sorted(missing))!r}"


@pytest.mark.parametrize(
    "text,value",
    [
        ("password=abc", "abc"),
        ("pwd=1234", "1234"),
        ("DB_PASSWORD=s3cr3t", "s3cr3t"),
        ("PASSWD: x", "x"),
        ('password="correct horse battery staple"', "correct horse battery staple"),
        ("password='correct horse battery staple'", "correct horse battery staple"),
        ('password: "correct horse battery staple"', "correct horse battery staple"),
        ("Password: correct horse battery staple", "correct horse battery staple"),
        ('{"password": "s3cr3t pass", "user": "bob"}', "s3cr3t pass"),
        ('password="a\\"b c" user="x"', 'a\\"b c'),
        ("password: str = 'hunter2'", "hunter2"),
        ("SMTP_PASS=hunter2", "hunter2"),
        # Curly quotes, as Notes types them.
        ("{\u201cpassword\u201d: \u201cs3cr3t pass\u201d, \u201cuser\u201d: \u201cbob\u201d}", "s3cr3t pass"),
        ("password=\u2018x y\u2019 user=bob", "x y"),
        # OCR's usual misreads of the colon, and a mark it saw but didn't read.
        ('password" "correct horse"', "correct horse"),
        ("password; hunter2", "hunter2"),
        ("password.. hunter2", "hunter2"),
        ('"password"" "s3cr3t pass" "user": "bob"', "s3cr3t pass"),
        (f'password: "{UNREAD_MARK}correct horse', "correct horse"),
        (f"{{'password': 's3cr3t pass {UNREAD_MARK} user': 'bob'}}", "s3cr3t pass"),
        (f"password {UNREAD_MARK} hunter2", "hunter2"),
    ],
)
def test_explicit_password_values_are_covered_whole(text, value):
    assert_value_covered(text, "api_key", value)


@pytest.mark.parametrize(
    "text,value,neighbour",
    [
        ("pwd=1234;user=bob", "1234", "bob"),
        ("pwd=1234&user=bob", "1234", "bob"),
        ("password: hunter2, username: alex", "hunter2", "alex"),
        ('password="x y" user="bob"', "x y", "bob"),
        ("password: hunter2  # rotate monthly", "hunter2", "rotate"),
    ],
)
def test_a_password_value_stops_at_the_next_field(text, value, neighbour):
    assert_value_covered(text, "api_key", value)
    start = text.rindex(neighbour)
    assert not set(range(start, start + len(neighbour))) & covered(text, "api_key")


@pytest.mark.parametrize(
    "text,label",
    [
        ("password=abc", "password="),
        ("Password: correct horse", "Password: "),
        ("Phone: +44 20 7946 0958", "Phone: "),
        ("API_KEY=" + "abcdef" + "123456", "API_KEY="),
    ],
)
def test_spans_are_the_value_not_its_label(text, label):
    assert not set(range(len(label))) & {i for _, a, b in find_matches(text) for i in range(a, b)}


@pytest.mark.parametrize(
    "text",
    [
        'password = os.getenv("DB_PASSWORD")',
        "password = os.environ['DB_PASSWORD']",
        "password=${DB_PASSWORD}",
        "password=$DB_PASSWORD",
        "password: process.env.DB_PASSWORD",
        "password = settings.DATABASE_PASSWORD",
        "password = get_secret('db')",
        "password = None",
        "def login(user, password: str):",
        'password: "{{ vault_db_password }}"',
        # The same, as small text comes out of fast OCR.
        'password = os.gelenv{"DB PASSWORD"I',
        # A spell-check underline in Notes garbled the call, but not the setting's name.
        'password = R%AoionY("D8_PASSWORD")',
        "password=S{DB_PASSWORD}",
        "password=$IDB_PASSWORDI",
    ],
)
def test_code_references_are_not_passwords(text):
    assert "api_key" not in {c for c, _, _ in find_matches(text)}


@pytest.mark.parametrize(
    "text",
    [
        "Order number: 482913",
        "Downloads: 12345678",
        "Invoice 20261108",
        "Build 4829130",
        "Order 482913 shipped",
        "port = 8080",
        "year = 2026",
    ],
)
def test_ordinary_numbers_are_not_codes(text):
    assert "2fa" not in {c for c, _, _ in find_matches(text)}


@pytest.mark.parametrize(
    "text,value",
    [
        ("Your verification code is 4829", "4829"),
        ("Your code: 482913", "482913"),
        ("Use 482913 to sign in", "482913"),
        ("Enter the one-time passcode 91234", "91234"),
        ("482913", "482913"),  # a code shown alone, as an authenticator does
        ("123 456", "123 456"),
    ],
)
def test_security_codes_are_codes(text, value):
    assert_value_covered(text, "2fa", value)


@pytest.mark.parametrize(
    "text",
    [
        "Commit SHA: 0123456789abcdef0123456789abcdef01234567",
        "commit 0123456789abcdef0123456789abcdef01234567",
        "SHA256: " + "0123456789abcdef" * 4,
        "sha256sum " + "ab" * 32 + "  release.tar.gz",
        "md5: " + "0f" * 16,
        "integrity sha512-" + "Ab1+" * 20,
        "CommitSHA: 0123456789abcdef0123456789abcdef01234567",  # words OCR ran together
        "Commil SHA.' 0123456789abcdef0123456789abcdef01234567",
    ],
)
def test_public_hashes_are_not_keys(text):
    assert "api_key" not in {c for c, _, _ in find_matches(text)}


@pytest.mark.parametrize(
    "text,value",
    [
        ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
        ("CLIENT_SECRET: " + "ab" * 32, "ab" * 32),
        ("api_key sha256 = " + "cd" * 32, "cd" * 32),  # a hash word can't excuse a key field
        ("sk-" + "proj-" + "FAKE1234567890abcdef", "sk-" + "proj-" + "FAKE1234567890abcdef"),
        # An AWS key with letters OCR read in lower case.
        ("key AKIA" + "IOSFODfvN7EXAMPLE", "AKIA" + "IOSFODfvN7EXAMPLE"),
    ],
)
def test_keys_stay_keys(text, value):
    assert_value_covered(text, "api_key", value)


@pytest.mark.parametrize(
    "text,value",
    [
        ("+44 20 7946 0958", "+44 20 7946 0958"),
        ("Phone: +44 20 7946 0958", "+44 20 7946 0958"),
        ("+49 30 901820", "+49 30 901820"),
        ("+81 3-1234-5678", "+81 3-1234-5678"),
        ("+33 1 23 45 67 89", "+33 1 23 45 67 89"),
        ("+1 (780) 555-0142", "+1 (780) 555-0142"),
        ("(780) 555-0142", "(780) 555-0142"),
        ("Tel: 020 7946 0958", "020 7946 0958"),
        ("Mobile 0151 23456789", "0151 23456789"),
        # A number no carrier would issue is still the one on screen.
        ("Phone: +44 0000 000000", "+44 0000 000000"),
    ],
)
def test_phones_are_found_in_any_region_format(text, value):
    assert_value_covered(text, "phone", value)


def test_a_labelled_phone_stops_before_the_next_field():
    text = "Phone: 555 0142 9999 Email"
    assert_value_covered(text, "phone", "555 0142 9999")
    assert not set(range(text.index("Email"), len(text))) & covered(text, "phone")


@pytest.mark.parametrize("text", ["+3 points", "Version 2.0.1", "+12 -3", "Total +45.00"])
def test_short_plus_numbers_are_not_phones(text):
    assert "phone" not in {c for c, _, _ in find_matches(text)}


def card(text):
    return [s for s in find_spans(text) if s.category == "card"]


@pytest.mark.parametrize(
    "text,value",
    [
        ("Card: 4242 4242 4242 4243", "4242 4242 4242 4243"),  # one digit misread
        ("4242 4242 4242 4243", "4242 4242 4242 4243"),  # printed in a card's groups
        ("Credit card number 4242424242424243", "4242424242424243"),
    ],
)
def test_a_card_with_one_misread_digit_stays_covered_as_uncertain(text, value):
    spans = card(text)
    assert spans and all(s.uncertain for s in spans)
    assert_value_covered(text, "card", value)


def test_a_valid_card_is_certain_and_ignores_the_expiry_after_it():
    spans = card("Card: 4242 4242 4242 4242 12/28")
    assert any(not s.uncertain for s in spans)
    assert not any(s.uncertain for s in spans)


@pytest.mark.parametrize(
    "text",
    ["Invoice 4242424242424243", "Order 1234567890123", "1234 5678 9012 3456", "Tracking 9400111202555842"],
)
def test_long_numbers_are_not_all_cards(text):
    assert not card(text)


@pytest.mark.parametrize(
    "text",
    [
        "The meeting is on Friday.",
        "Server runs on port 8080",
        "@property",
        "user@localhost",
        "Pass: 3 Fail: 0",
        "bypass: true",
        "compass: north",
        "Forgot password?",
        "Password must be 8 characters",
        "Change your password",
        "TOKEN_COUNT = 100",
        "max_tokens: 4096",
        "token: abc",
        "Phone settings",
    ],
)
def test_harmless_lookalikes_are_left_alone(text):
    assert not find_matches(text)


# --- B. the detector on OCR-shaped words ---------------------------------------------


class WordReader:
    """Fake OCR returning fixed words, as (text, x1, y1, x2, y2), wherever it's asked."""

    def __init__(self, words):
        self.words = words

    def __call__(self, image):
        return [
            ([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, 0.9)
            for text, x1, y1, x2, y2 in self.words
        ], None


def covers(words, image=None, **options):
    detector = Detector(reader=WordReader(words), **options)
    image = np.full((200, 1600, 3), 255, np.uint8) if image is None else image
    return detector, detector.detect(image)


def hit(boxes, x1, x2):
    return any(x < x2 and x + w > x1 for x, _, w, _, _ in boxes)


def test_a_thin_equals_box_does_not_part_a_password_from_its_label():
    # Vision boxes "=" a few pixels tall: the gap after it was judged against that.
    _, boxes = covers([("password", 10, 100, 120, 124), ("=", 130, 110, 142, 113),
                       ("hunter2", 170, 100, 260, 120)])
    assert hit(boxes, 170, 260)
    assert not hit(boxes, 10, 120)  # the label stays readable


def test_text_in_another_window_on_the_same_row_is_no_value():
    _, boxes = covers([("password:", 10, 100, 120, 124), ("hunter2", 900, 100, 990, 120)])
    assert boxes == []


def test_a_dropped_equals_sign_is_found_in_the_pixels():
    image = np.full((200, 1600, 3), 255, np.uint8)
    image[108:112, 150:170] = 0  # the "=" OCR saw no text in
    words = [("password", 10, 100, 120, 124), ("hunter2", 200, 100, 290, 120)]
    _, boxes = covers(words, image)
    assert hit(boxes, 200, 290) and not hit(boxes, 10, 120)
    _, boxes = covers(words)  # a blank gap: "password" just stands far from the next word
    assert boxes == []


def test_a_long_key_split_into_pieces_is_judged_with_its_label():
    hexes = ["0123456789abcdef", "0123456789abcdef", "01234567"]
    words = [("Commit", 10, 100, 100, 120), ("SHA:", 120, 100, 176, 120)]
    x = 200
    for piece in hexes:
        words.append((piece, x, 100, x + 250, 120))
        x += 254  # Vision's pieces sit a few pixels apart
    _, boxes = covers(words)
    assert boxes == []
    words[0:2] = [("API_KEY", 10, 100, 100, 120), ("=", 110, 108, 120, 112)]
    _, boxes = covers(words)
    assert hit(boxes, 200, x)


def ink_between(image, words):
    """Draw a mark in the gap after each word, as punctuation OCR didn't read."""
    for (_, _, _, x2, _), (_, x1, *_rest) in zip(words, words[1:]):
        if x1 - x2 > 20:
            image[106:114, x2 + 8 : x1 - 8] = 0
    return image


def test_a_value_stops_at_punctuation_ocr_dropped():
    # {"password": "s3cr3t pass", "user": "bob"} with every quote and colon dropped.
    words = [("{", 10, 100, 20, 124), ("password", 50, 100, 160, 124),
             ("s3cr3t", 200, 100, 280, 120), ("pass", 288, 100, 340, 120),
             ("user", 380, 100, 430, 120), ("bob", 470, 100, 510, 120)]
    image = ink_between(np.full((200, 1600, 3), 255, np.uint8), words[1:])
    _, boxes = covers(words, image)
    assert hit(boxes, 200, 280) and hit(boxes, 288, 340)
    assert not hit(boxes, 380, 510) and not hit(boxes, 50, 160)


def test_a_label_line_gives_its_meaning_to_the_line_below():
    hexes = "0123456789abcdef0123456789abcdef01234567"
    _, boxes = covers([("Commit", 10, 100, 90, 120), ("SHA:", 100, 100, 150, 120),
                       (hexes, 10, 132, 500, 152)])
    assert boxes == []  # a commit, not a key, though it wrapped onto its own line
    _, boxes = covers([("Password:", 10, 100, 120, 120), ("hunter2", 10, 132, 100, 152)])
    assert hit(boxes, 10, 100) and all(y >= 125 for _, y, *_ in boxes)
    _, boxes = covers([("Order", 10, 100, 70, 120), ("number:", 80, 100, 170, 120),
                       ("482913", 10, 132, 90, 152)])
    assert boxes == []
    _, boxes = covers([("Your", 10, 100, 60, 120), ("code:", 70, 100, 130, 120),
                       ("482913", 10, 132, 90, 152)])
    assert {box[4] for box in boxes} == {"2fa"}
    # A label far above, or off to the side, labels nothing.
    _, boxes = covers([("Password:", 10, 100, 120, 120), ("hunter2", 10, 190, 100, 210)])
    assert boxes == []


def test_competing_readings_are_never_joined_into_one_token():
    # Two reads of one place, each half a key: joined they'd make one neither had.
    _, boxes = covers([("sk-" + "proj", 10, 100, 120, 120), ("FAKE1234567890abcdef", 30, 100, 280, 120)])
    assert boxes == []


def test_a_public_label_turns_down_the_number_beside_it():
    _, boxes = covers([("Order", 10, 100, 70, 120), ("number:", 80, 100, 170, 120),
                       ("482913", 185, 100, 260, 120)])
    assert boxes == []
    _, boxes = covers([("Your", 10, 100, 60, 120), ("code:", 70, 100, 130, 120),
                       ("482913", 145, 100, 220, 120)])
    assert {box[4] for box in boxes} == {"2fa"} and not hit(boxes, 10, 130)


def test_a_line_box_still_covers_from_the_label_where_words_cant_be_placed():
    # Line-level OCR gives one box: the value's first glyph can't be placed from text alone.
    _, boxes = covers([("password: hunter2", 10, 100, 210, 120)])
    assert boxes and min(x for x, *_ in boxes) <= 10


def test_shared_detections_keep_their_contract():
    detect = TextDetector(reader=WordReader([("pwd=1234", 10, 20, 90, 40)]))
    found = detect(Frame(np.zeros((80, 200, 3), np.uint8), 0.0, "screen", 3))
    assert [(d.category, d.frame_id, d.confidence) for d in found] == [("api_key", 3, 1.0)]


class CountingReader(WordReader):
    def __init__(self, words):
        super().__init__(words)
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return super().__call__(image)


def test_rescue_rereads_only_doubtful_lines_and_adds_what_it_finds():
    fast = [("password.", 10, 100, 120, 124), ("correcl", 135, 100, 220, 120)]
    careful = CountingReader([("password:", 9, 13, 119, 37), ('"correct', 134, 13, 220, 33)])
    detector = Detector(reader=WordReader(fast), rescue=careful)
    image = np.full((200, 1600, 3), 255, np.uint8)
    boxes = detector.detect(image)
    assert careful.calls == 1 and hit(boxes, 135, 220)
    assert detector.detect(image) == boxes and careful.calls == 1  # unchanged: no reread


def test_rescue_never_runs_on_settled_lines_and_failures_keep_the_fast_read():
    class Broken:
        def __call__(self, image):
            raise RuntimeError("careful OCR died")

    detector = Detector(reader=WordReader([("pwd=1234", 10, 100, 100, 120)]), rescue=Broken())
    assert detector.detect(np.full((200, 600, 3), 255, np.uint8))
    assert detector.stats["rescue_failed"] == 0  # pwd=1234 was settled: nothing reread
    detector = Detector(reader=WordReader([("password.", 10, 100, 120, 124)]), rescue=Broken())
    assert detector.detect(np.full((200, 600, 3), 255, np.uint8)) == []
    assert detector.stats["rescue_failed"] == 1


def test_rescue_keeps_to_its_time_budget():
    words = [(f"password{i}.", 10, 30 * i, 120, 30 * i + 20) for i in range(1, 40)]
    detector = Detector(reader=WordReader(words), rescue=WordReader([]), rescue_budget=0.0)
    detector._rescue_allowance = 0.0
    detector.detect(np.full((1300, 600, 3), 255, np.uint8))
    assert detector.stats["rescues"] == 0 and detector.stats["rescue_skipped"] > 0


# --- C. real OCR, judged on the final pixels -------------------------------------------


def _vision():
    if sys.platform != "darwin":
        return None
    try:
        from eclipse.platform import AppleVisionReader

        reader = AppleVisionReader()
        reader(np.full((40, 120, 3), 255, np.uint8))
        return reader
    except Exception:
        return None


VISION = _vision()
needs_vision = pytest.mark.skipif(VISION is None, reason="needs macOS Vision")


# What Vision's fast read garbles past recovery at 10 pt Helvetica, measured: the JSON
# line on selected text loses its quotes and colon, and ${DB_PASSWORD} reads as
# "SIDB PASSWORD}" or "qDB PASSWORD}", which is covered: wrong, but safe. Nothing else
# may miss or cover falsely, and these may not spread.
KNOWN_FAST_LIMITS = {
    ("helvetica", "small", "selected"): ({"json_password"}, {"template_reference"}),
    ("helvetica", "small", "selected_dark"): ({"json_password"}, {"template_reference"}),
    ("helvetica", "small", "dark"): (set(), {"template_reference"}),
    ("helvetica", "small", "light"): (set(), {"template_reference"}),
    ("helvetica", "small", "colored"): (set(), {"template_reference"}),
}


@needs_vision
@pytest.mark.parametrize("font", ["menlo", "helvetica", "sf"])
@pytest.mark.parametrize("theme", ["dark", "light", "selected", "selected_dark", "colored"])
@pytest.mark.parametrize("size", ["normal", "small"])
def test_real_ocr_screens_hide_secrets_and_leave_lookalikes(font, theme, size):
    from eval.accuracy_scene import run_screen

    result = run_screen(font, size, theme, reader=VISION)
    may_miss, may_cover = KNOWN_FAST_LIMITS.get((font, size, theme), (set(), set()))
    assert set(result["missed"]) <= may_miss, result["cases"]
    assert set(result["false"]) <= may_cover, result["cases"]


@needs_vision
@pytest.mark.parametrize("theme", ["dark", "light", "colored"])
def test_rescue_reads_recover_small_text_the_fast_read_garbles(theme):
    from eval.accuracy_scene import run_screen

    result = run_screen("helvetica", "small", theme, reader=VISION, rescue=True)
    assert not result["missed"], result["cases"]
    assert result["rescues"] <= 4  # only the doubtful lines, batched


@needs_vision
def test_real_ocr_covers_go_out_through_the_compositor_padding():
    from eval.accuracy_scene import render

    image, planted = render("menlo", "normal", "dark")
    image.flags.writeable = False
    detect = TextDetector(reader=VISION, pixel_scale=2)
    output = Compositor(padding=PADDING * 2).render(Frame(image, 0.0, "screen", 0), detect).image
    for item in planted:
        if item.secret:
            assert not (item.mask & (output == image).all(axis=2)).any(), item.case
