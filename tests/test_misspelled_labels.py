"""Password labels typed with a typo or misread by OCR still protect their whole value.

The reported miss had two causes: a misspelled label (pasword=12345) and short values.
Every value here is fake.
"""

import pytest

from eclipse.text_detect import labels
from eclipse.text_detect.detection import classify_text, find_matches
from eclipse.text_detect.rules import ends_in_label, names_a_secret


def values(text):
    return [text[start:end] for category, start, end in find_matches(text) if category == "api_key"]


@pytest.mark.parametrize(
    "text, value",
    [
        ("pasword=12345", "12345"),
        ("password=12345", "12345"),
        ("pasword=hunter2", "hunter2"),
        ("password=hunter2", "hunter2"),
        ("password=abc", "abc"),
        ("pwd=1234", "1234"),
        ('password="correct horse battery staple"', "correct horse battery staple"),
    ],
)
def test_the_reported_examples_cover_exactly_their_value(text, value):
    assert values(text) == [value]


def one_edit_labels(word="password"):
    """Every label one insertion, deletion or substitution over a-z0-9, or one swap of
    neighbours, away from word."""
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    found = set()
    for i in range(len(word) + 1):
        for char in letters:
            found.add(word[:i] + char + word[i:])
    for i in range(len(word)):
        found.add(word[:i] + word[i + 1 :])
        for char in letters:
            found.add(word[:i] + char + word[i + 1 :])
    for i in range(len(word) - 1):
        found.add(word[:i] + word[i + 1] + word[i] + word[i + 2 :])
    found.discard(word)
    return sorted(found)


GENERATED = one_edit_labels()


def test_the_generated_labels_are_the_whole_one_edit_space():
    assert len(GENERATED) == 609


@pytest.mark.parametrize("template", ["{}={}", "{}: {}", "{} = {}", '"{}": "{}"'])
def test_every_one_edit_label_protects_its_value(template):
    missed = []
    for label in GENERATED:
        for value in ("hunter2", "12345", "a"):
            text = template.format(label, value)
            if values(text) != [value]:
                missed.append((text, values(text)))
    assert not missed, missed[:10]


@pytest.mark.parametrize(
    "text",
    [
        "DB_PASWORD=hunter2",
        "MYSQL_ROOT_PASSOWRD=hunter2",
        "dbPasword=hunter2",
        "db.passwrod=hunter2",
        "export ADMIN_PASSWROD=hunter2",
        "NEW_PASWORD=hunter2",
        "pasword_confirm=hunter2",
        "passw0rd=hunter2",
        "pa55word=hunter2",
        "pa55w0rd=hunter2",
        "p4ssw0rd=hunter2",
        "passvvord=hunter2",
        "pass word=hunter2",
        "p a s s w o r d=hunter2",
        "ｐａｓｓｗｏｒｄ=hunter2",
        "pass​word=hunter2",
        "pass word=hunter2",
        "pаssword=hunter2",
        "pássword=hunter2",
    ],
)
def test_misspelled_misread_and_unicode_labels(text):
    assert values(text) == ["hunter2"], find_matches(text)


@pytest.mark.parametrize(
    "text, value",
    [
        ("pasword: correct horse battery staple", "correct horse battery staple"),
        ('{"pasword":"hunter2","user":"bob"}', "hunter2"),
        ("pasword=ab; user=bob", "ab"),
    ],
)
def test_misspelled_labels_get_the_same_value_parsing(text, value):
    assert values(text) == [value], find_matches(text)


@pytest.mark.parametrize(
    "text",
    [
        # Two edits from password, or a short label one edit off: not a label alone.
        "paswrd=hunter2",
        "pwf=1234",
        "pw=1234",
        # Names of other fields, and words that only look like the label.
        "pasword_length=12",
        "password_strength=strong",
        "pasword_reset=true",
        "paswordField.enabled=true",
        "passwordless=true",
        "passport=enabled",
        "passage=chapter1",
        "keyword=example",
        "bypass=1",
        "swordfish=1",
        # References and prose.
        'pasword = os.getenv("DB_PASSWORD")',
        "pasword=${DB_PASSWORD}",
        "def login(user, pasword: str):",
        "Forgot pasword?",
        "Passwords: 3 saved",
        "Paswords: 3 saved",
        "a <= b",
    ],
)
def test_near_labels_and_metadata_are_left_alone(text):
    assert "api_key" not in classify_text(text), find_matches(text)


def test_a_two_edit_label_asks_for_a_careful_read():
    """paswrd isn't trusted as a label, but OCR may have garbled password: the line is
    read again carefully rather than covered or trusted."""
    assert names_a_secret("paswrd = hunter2")
    assert not names_a_secret("background = blue")


@pytest.mark.parametrize("word", ["pasword", "DB_PASWORD", "passw0rd", '"pasword"'])
def test_a_misspelled_label_ending_a_word_is_checked_for_a_dropped_separator(word):
    assert ends_in_label(word)


@pytest.mark.parametrize("word", ["password_length", "passport", "Forgot", "pwf", "passwor", "DB_PASSWOR"])
def test_other_words_are_not(word):
    assert not ends_in_label(word)


def test_other_scripts_are_not_folded_to_latin():
    assert labels.fold("пароль") == "пароль"
    assert labels.fold("pаssword") == "password"


def test_dense_lines_stay_cheap():
    import time

    line = " ".join(f"pasword{i}=x{i} passwrd{i}: y" for i in range(300))
    started = time.perf_counter()
    find_matches(line)
    assert time.perf_counter() - started < 0.1


@pytest.mark.parametrize(
    "text, value",
    [
        ("--password hunter2 --user bob", "hunter2"),
        ("mysql --pasword=hunter2", "hunter2"),
        ("login --pasword hunter2", "hunter2"),
        ("-password hunter2 --user bob", "hunter2"),
    ],
)
def test_password_flags(text, value):
    assert values(text) == [value], find_matches(text)


@pytest.mark.parametrize("text", ["ls --color auto", "--password-file /tmp/x", "-p 8080"])
def test_other_flags_are_left_alone(text):
    assert "api_key" not in classify_text(text), find_matches(text)


@pytest.mark.parametrize(
    "text, value",
    [
        ("Recovery code: abcd-efgh-ijkl", "abcd-efgh-ijkl"),
        ("Verification code: AB12-CD34", "AB12-CD34"),
    ],
)
def test_security_codes_with_letters(text, value):
    spans = [text[a:b] for category, a, b in find_matches(text) if category == "2fa"]
    assert value in spans, find_matches(text)


@pytest.mark.parametrize(
    "text, value",
    [("Promo code: SAVE-2026", "SAVE-2026"), ("Error code: E-1234", "E-1234")],
)
def test_other_codes_with_letters_are_not_security_codes(text, value):
    # Only a security label makes a code with letters one. (The digits rule may still
    # flag the number in them, near the word "code"; that's a separate, older rule.)
    assert value not in [text[a:b] for _, a, b in find_matches(text)], find_matches(text)


def test_a_careful_reread_accepts_a_label_two_edits_off():
    """Notes underlines a misspelled label in red, and the fast read turns it to junk
    (passwrod: as RSIAI.wiod:). That junk asks for a careful read, which came back as
    Rasswrod: -- two edits off. Only a careful reread's words are allowed that far."""
    from eclipse.text_detect.rules import find_careful_spans

    assert names_a_secret("RSIAI.wiod: abc")
    assert values("Rasswrod: abc") == []
    assert [("Rasswrod: abc")[s.start:s.end] for s in find_careful_spans("Rasswrod: abc")] == ["abc"]
    assert find_careful_spans("passport: AB123456") == ()


def _tracks_after_misses(careful, misses):
    """A cover over text whose pixels then change, as in a fractional scroll, and that
    reads then find nothing at."""
    import numpy as np

    from eclipse.text_detect.covers import CoverTracks, Sighting

    tracks = CoverTracks()
    image = np.zeros((100, 200, 3), np.uint8)
    image[40:60, 50:150] = 200
    tracks.update(image, [Sighting("api_key", (50, 40, 150, 60), 1, True, careful)], [(0, 100, 0, 200)])
    changed = image.copy()
    changed[40:60, 50:150] = 90  # the same text, shifted by part of a pixel
    for _ in range(misses):
        tracks.update(changed, [], [(0, 100, 0, 200)])
    return tracks.covers()


def test_a_cover_only_a_careful_read_found_survives_fast_reads_that_miss_it():
    """The fast read never sees a value under a spell-check underline, so two of its
    misses used to end the cover mid-scroll, and the next careful read put it back:
    the box blinked."""
    assert _tracks_after_misses(careful=False, misses=2) == []
    assert _tracks_after_misses(careful=True, misses=7) != []
    assert _tracks_after_misses(careful=True, misses=8) == []  # deleted text still uncovers


def test_a_word_ocr_cut_short_before_a_dropped_separator_is_not_a_label():
    """Vision read password_length=12 as "passwor" and "engt". Taking passwor for a
    misspelled label with its = lost covered "engt" for a frame, then a big box."""
    from eclipse.text_detect.rules import UNREAD_MARK

    assert values(f"passwor {UNREAD_MARK} engt") == []
    assert values("passwor=hunter2") == ["hunter2"]  # with its = read, it's still a label


@pytest.mark.parametrize(
    "text, expected",
    [
        # Ink in the gap before "=" made the detector mark an unread separator there. The
        # garbled reference after the real "=" was then covered as a value.
        ('password ⁣ = R{.A¢.tonv("DB_PASSWORD")', []),
        ("password ⁣ = hunter2", ["hunter2"]),
        ("pasword ⁣ = hunter2", ["hunter2"]),
    ],
)
def test_an_unread_mark_before_a_real_separator_defers_to_it(text, expected):
    assert values(text) == expected, find_matches(text)


@pytest.mark.parametrize(
    "text",
    [
        'password = o{.&¢.toJ).¥("DB_PASSWORD"}',
        'password = os.getenv("DB_PASSWORD"]',
        'password = os.getenv("DB_PASSWORD"|',
    ],
)
def test_a_reference_whose_closing_bracket_ocr_misread_is_still_a_reference(text):
    """A spell-check underline made Vision read ) as }; the harmless line was covered,
    and as it then stood still, its cover stayed."""
    assert values(text) == [], find_matches(text)
