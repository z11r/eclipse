"""Secret rules against OCR-style text. Every value here is fake.

Keys are joined from pieces so secret scanners don't mistake them for real ones.
"""

import pytest

from eclipse.text_detect.detection import classify_text


@pytest.mark.parametrize(
    "text",
    [
        "sk_" + "live_" + "a1" * 12,
        "rk_" + "test_" + "a1" * 12,
        "xoxb-" + "1" * 11 + "-" + "a" * 24,
        "AIza" + "a" * 35,
        "AKIA IOSFODNN7EXAMPLE",
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
        "MIIE" + "abcDEF12+/" * 6,
        "TOKEN=abcdef123456",
        "PASSWD=abcdef123456",
        "DB_PASSWORD=abcdef123456",
        "CLIENT_SECRET: abcdef123456",
        "AWS_SECRET_ACCESS_KEY=abcdef123456",
        "postgres://user:hunter2@localhost/app",
    ],
)
def test_secrets_are_found(text):
    assert "api_key" in classify_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "demo@example.com",
        "demo @ example . com",
        "demo@example,com",
        "demo©example.com",
        "demo@example com",
    ],
)
def test_emails_are_found_through_ocr_misreads(text):
    assert "email" in classify_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "hi @alice hello there",
        "@app.route",
        "def build_overlay(theme):",
        "return panel.render(theme)",
        "TOKEN_COUNT = 100",
        "max_tokens: 4096",
        "/Users/fake/Documents/project/src/components/panel/index",
    ],
)
def test_ordinary_text_is_left_alone(text):
    assert not classify_text(text) & {"email", "api_key"}


@pytest.mark.parametrize(
    "text", ["Main Street", "main street", "MainStreet", "Main  Street", "Ma1n Street", "MAIN 5TREET"]
)
def test_keywords_survive_spacing_and_misreads(text):
    assert "keyword" in classify_text(text, ["main street"])


def test_long_keyword_still_matches_inside_a_word():
    assert "keyword" in classify_text("github.com/fakeuser99", ["fakeuser"])


def test_short_keyword_matches_whole_words_only():
    assert "keyword" not in classify_text("final total", ["al"])
    assert "keyword" in classify_text("hi al, welcome", ["al"])


def test_keyword_with_symbols_is_matched_literally():
    assert "keyword" in classify_text("user c++dev here", ["c++dev"])
    assert "keyword" not in classify_text("user cdev here", ["c++dev"])


@pytest.mark.parametrize('text', [
    'password=12345',
    'PASSWORD = "a"',
    'DB_PASSWORD=abc',
    'pwd: x',
])
def test_short_labeled_passwords_are_secrets(text):
    assert 'api_key' in classify_text(text)


@pytest.mark.parametrize('text', ['password=', 'password=""', 'pwd: ', 'password strength is low'])
def test_empty_or_unassigned_passwords_are_not_secrets(text):
    assert 'api_key' not in classify_text(text)


@pytest.mark.parametrize('text', ['Yourcode:482619', 'Yourcode:918273', 'Securitycode:1234'])
def test_code_hints_survive_dropped_spaces(text):
    assert '2fa' in classify_text(text)


@pytest.mark.parametrize('text', ['Server runs on port 8080', 'Year: 2026', 'barcode:482619'])
def test_code_hint_fix_does_not_treat_every_number_as_a_code(text):
    assert '2fa' not in classify_text(text)
