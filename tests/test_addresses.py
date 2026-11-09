"""Postal addresses in OCR text. Every address here is made up or a well-known public
landmark; none is anyone's home."""

import pytest

from eclipse.text_detect.addresses import address_spans
from eclipse.text_detect.detection import CATEGORIES, classify_text, find_matches


def covered(text):
    return [text[start:end] for start, end in address_spans(text)]


@pytest.mark.parametrize(
    "text, part",
    [
        ("1458 Maple Street", "1458 Maple Street"),
        ("Springfield, OR 97477", "Springfield, OR 97477"),
        ("Springfield, Oregon 97477", "Springfield, Oregon 97477"),
        ("742 EVERGREEN TERRACE", "742 EVERGREEN TERRACE"),
        ("221B Baker Street", "221B Baker Street"),
        ("123 N Main St Apt 4B", "123 N Main St Apt 4B"),
        # Edmonton's numbered streets, and a unit written before the building
        ("99901 Example Ave NW", "99901 Example Ave NW"),
        ("99902 999 Ave NW", "99902 999 Ave NW"),
        ("204-99903 999 St NW", "204-99903 999 St NW"),
        ("Edmonton, AB T6G 2R3", "Edmonton, AB T6G 2R3"),
        ("Toronto ON M5V 9Z9", "Toronto ON M5V 9Z9"),
        ("T6G 2R3", "T6G 2R3"),
        ("1 Infinite Loop, Cupertino, CA 95014", "1 Infinite Loop"),
        ("1 Infinite Loop, Cupertino, CA 95014", "Cupertino, CA 95014"),
        ("P.O. Box 1234", "P.O. Box 1234"),
        ("PO Box 77", "PO Box 77"),
        ("10 Downing Street, London SW1A 2AA", "SW1A 2AA"),
        ("12 rue des Exemples, 75001 Paris", "12 rue des Exemples"),
        ("Beispielstraße 12, 10115 Berlin", "Beispielstraße 12"),
        ("Calle Mayor 5", "Calle Mayor 5"),
        ("Address: 4 Privet Drive", "4 Privet Drive"),
        ("Home address: 9 Example Wynd", "9 Example Wynd"),
        ("Address: 12 Example", "12 Example"),  # no street type, but the label says so
        ("i live at 12 maple st", "12 maple st"),
        ("ship to 77 elm grove please", "77 elm grove"),
    ],
)
def test_addresses_are_found(text, part):
    assert part in covered(text)


@pytest.mark.parametrize(
    "text",
    [
        "3 Google Drive files",
        "took 2 test drive laps",
        "came in 3rd place",
        "2 way street",
        "Chapter 12 Sesame Street",
        "IP address: 192.168.1.10",
        "Email address: jordan@example.com",
        "MAC address 00:1A:2B:3C:4D:5E",
        "Wallet address 0x52908400098527886E0F7030069857D2E4169EE7",
        "Address 0x7ffd5e8c",
        "the address bar",
        "Apple M2 8GB",
        "16 GB RAM",
        "Customer since 2021 · 14 orders",
        "March 14, 1994",
        "Start at 7 PM, 3 hour stream",
        "Return to 3 Main Menu",
        "Room 204",
        "RETRY_LIMIT = 5",
        'API_VERSION = "2026-11-01"',
        'ALLOWED_HOSTS = ["localhost", "127.0.0.1"]',
        "total > 10_000:",
        "version 4.18.2",
        "page 3 of 12",
    ],
)
def test_lookalikes_are_not_addresses(text):
    assert covered(text) == []


def test_a_lowercase_street_needs_a_hint():
    assert covered("we drove 12 maple st") == []
    assert covered("I live at 12 maple st") == ["12 maple st"]


def test_a_labels_value_stops_after_a_few_words():
    text = "Address 1458 Maple Street Springfield OR Date of birth March 14 1994 Plan Pro"
    assert all(len(part.split()) <= 8 for part in covered(text))


def test_address_is_a_detector_category():
    assert "address" in CATEGORIES
    assert classify_text("1458 Maple Street") == {"address"}


def test_only_the_value_is_covered_not_the_label():
    text = "Address 1458 Maple Street"
    spans = {(start, end) for category, start, end in find_matches(text) if category == "address"}
    assert spans == {(text.index("1458"), len(text))}


@pytest.mark.parametrize(
    "text, part",
    [
        # Apple OCR on a Notes window: a spell-check underline adds a dot, and ß reads as B
        ("i live at 12 maple s.t, by", "12 maple s.t"),
        ("BeispielstraBe 12", "BeispielstraBe 12"),
    ],
)
def test_ocr_misreads_of_an_address_still_match(text, part):
    assert part in covered(text)
