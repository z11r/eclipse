"""Finds where phone numbers appear in a text input"""
import phonenumbers

def find_phone_numbers(text: str) -> list[tuple[int, int]]:
    """Return (start, end) positions of international phone numbers."""
    return [
        (match.start, match.end)
        for match in phonenumbers.PhoneNumberMatcher(
            text,
            None,
            leniency=phonenumbers.Leniency.VALID,
        )
    ]