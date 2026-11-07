"""Finds where emails appear in a text input."""
import re

# Checks for [name]+[@]+[domain]+[.][com, org, etc] format
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

def find_emails(text: str) -> list[tuple[int, int]]:
    """Return (start, end) positions of every email address in text."""
    return [match.span() for match in _EMAIL.finditer(text)]
