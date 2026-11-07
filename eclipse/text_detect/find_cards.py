"""Finds where cards appear in a text input"""
import re
from eclipse.text_detect.luhn import passes_luhn

# 13 to 19 digits, with at most one space or dash between digits,
# and no extra digit touching either end.
_CANDIDATE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")


def find_cards(text: str) -> list[tuple[int, int]]:
    """Return (start, end) positions of every card number in text

    Only numbers that also pass the Luhn check count
    """
    return [
        match.span()
        for match in _CANDIDATE.finditer(text)
        if passes_luhn(match.group())
    ]
