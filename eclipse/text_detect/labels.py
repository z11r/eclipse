"""Password labels however they're spelled: typed with a typo, or misread by OCR.

Labels are compared in a folded view: case, accents, full-width forms, invisible
characters, Latin look-alikes from other scripts (only inside a word that mixes scripts),
and OCR's look-alikes such as 0 for o and 5 for s. Long labels may also be one edit off:
one letter added, dropped, changed, or two neighbours swapped, so pasword, passwrod and
DB_PASWORD count. Short labels such as pwd stay exact, as one edit there makes another
word. A compound name counts only if it names the password itself: DB_PASWORD does,
password_length and passwordField don't.

Only labels are folded. Values are never changed, so a cover lands on what's on screen.
"""

from __future__ import annotations

import functools
import re
import unicodedata

# Labels compared exactly, after folding. passphrase and password are long enough to
# also match one edit away; the rest are short, where one edit is a different word.
_EXACT_LABELS = frozenset({'password', 'passwd', 'pwd', 'passphrase', 'pswd', 'paswd'})
_FUZZY_LABELS = ('password', 'passphrase')
_FUZZY_MIN = 7  # letters a folded label needs before one edit is allowed
# A compound name ending in one of these after the password word still names the
# password itself: new_password, password_confirm, PASSWORD2.
_SAME_FIELD = frozenset({
    'confirm', 'confirmation', 'again', 'repeat', 'new', 'old', 'current', 'plain',
    'plaintext', 'value', 'text', 'input',
})
# OCR look-alikes, applied to labels only. Two-letter shapes first.
_LOOKALIKE_PAIRS = (('vv', 'w'), ('cl', 'd'))
_LOOKALIKE_CHARS = str.maketrans({
    '0': 'o', '4': 'a', '@': 'a', '5': 's', '$': 's', '1': 'l', '|': 'l', '3': 'e',
})
# Latin look-alikes from other scripts. Applied only to a word that mixes them with Latin
# letters, so ordinary Cyrillic or Greek text is never read as Latin.
_SCRIPT_LOOKALIKES = str.maketrans({
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'ѕ': 's', 'ԁ': 'd', 'ѡ': 'w',
    'ԝ': 'w', 'х': 'x', 'у': 'y', 'і': 'i', 'ο': 'o', 'ρ': 'p', 'α': 'a', 'ѵ': 'v',
})

_MAX_KEY = 48  # characters looked back from a separator for its label

_CLOSERS = '"\'`’”」]'


def _invisible(char: str) -> bool:
    return unicodedata.category(char) in ('Cf', 'Mn', 'Me')


@functools.lru_cache(maxsize=4096)
def fold(word: str) -> str:
    """A label's matching form: lower case, plain letters, look-alikes replaced.

    Only for comparing labels. Values are never folded.
    """
    letters = []
    for char in word:
        if _invisible(char):
            continue
        for part in unicodedata.normalize('NFKD', char):
            if not _invisible(part):
                letters.append(part)
    folded = ''.join(letters).casefold()
    has_latin = any('a' <= char <= 'z' for char in folded)
    if has_latin and not folded.isascii():
        folded = folded.translate(_SCRIPT_LOOKALIKES)
    if any(char.isalpha() for char in folded):
        for pair, single in _LOOKALIKE_PAIRS:
            folded = folded.replace(pair, single)
        folded = folded.translate(_LOOKALIKE_CHARS)
    return folded


def _one_edit(word: str, target: str) -> bool:
    """Whether word is at most one insertion, deletion, substitution or swap of
    neighbours away from target (Damerau-Levenshtein distance 1)."""
    if word == target:
        return True
    a, b = len(word), len(target)
    if abs(a - b) > 1:
        return False
    first = 0
    while first < min(a, b) and word[first] == target[first]:
        first += 1
    if a == b:
        if word[first + 1:] == target[first + 1:]:
            return True  # substitution
        return (
            first + 1 < a and word[first] == target[first + 1]
            and word[first + 1] == target[first] and word[first + 2:] == target[first + 2:]
        )
    if a > b:
        return word[first + 1:] == target[first:]  # insertion
    return word[first:] == target[first + 1:]  # deletion


@functools.lru_cache(maxsize=8192)
def is_label(word: str) -> bool:
    """Whether one word, such as PASWORD or passw0rd, is a password label."""
    folded = fold(word)
    if folded in _EXACT_LABELS:
        return True
    return len(folded) >= _FUZZY_MIN and any(_one_edit(folded, label) for label in _FUZZY_LABELS)


def _parts(name: str) -> list[str]:
    """Split a compound name, DB_PASSWORD or dbPassword or db.password, into words."""
    words = []
    for piece in re.split(r'[_\-.\s]+', name):
        if not piece:
            continue
        # camelCase and PascalCase, not ALLCAPS or digits inside a word.
        words.extend(re.findall(r'[A-Z]?[^A-Z\s]+|[A-Z]+(?![^A-Z\s])', piece) or [piece])
    return words


@functools.lru_cache(maxsize=8192)
def names_password(name: str) -> bool:
    """Whether a field name, possibly compound, names a password itself.

    DB_PASWORD and dbPassword do; password_length and passwordField don't.
    """
    if is_label(name):
        return True
    words = _parts(name)
    if not words:
        return False
    while len(words) > 1 and (fold(words[-1]) in _SAME_FIELD or words[-1].isdigit()):
        words = words[:-1]
    if is_label(words[-1]):
        return True
    # A run of digits glued on, as in password2.
    return bool(re.fullmatch(r'.*\D\d{1,2}', words[-1])) and is_label(words[-1].rstrip('0123456789'))


def key_before(text: str, end: int) -> tuple[str, int] | None:
    """The field name that ends at end, after any closing quote or bracket, and its start.

    Letters spaced out, as in "p a s s w o r d" or "pass word", are joined back up.
    """
    position = end
    while position > 0 and text[position - 1] in ' \t\n\r\u00a0':
        position -= 1
    closers = 0
    while position > 0 and text[position - 1] in _CLOSERS and closers < 3:
        position -= 1
        closers += 1
    stop = position
    limit = max(0, stop - _MAX_KEY)
    start = stop
    while start > limit and (
        text[start - 1].isalnum() or text[start - 1] in '_-.$@' or _invisible(text[start - 1])
    ):
        start -= 1
    if start == stop:
        return None
    word = text[start:stop]
    if names_password(word):
        return word, start
    # Spaced-out letters: try joining the last few words, exactly.
    pieces = [word]
    cursor = start
    while len(pieces) < 10 and cursor > limit:
        gap = cursor
        while gap > limit and text[gap - 1] in ' \u00a0':
            gap -= 1
        if gap == cursor or cursor - gap > 1 or gap == limit:
            break
        begin = gap
        while begin > limit and text[begin - 1].isalnum():
            begin -= 1
        if begin == gap or gap - begin > 6:
            break
        pieces.insert(0, text[begin:gap])
        cursor = begin
        if len(pieces) > 1 and fold(''.join(pieces)) in _EXACT_LABELS:
            return ''.join(pieces), cursor
    return None



_HINT_WORD = re.compile(r'[^\W\d_][\w\u200b-\u200d\u2060\u00ad]{2,15}')


@functools.lru_cache(maxsize=8192)
def _near_label(word: str) -> bool:
    folded = fold(word)
    if len(folded) < 5 or len(folded) > 12:
        return False
    for label in _FUZZY_LABELS:
        if abs(len(folded) - len(label)) <= 2 and _distance_at_most_two(folded, label):
            return True
    return False


def _distance_at_most_two(word: str, target: str) -> bool:
    """Damerau-Levenshtein distance of at most two, for asking for a careful read only."""
    rows = len(target) + 1
    previous2 = None
    previous = list(range(rows))
    for i, char in enumerate(word, 1):
        current = [i] + [0] * (rows - 1)
        for j in range(1, rows):
            cost = char != target[j - 1]
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if (
                previous2 is not None and i > 1 and j > 1
                and char == target[j - 2] and word[i - 2] == target[j - 1]
            ):
                current[j] = min(current[j], previous2[j - 2] + 1)
        if min(current) > 2:
            return False
        previous2, previous = previous, current
    return previous[-1] <= 2


def looks_like_label(text: str) -> tuple[int, int] | None:
    """Span of a word that may be a password label OCR misread, two edits away or less.

    Only for asking for a careful read: never enough to cover anything on its own.
    """
    for word in _HINT_WORD.finditer(text):
        if _near_label(word.group()):
            return word.span()
    return None


# A word OCR garbled, such as a misspelled label under Notes' red spell-check underline
# (passwrod: read as RSIAI.wiod:): letters mixed with stray marks, right before : or =.
_GARBLED_LABEL = re.compile(
    r'(?<!\S)(?=(?:[^\s:=]*[A-Za-z]){2})(?=[^\s:=]*[^\w\s:="\'(){}\[\]])[^\s:=]{4,16}[:=]'
)


def garbled_label(text: str) -> bool:
    """Whether a word that reads as junk sits where a label would, before : or =.

    Only for asking for a careful read: it says nothing about what the word was.
    """
    return _GARBLED_LABEL.search(text) is not None


# Real words two edits or fewer from a label, from the system word list. They are words,
# not misread labels.
_REAL_NEAR_WORDS = frozenset({'basswood', 'paraphrase', 'passport', 'passports', 'passworts'})


@functools.lru_cache(maxsize=8192)
def names_password_carefully(name: str) -> bool:
    """names_password, but a label may be two edits off, for words a careful reread gave.

    The fast read garbled the line and the careful read came close (Rasswrod: for a
    misspelled, underlined passwrod:), which together are stronger evidence than either.
    """
    if names_password(name):
        return True
    words = _parts(name) or [name]
    last = words[-1]
    return fold(last) not in _REAL_NEAR_WORDS and _near_label(last)


@functools.lru_cache(maxsize=4096)
def cut_short(word: str) -> bool:
    """Whether a word is a label with letters missing from its end, as OCR leaves a longer
    word it cut short: passwor from password_length."""
    folded = fold(word)
    return any(len(folded) >= 4 and label.startswith(folded) and folded != label
               for label in _FUZZY_LABELS + tuple(_EXACT_LABELS))
