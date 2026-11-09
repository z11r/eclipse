"""Rules that find sensitive text in what OCR read, one line at a time.

Each rule weighs its own evidence. A provider prefix, a checksum or a field label is strong;
a long run of hex digits or a bare number is weak and depends on what else the line says.
Rules return the span of the sensitive value, not its label, so a cover can leave
"password:" showing. Where OCR gives one box for a whole line, the detector widens the
cover back to the label, as it can't place the value's first character from the text alone.

Every rule works on the text as read. Positions point into that text, even after look-alike
characters such as full-width digits are converted.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Iterable, NamedTuple

from eclipse.text_detect import addresses, labels


class Span(NamedTuple):
    """One sensitive value in a line of text.

    start and end bound the value itself. anchor is where its evidence starts, such as the
    label in "password: x", and equals start when there is no label. standalone means the
    value needs nothing around it to count, so it can be found in one OCR word on its own.
    uncertain means the shape fits but a check failed, as when one misread digit breaks a
    card's checksum: it's still covered, and counted apart.
    """

    category: str
    start: int
    end: int
    anchor: int
    standalone: bool
    uncertain: bool = False


# Stands for a mark OCR saw ink for but didn't read, between a label and its value, as
# when Vision drops the "=" in "password = hunter2". The detector puts it in a line's
# text only where the pixels show something there. It isn't whitespace to any regex.
UNREAD_MARK = '\u2063'
# OCR can read @ as a copyright sign, and the last dot as a comma or a space. Only
# common endings count without a dot, so "@alice hello" isn't an email.
_EMAIL = re.compile(
    r'[A-Z0-9._%+\-]+\s*[@©]\s*[A-Z0-9.\-]+'
    r'(?:\s*\.\s*[A-Z]{2,}|[\s,]+(?:com|org|net|edu|gov|io|ca|co|dev|ai)\b)',
    re.I,
)
# North American numbers, which need no label: their shape is distinctive.
_PHONE = re.compile(
    r'(?<!\w)(?:\+\d{1,3}[ .\-]?)?(?:\(\d{3}\)|\d{3})[ .\-]?\d{3}[ .\-]?\d{4}(?!\w)'
)
# Any number written with its country code: a plus, then 8 to 15 digits in all (the most
# E.164 allows), in groups. Demo numbers count too: covering them is what a demo needs.
_PHONE_INTERNATIONAL = re.compile(
    r'(?<![\w+])\+[1-9]\d{0,2}(?:[ .\-]?\(?\d{1,5}\)?){1,6}(?![\w+])'
)
_PHONE_DIGITS = (8, 15)
# A number right after a word that ends in a stray 1, l or I: OCR reads "!" as one of
# these and joins it to the word, so "stream! 780-555-0123" comes back as
# "stream1780-555-0123". Only a number written with its separators counts, so an ID like
# "build1780555012" isn't taken for one.
_PHONE_AFTER_MARK = re.compile(
    r'(?<=[A-Za-z][1lI])(?:\(\d{3}\) ?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?!\w)'
)
# A labeled phone remains private even when OCR misreads some digits, and in any region's
# own format. Without a colon the number must start right after the label.
_PHONE_LABEL = re.compile(
    r'\b(?:phone|telephone|mobile|tel|cell|whatsapp|fax)\b\.?\s*(?:number|no\.?|#)?'
    rf'(?:\s*[:={UNREAD_MARK}]\s*([^\r\n:;]{{7,40}})|\s+([+(]?\d[^\r\n:;]{{6,39}}))',
    re.I,
)
# Security words that make a short number a code. Words like "number" or "downloads" don't.
_CODE_HINT = re.compile(
    r'\b(?:code|otp|verification|verify|2fa|mfa|passcode|pin|one[- ]?time|authenticat\w*'
    r'|two[- ]factor|2[- ]step|security|sign[- ]?in|log[- ]?in)\b',
    re.I,
)
_CODE = re.compile(r'(?<!\w)\d(?:[ \-]?\d){3,7}(?!\w)')
# A 6-8 digit number with nothing else on its line, as an authenticator app or a code
# screen shows one. Next to a label that isn't a security word, it's an ordinary number.
_CODE_ALONE = re.compile(r'(?<![\w.:/\-])(?:\d{6,8}|\d{3}[ \-]\d{3})(?![\w.:/\-])')
_CARD = re.compile(r'(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)')
# A card number printed in its usual groups: 4-4-4-4 (and 4-4-4-4-3), or 4-6-5 for Amex.
_CARD_GROUPED = re.compile(
    r'(?<![\d\-])(?:[2-6]\d{3}([ \-])\d{4}\1\d{4}\1\d{4}(?:\1\d{3})?'
    r'|3\d{3}([ \-])\d{6}\2\d{4,5})(?![\d\-])'
)
_CARD_LABEL = re.compile(
    r'\b(?:card|credit|debit|visa|master\s?card|amex|american\s+express|discover|cc'
    r'|pan|payment)\b',
    re.I,
)
_KEYS = [
    re.compile(r'\b(?:sk|rk)-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{16,}\b'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    # AKIA or ASIA then 16 letters and digits. Real keys are upper case, but OCR reads
    # some letters as lower case and can split or merge one ("NN" as "fvN"), and the
    # prefix is evidence enough.
    re.compile(r'\b(?:AKIA|ASIA)\s?[A-Za-z0-9]{14,18}\b'),
    re.compile(r'\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b'),
    re.compile(r'\bxox[abeoprs]-[A-Za-z0-9\-]{10,}'),
    re.compile(r'\bAIza[A-Za-z0-9_\-]{30,}'),
    re.compile(r'\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b'),
    re.compile(r'\bBearer\s+[A-Za-z0-9._~+/=\-]{12,}', re.I),
    # A password inside a link: postgres://user:password@host
    re.compile(r'\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s/@]+@', re.I),
    # A private key's header.
    re.compile(r'-{2,}\s*BEGIN [A-Z ]*PRIVATE KEY'),
]
# Long runs of hex, and long base64 lines such as a private key's body. Random-looking
# alone, which is weak evidence: a line that says it shows a hash or commit is public.
_GENERIC_KEYS = [
    re.compile(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])'),
    re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/])'),
]
_HASH_CONTEXT = re.compile(
    r'(?:\b|_)(?:sha(?:-?\d{1,3})?(?:sum)?|md5(?:sum)?|checksum|digest|hash|commi[tl1]|integrity'
    r'|etag|revision|rev|git|blob|tree|fingerprint|crc32?)(?:\b|_|[:=])',
    re.I,
)
_CASE_CHANGE = re.compile(r'([a-z])([A-Z])')
_CREDENTIAL_CONTEXT = re.compile(
    r'(?:\b|_)(?:api[_ -]?key|key|token|secret|passw\w*|pwd|credentials?|bearer'
    r'|auth(?:orization)?)(?:\b|_)',
    re.I,
)
# A field name that labels a secret, ending in one of these words, so DB_PASSWORD and
# AWS_SECRET_ACCESS_KEY count. Then its separator: = : => := ==, or an unread mark.
_PASSWORD_WORDS = r'passw(?:or)?d|passphrase|pwd|[_\-]pass'
_SECRET_WORDS = (
    r'api[_ -]?key|access[_ -]?key|access[_ -]?token|auth[_ -]?token|token'
    r'|secret(?:[_ -]?key)?|private[_ -]?key|authorization'
)
_SEPARATOR = rf'(?:=>|:=|==|=|:|{UNREAD_MARK})(?!=)'
_LABEL = re.compile(
    r'(?<![A-Za-z0-9_\-])(?P<open>["\x27])?[A-Za-z0-9_\-]*?'
    rf'(?:(?P<password>{_PASSWORD_WORDS})|(?P<secret>{_SECRET_WORDS}))(?![A-Za-z0-9])'
    rf'(?(open)(?P=open)\]?(?:\s*(?P<sep>{_SEPARATOR})|(?P<misquoted>\.\.|[;"\x27]\.?)(?=\s))'
    # OCR reads a colon as a quote or a semicolon. Only after a name with no opening
    # quote, as a quote after "password" then closes it.
    rf'|(?:\]?\s*(?P<bare>{_SEPARATOR})|(?P<misread>\.\.|[;"\x27]\.?)(?=\s)|(?P<dotted>\.)(?=\s*{UNREAD_MARK})))'
    # Then spaces, and any mark OCR didn't read, like an opening quote.
    rf'[\s{UNREAD_MARK}]*',
    re.I,
)
# A word ending in a label with no separator read after it, so the detector checks the
# pixels before the next word for one OCR missed.
_LABEL_END = re.compile(
    rf'(?:{_PASSWORD_WORDS}|{_SECRET_WORDS}|phone|tel|mobile|card|pin|code)["\x27]?$', re.I
)


_LAST_WORD = re.compile(r'([\w.\-$@]+)["\x27]?$')
# A separator a misspelled label can stand before. The exact labels above also take
# separators OCR misread; a label that is itself a guess needs a real one, or an unread mark.
_FUZZY_SEPARATOR = re.compile(rf'(?:=>|:=|==|=|:|{UNREAD_MARK})(?!=)')
# A separator read after an unread mark: the mark was only the gap before it.
_REAL_SEPARATOR = re.compile(rf'(=>|:=|==|=|:)(?!=)[\s{UNREAD_MARK}]*')
_FUZZY_MAX_SEPARATORS = 64  # separators tried in one line, so dense text stays cheap


# Words that say a secret is on this line. When no value is found beside one, the fast
# read may have garbled it, so the line is worth reading again more carefully.
_NAMES_SECRET = re.compile(
    r'(?:passw(?:or)?d|passphrase|pwd|secret|api[_ -]?key|private[_ -]?key|access[_ -]?token'
    r'|card\s*(?:number|no)|cvv|cvc'
    # Codes with letters, which a spell-check underline turns to junk: read them again.
    r'|(?:verification|recovery|backup|security)\s+codes?)',
    re.I,
)


@functools.lru_cache(maxsize=16384)
def names_a_secret(text: str) -> bool:
    """Whether a line mentions a secret by name, whatever follows. A word up to two edits
    from "password" counts, as OCR may have garbled the label, and so does a junk word
    before : or =, as a spell-check underline makes of a misspelled label. A careful read
    settles it."""
    return (
        _NAMES_SECRET.search(text) is not None
        or labels.looks_like_label(text) is not None
        or labels.garbled_label(text)
    )


@functools.lru_cache(maxsize=16384)
def ends_in_label(text: str) -> bool:
    """Whether a word ends in a secret's label without a separator after it, including a
    misspelled password label such as pasword or DB_PASWORD."""
    if _LABEL_END.search(text) is not None:
        return True
    # A misspelled label with no separator read after it. Not one cut short (passwor),
    # which is likelier the start of a longer word OCR split, as password_length.
    word = _LAST_WORD.search(text)
    return (
        word is not None
        and labels.names_password(word.group(1))
        and not labels.cut_short(_parts_last(word.group(1)))
    )


def _parts_last(name: str) -> str:
    parts = labels._parts(name)
    return parts[-1] if parts else name


_SECRET_MIN = 6  # an unlabelled-looking token: shorter values after "token:" are words
_WORDS_AFTER_COLON = 6  # most words a "password: correct horse ..." value runs on for
# Values that only name where a secret lives: calls, environment and template variables,
# and lookups on settings objects. The secret itself isn't on screen.
_REFERENCE = re.compile(
    # OCR can read a space after a dot: os. getenv(
    r'(?:[A-Za-z_][\w.]*(?:\.[ \t]+[A-Za-z_][\w.]*)*\s*\('
    # A lookup by a setting's name, however OCR garbles the call before it: a spell-check
    # underline turned os.getenv("DB_PASSWORD") into R%AoionY("D8_PASSWORD").
    # The closing ) OCR can read as } ] | l J or 1, as under a spell-check underline.
    r'|\S*\(\s*["\x27`]?[A-Z][A-Z0-9_]{2,}["\x27`]?\s*[)}\]|lJ1]'
    # ${NAME} as OCR reads it: $ as S or 5, { as I, l or (.
    r'|[S5][{(Il1|]\s*[A-Z][A-Z0-9]*_[A-Z0-9_]+\s*[})Il1|]'
    r'|\$\{|\$\(|\$[A-Z_][A-Z0-9_]*(?![\w])|%[A-Za-z_]\w*%|\{\{|<%'
    # Lookups of the environment, through OCR's usual misreads of t, ( and _ at small
    # sizes: os.gelenv{"DB, os.getenvl"DB, and a space after the dot: os. getenvl"DB.
    r'|(?:os\.[ \t]*)?ge[tl1]env|os\.[ \t]*environ|process\.[ \t]*env'
    r'|(?:import\.meta\.env|ENV|env|environ|settings'
    r'|config|conf|cfg|self|this|secrets|vault|args|opts|options|params|request|props)'
    r'(?:\.|\[))'
)
_EMPTY_VALUES = frozenset(
    {'none', 'null', 'nil', 'undefined', 'true', 'false', 'required', 'optional'}
)
# Type annotations, as in def login(password: str). What follows "=" is still a value.
_TYPE_NAME = re.compile(
    r'(?:str|bytes|int|bool|string|String|SecretStr|Optional\[[^\]]*\]|Any|char\s*\*?)'
    r'\s*(?:(=)|[,)\]]|$)'
)
_FIELD_START = re.compile(r'[A-Za-z_][\w.\-]*["\x27]?\s*(?:=>|:=|=|:)')
_PUNCTUATION_ONLY = re.compile(r'[\s"\x27`()\[\]{},;:.]*')
# Characters OCR tends to swap, for matching keywords.
_LOOKALIKES = {
    '0': 'o', 'o': '0', '1': 'il|', 'i': '1l|', 'l': '1i|',
    '2': 'z', 'z': '2', '5': 's', 's': '5', '8': 'b', 'b': '8',
}
_SHORT_KEYWORD = 4  # below this many characters, match whole words only


def luhn_valid(value: str) -> bool:
    digits = [int(c) for c in value if c.isascii() and c.isdigit()]

    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False

    total = 0

    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


def keyword_pattern(word: str) -> re.Pattern:
    """Match a keyword through spacing changes and common OCR misreads.

    OCR can drop spaces or add them inside a word, so any gap is allowed anywhere.
    """
    letters = ''.join(unicodedata.normalize('NFKC', word).lower().split())
    pattern = r'\s*'.join(
        '[' + re.escape(char + _LOOKALIKES.get(char, '')) + ']' for char in letters
    )

    if len(letters) < _SHORT_KEYWORD:
        pattern = rf'(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])'

    return re.compile(pattern, re.I)


# Curly quotes, as Notes, Pages and Word type them, count as straight ones.
_QUOTES = {'\u201c': '"', '\u201d': '"', '\u201e': '"', '\u2018': "'", '\u2019': "'"}


def standard_characters(text: str) -> tuple[str, list[int] | None]:
    """Convert look-alike characters, such as full-width brackets or curly quotes, to
    standard ones.

    One character can become several, so this also returns where each one came from.
    Covers need positions in the text OCR gave us, not in the converted text.
    """
    if text.isascii():
        return text, None

    converted = []
    origins = []

    for index, char in enumerate(text):
        standard = _QUOTES.get(char) or unicodedata.normalize('NFKC', char)
        converted.append(standard)
        origins.extend([index] * len(standard))

    return ''.join(converted), origins


def _digits(text: str) -> int:
    return sum(char.isdecimal() for char in text)


def _closing_quote(text: str, start: int, quote: str) -> int:
    """Where the quote opened just before start closes, skipping escaped ones; -1 if never."""
    index = start
    while index < len(text):
        char = text[index]
        if char == '\\':
            index += 2
            continue
        if char == quote:
            return index
        index += 1
    return -1


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while end > start and text[end - 1] in ' \t,;':
        end -= 1
    return start, end


def _token_end(text: str, start: int) -> int:
    """The end of an unquoted value with no spaces in it, as in config files and URLs.

    It stops before ; , or & that starts the next field, as in pwd=1234;user=bob.
    """
    index = start
    while index < len(text) and not text[index].isspace() and text[index] != UNREAD_MARK:
        if text[index] in ';,&' and (
            index + 1 == len(text) or text[index + 1].isspace()
            or _FIELD_START.match(text, index + 1)
        ):
            break
        index += 1
    return index


def _words_end(text: str, start: int) -> int:
    """The end of an unquoted value after a colon, which can run over several words, as in
    "Password: correct horse battery staple". It stops at the next field's label or a
    comment, and after a few words, so the rest of a sentence isn't taken as the value."""
    index, words = start, 0
    end = start
    while index < len(text) and words < _WORDS_AFTER_COLON:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] == UNREAD_MARK:
            break  # punctuation OCR didn't read, like the quote and comma after a value
        if _FIELD_START.match(text, index) and words:
            break
        if text.startswith(('#', '//'), index) and words:
            break
        stop = _token_end(text, index)
        if stop == index:  # a separator that starts the next field
            break
        end = index = stop
        words += 1
        if index < len(text) and text[index] in ';,&':
            break
    return end


def _value(text: str, start: int, separator: str, allow_words: bool) -> tuple[int, int] | None:
    """The span of a labelled value starting at start, or None if there's no literal value."""
    if start >= len(text):
        return None
    quote_at = start
    if text[start] in 'frbuFRBU' and start + 1 < len(text) and text[start + 1] in '"\'`':
        quote_at = start + 1  # f"...", r'...'
    if text[quote_at] in '"\'`':
        inside = quote_at + 1
        while inside < len(text) and (text[inside] == UNREAD_MARK or text[inside] == ' '):
            inside += 1  # a mark right after the quote is more of the opening punctuation
        close = _closing_quote(text, inside, text[quote_at])
        mark = text.find(UNREAD_MARK, inside)
        if mark >= 0 and (close < 0 or mark < close):
            close = mark  # OCR dropped the closing quote, and left a mark where it was
        if close < 0:  # OCR dropped the closing quote, or the value runs on past the line
            begin, end = _trim(text, inside, len(text))
        else:
            begin, end = _trim(text, inside, close)
        value = text[begin:end]
        if _REFERENCE.match(value) and ('${' in value or '{{' in value or '%' in value):
            return None
    else:
        if _REFERENCE.match(text, start):
            return None
        if separator == ':':
            typed = _TYPE_NAME.match(text, start)
            if typed:
                if not typed.group(1):
                    return None
                after = typed.end()
                while after < len(text) and text[after].isspace():
                    after += 1
                return _value(text, after, '=', False)
        end = _words_end(text, start) if allow_words and separator == ':' else _token_end(
            text, start
        )
        begin, end = _trim(text, start, end)
        value = text[begin:end]
        if value.lower().rstrip('.,;)') in _EMPTY_VALUES:
            return None
    if not value or _PUNCTUATION_ONLY.fullmatch(value):
        return None
    return begin, end


def _labelled_secrets(text: str) -> Iterable[Span]:
    for match in _LABEL.finditer(text):
        if match.group('misread') and text[max(match.start() - 1, 0)] in '"\x27':
            continue  # "password" "x": the quote closes a quoted name, so it's no colon
        password = match.group('password') is not None
        start = match.end()
        if not password and text.startswith(('Bearer ', 'bearer '), start):
            start += len('Bearer ')
        # A colon OCR read as a quote or semicolon stands for a colon.
        separator = match.group('sep') or match.group('bare') or ':'
        if separator == UNREAD_MARK:
            real = _REAL_SEPARATOR.match(text, start)
            if real:
                # The mark stood for the gap before a separator OCR did read, as in
                # password = os.getenv(...): the value is what follows that one.
                separator, start = real.group(1), real.end()
            else:
                # Unknown, perhaps a colon: a password may run on, to the next unread mark.
                separator = ':' if password else '='
        found = _value(text, start, separator, allow_words=password)
        if found is None:
            continue
        begin, end = found
        if not password and len(text[begin:end].strip()) < _SECRET_MIN:
            continue
        yield Span('api_key', begin, end, match.start(), False)
    yield from _misspelled_passwords(text)
    yield from _password_flags(text)


# --password hunter2 on a command line, or -password as some tools and OCR write it. Only
# a flag that spells a label counts: -p can mean anything.
_FLAG = re.compile(r'(?<![\w\-])--?([A-Za-z][\w\-]{2,30})[ \t]+(?=[^\s\-])')


def _password_flags(text: str) -> Iterable[Span]:
    for flag in _FLAG.finditer(text):
        if labels.names_password(flag.group(1)):
            found = _value(text, flag.end(), '=', allow_words=False)
            if found is not None:
                yield Span('api_key', found[0], found[1], flag.start(), False)


def _misspelled_passwords(text: str, careful: bool = False) -> Iterable[Span]:
    """Passwords after a label the exact rule doesn't know: pasword=12345, DB_PASWORD=x,
    passw0rd: x, pass word = x. The value is parsed as for any password label. careful
    lets a label be two edits off, for words a careful reread gave."""
    for count, separator in enumerate(_FUZZY_SEPARATOR.finditer(text)):
        if count >= _FUZZY_MAX_SEPARATORS:
            break
        found = labels.key_before(text, separator.start())
        if found is None and careful:
            found = _careful_key_before(text, separator.start())
        if found is None:
            continue
        key, key_start = found
        if _LABEL.match(text, key_start) or _LABEL.match(text, max(key_start - 1, 0)):
            continue  # an exact label: found above
        sign = separator.group()
        if sign == '=' and text[separator.start() - 1 : separator.start()] in ('<', '>', '!'):
            continue
        if sign == UNREAD_MARK and labels.cut_short(_parts_last(key)):
            continue  # passwor, then a gap: likelier password_length split by OCR
        start = separator.end()
        while start < len(text) and (text[start].isspace() or text[start] == UNREAD_MARK):
            start += 1
        if sign == UNREAD_MARK:
            real = _REAL_SEPARATOR.match(text, start)
            if real:
                sign, start = real.group(1), real.end()
            else:
                sign = ':'
        value = _value(text, start, ':' if sign == ':' else '=', allow_words=True)
        if value is None:
            continue
        begin, end = value
        if sign == ':' and key[-1:] in 'sS' and re.fullmatch(r'\d{1,3}|\d+\s+[A-Za-z]+', text[begin:end]):
            continue  # "Passwords: 3 saved" counts passwords
        yield Span('api_key', begin, end, key_start, False)


def _phones(text: str) -> Iterable[Span]:
    for match in _PHONE.finditer(text):
        yield Span('phone', match.start(), match.end(), match.start(), True)
    for match in _PHONE_AFTER_MARK.finditer(text):
        yield Span('phone', match.start(), match.end(), match.start(), True)
    for match in _PHONE_INTERNATIONAL.finditer(text):
        if _PHONE_DIGITS[0] <= _digits(match.group()) <= _PHONE_DIGITS[1]:
            yield Span('phone', match.start(), match.end(), match.start(), True)
    for match in _PHONE_LABEL.finditer(text):
        group = 1 if match.group(1) is not None else 2
        value = match.group(group)
        if _digits(value) < 7:
            continue
        # From the first character that can start a number to the last digit, so words
        # after it, as in "Phone: 555 0142 Email", aren't covered. Letters inside stay, as
        # OCR misreads digits as letters.
        offset = match.start(group)
        first = next(i for i, c in enumerate(value) if c in '+(' or c.isalnum())
        last = max(i for i, c in enumerate(value) if c.isdecimal() or c == ')')
        yield Span('phone', offset + first, offset + last + 1, match.start(), False)


# A code with letters, after a label that says it's a security code:
# "Recovery code: abcd-efgh-ijkl", "Verification code: AB12-CD34". Digits alone are above.
_LETTER_CODE = re.compile(
    r'\b(?:verification|recovery|backup|security|authentication|login|sign[- ]?in|one[- ]?time'
    r'|2fa|mfa|otp|access|confirmation)\s+codes?\s*[:=]\s*'
    r'(?P<value>[A-Za-z0-9]{2,10}(?:-[A-Za-z0-9]{2,10}){1,5})(?![\w\-])',
    re.I,
)


def _codes(text: str) -> Iterable[Span]:
    for match in _LETTER_CODE.finditer(text):
        yield Span('2fa', match.start('value'), match.end('value'), match.start(), False)
    if _CODE_HINT.search(text):
        for match in _CODE.finditer(text):
            yield Span('2fa', match.start(), match.end(), match.start(), False)
        return
    for match in _CODE_ALONE.finditer(text):
        rest = text[:match.start()] + text[match.end():]
        if not any(char.isalnum() for char in rest):
            yield Span('2fa', match.start(), match.end(), match.start(), False)


def _cards(text: str) -> Iterable[Span]:
    certain = []
    for match in _CARD.finditer(text):
        if luhn_valid(match.group()):
            certain.append((match.start(), match.end()))
            yield Span('card', match.start(), match.end(), match.start(), True)
    labelled = _CARD_LABEL.search(text) is not None
    grouped = [(m.start(), m.end()) for m in _CARD_GROUPED.finditer(text)]
    for start, end in grouped:
        if luhn_valid(text[start:end]):
            if not any(a <= start and end <= b for a, b in certain):
                yield Span('card', start, end, start, True)
        else:
            # Printed like a card, so likely one OCR misread a digit of. Covered, but
            # uncertain, rather than shown because a check failed.
            yield Span('card', start, end, start, False, True)
    if labelled:
        for match in _CARD.finditer(text):
            start, end = match.span()
            # A run that takes in a whole card and the expiry after it is that card.
            known = any(a < end and start < b for a, b in certain + grouped)
            if not known and not luhn_valid(match.group()):
                yield Span('card', start, end, start, False, True)


def _keys(text: str) -> Iterable[Span]:
    for pattern in _KEYS:
        for match in pattern.finditer(text):
            yield Span('api_key', match.start(), match.end(), match.start(), True)
    for pattern in _GENERIC_KEYS:
        for match in pattern.finditer(text):
            around = text[:match.start()] + ' ' + text[match.end():]
            # OCR runs words together ("CommitSHA:"), so split where case changes too.
            around = _CASE_CHANGE.sub(r'\1 \2', around)
            if _HASH_CONTEXT.search(around) and not _CREDENTIAL_CONTEXT.search(around):
                continue  # a commit, checksum or digest the line says it shows
            # Standalone only when a word read alone can't hide a hash label beside it.
            yield Span('api_key', match.start(), match.end(), match.start(), False)


def _match(text: str, keywords: tuple[str, ...]) -> list[Span]:
    standard, origins = standard_characters(text)
    spans = []
    for found in (
        _emails(standard), _phones(standard), _codes(standard), _keys(standard),
        _labelled_secrets(standard), _cards(standard), _addresses(standard),
    ):
        spans.extend(found)
    for word in keywords:
        if word.strip():
            for match in keyword_pattern(word).finditer(standard):
                spans.append(Span('keyword', match.start(), match.end(), match.start(), True))
    if origins is None:
        return spans
    placed = []
    for span in spans:
        if span.end > span.start:
            placed.append(span._replace(
                start=origins[span.start], end=origins[span.end - 1] + 1,
                anchor=origins[min(span.anchor, span.start)],
            ))
    return placed


# Endings a real address has. Only checked when a bare number stands before the @.
_EMAIL_ENDINGS = frozenset({
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'io', 'ca', 'co', 'dev', 'ai', 'app',
    'me', 'us', 'uk', 'de', 'fr', 'nl', 'au', 'jp', 'cn', 'in', 'br', 'ru', 'it', 'es',
    'se', 'no', 'ch', 'info', 'biz', 'xyz', 'tech', 'online', 'site', 'email', 'mail',
})
# Methods a decorator calls: @app.get("/health"), @router.post(...), @pytest.fixture.
_DECORATOR_CALLS = frozenset({
    'get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'route', 'api', 'websocket',
    'middleware', 'fixture', 'mark', 'parametrize', 'command', 'group', 'task', 'setter',
    'getter', 'deleter', 'wraps', 'cache', 'callback', 'listener', 'handler', 'validator',
    'register', 'event', 'on', 'hook', 'job', 'cron', 'retry', 'skip', 'skipif', 'xfail',
})


def _decorator_not_email(match: re.Match) -> bool:
    """An editor's line number joined to a decorator, as words are when tried with no
    space: 31 @app.get("/health") reads as 31@app.getl"/health"), with ( misread as l.

    It's a decorator when the ending is a method called with ( right after it, or the
    ( misread and a quoted argument after. With a bare number before the @, any ending
    an address doesn't have counts too (12@pytest.fixture). Endings an address has stay
    emails, so 12345@gmail.com is still covered."""
    found = match.group()
    at = next(index for index, char in enumerate(found) if char in '@©')
    ending = re.split(r'[\s.,]+', found[at + 1 :].strip())[-1].lower()
    after = match.string[match.end() : match.end() + 1]
    if ending in _DECORATOR_CALLS and after == '(':
        return True
    if ending[-1:] in ('l', 'i') and ending[:-1] in _DECORATOR_CALLS and after in ('"', "'", ')'):
        return True
    name = found[:at].strip(' \t_')
    return name.isdigit() and ending not in _EMAIL_ENDINGS


def _emails(text: str) -> Iterable[Span]:
    for match in _EMAIL.finditer(text):
        if not _decorator_not_email(match):
            yield Span('email', match.start(), match.end(), match.start(), True)


def _addresses(text: str) -> Iterable[Span]:
    # Not standalone: an address is several words, and needs them read together.
    for start, end, anchor in addresses.address_parts(text):
        yield Span('address', start, end, anchor, False)


@functools.lru_cache(maxsize=16384)
def find_spans(text: str, keywords: tuple[str, ...] = ()) -> tuple[Span, ...]:
    """Every sensitive value in one line of text, with its evidence.

    Answers are remembered. A frame re-checks every line on screen, nearly all of them
    unchanged. The answer depends only on the text and the keywords, so a remembered one
    is always the same answer.
    """
    return tuple(_match(text, tuple(keywords)))


def _careful_key_before(text: str, end: int) -> tuple[str, int] | None:
    """The name before a separator, if it is within two edits of a password label."""
    stop = end
    while stop > 0 and text[stop - 1] in ' \t"\x27':
        stop -= 1
    start = stop
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in '_-.$@'):
        start -= 1
    name = text[start:stop]
    return (name, start) if name and labels.names_password_carefully(name) else None


@functools.lru_cache(maxsize=4096)
def find_careful_spans(text: str) -> tuple[Span, ...]:
    """Passwords whose label is two edits off, for a line a careful reread gave.

    Only for such lines: on a fast read, two edits isn't enough to call a word a label.
    """
    standard, origins = standard_characters(text)
    spans = [span for span in _misspelled_passwords(standard, careful=True)]
    if origins is None:
        return tuple(spans)
    return tuple(
        span._replace(start=origins[span.start], end=origins[span.end - 1] + 1,
                      anchor=origins[min(span.anchor, span.start)])
        for span in spans if span.end > span.start
    )
