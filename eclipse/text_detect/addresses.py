"""Postal addresses in one line of OCR text: where a street, a town line or a label's
value is, so it can be covered.

Rules, not a model, as pyap and Microsoft Purview's address classifier do it: a house
number, a name and a street type from a list; a town with its state or province and
postal code; a value after an "Address" label; a PO box. Parsers such as libpostal and
usaddress label any text they're given as parts of an address, so they can't say
whether a line holds one, and libpostal needs about 2 GB of data. A line is checked
once and its answer remembered, so rules cost nothing per frame.

A street needs a capitalized name and type, as addresses are written: "1458 Maple
Street", "99901 Example Ave NW". Typed in lowercase, as in a chat, it needs a hint on
the line too ("I live at 12 maple st"), so "3 test drive" stays uncovered.
"""

from __future__ import annotations

import re

# Street types, from USPS Publication 28 and Canada Post's list, the ones in common use.
# Words that more often name something else on a screen are left out: Center, Point,
# Station, View, Line, Park.
_TYPES = (
    'Street', 'St', 'Str', 'Avenue', 'Ave', 'Av', 'Road', 'Rd', 'Boulevard', 'Blvd',
    'Lane', 'Ln', 'Drive', 'Dr', 'Court', 'Ct', 'Crescent', 'Cres', 'Close', 'Way',
    'Place', 'Pl', 'Terrace', 'Terr', 'Ter', 'Trail', 'Trl', 'Parkway', 'Pkwy',
    'Highway', 'Hwy', 'Circle', 'Cir', 'Square', 'Sq', 'Alley', 'Grove', 'Gardens',
    'Gate', 'Heights', 'Hts', 'Row', 'Mews', 'Walk', 'Loop', 'Path', 'Pike', 'Plaza',
    'Run', 'Expressway', 'Expy', 'Freeway', 'Fwy', 'Turnpike', 'Tpke', 'Landing',
    'Bend', 'Cove', 'Crossing', 'Xing', 'Glen', 'Hollow', 'Ridge', 'Common', 'Commons',
    'Green', 'Meadow', 'Meadows', 'Promenade', 'Esplanade', 'Wynd', 'Quay', 'Rise',
)
_DIRECTIONS = r'(?:N|S|E|W|NE|NW|SE|SW|North|South|East|West|Northeast|Northwest|Southeast|Southwest)'
_DIRECTIONS_LOWER = r'(?:n|s|e|w|ne|nw|se|sw|north|south|east|west)'


def _split(word: str) -> str:
    """A word OCR may split in two: Apple OCR reads monospace Street as St reet, and a
    spell-check underline below st as s.t."""
    return r'[ \t.]?'.join(re.escape(char) for char in word)


def _either_case(words) -> str:
    """Title Case or UPPER CASE, with an optional dot after an abbreviation."""
    forms = sorted({form for word in words for form in (word, word.upper())}, key=len, reverse=True)
    return '(?:' + '|'.join(_split(form) for form in forms) + r')\.?(?![A-Za-z])'


# 1458, 12B, 204-99903 (unit then building, as written in Edmonton), #5-120
_NUMBER = r'(?<![\w.,:/$#\-])#?(?:\d{1,6}[A-Za-z]?[ \t]*[-–][ \t]*)?\d{1,6}(?:[A-Za-z](?![A-Za-z]))?'
_ORDINAL = r'\d{1,4}(?:st|nd|rd|th|ST|ND|RD|TH)?'
# A capitalized word, with the rest of it if OCR split it off: Springf ield.
_CAPITAL_WORD = r"[A-Z][A-Za-z'’.\-]*(?:[ \t][a-z][a-z'’\-]*)?"
_NAME_WORD = rf"(?:{_CAPITAL_WORD}|{_ORDINAL}|{_DIRECTIONS}\.?)"
_NAME_WORD_LOWER = rf"(?:[a-z][a-z'’.\-]*|{_ORDINAL})"
_UNIT = (
    r'(?:[ \t,]+(?:Apt|APT|Apartment|Unit|UNIT|Suite|SUITE|Ste|STE|Room|Rm|Floor|Fl|#)\.?'
    r'[ \t]*#?[A-Za-z0-9\-]{1,6}(?![\w]))?'
)

_STREET = re.compile(
    rf'{_NUMBER}[ \t]+(?:{_NAME_WORD}[ \t]+){{1,4}}?{_either_case(_TYPES)}'
    rf'(?:[ \t,]+{_DIRECTIONS}\.?(?![A-Za-z]))?{_UNIT}'
)
_STREET_LOWER = re.compile(
    rf'{_NUMBER}[ \t]+(?:{_NAME_WORD_LOWER}[ \t]+){{1,4}}?'
    r'(?:' + '|'.join(_split(word) for word in sorted({word.lower() for word in _TYPES}, key=len, reverse=True)) + r')\.?(?![A-Za-z])'
    rf'(?:[ \t,]+{_DIRECTIONS_LOWER}\.?(?![A-Za-z]))?'
)
# Name and type together that name something else: 3 Google Drive files, the 2 Way plan.
_NOT_STREETS = frozenset({
    'google drive', 'one drive', 'onedrive', 'hard drive', 'flash drive', 'usb drive',
    'disk drive', 'test drive', 'external drive', 'network drive', 'shared drive',
    'my drive', 'thumb drive', 'ssd drive', 'no way', 'one way', 'two way', 'milky way',
    'any way', 'the way', 'this way', 'that way', 'first place', 'second place',
    'third place', 'last place', 'market place', 'work place', 'fire place', 'food court',
    'high court', 'supreme court', 'district court', 'tennis court', 'basketball court',
    'times square', 'town square', 'wall street', 'easy street', 'sesame street',
})
# A street type written before its name: 12 rue des Exemples, 5 Calle Mayor.
_PREFIX_TYPES = re.compile(
    rf'{_NUMBER},?[ \t]+'
    r'(?:[Rr]ue|RUE|[Bb]oulevard|[Bb]oul\.|[Cc]hemin|[Cc]alle|[Aa]venida|[Rr]ua|Avenue|Av\.'
    r'|Allée|Impasse|Quai|Place|Via|Viale|Plaza)'
    r"[ \t]+(?:(?:de la|de|du|des|del|della|do|da|of the|of|d')[ \t]*)?"
    r"[A-Z][A-Za-z'’\-]*(?:[ \t]+[A-Z][A-Za-z'’\-]*){0,3}"
)
# The number after the name: Beispielstraße 12, Kalverstraat 92, Calle Mayor 5, Via Roma 10.
_NUMBER_AFTER = re.compile(
    r"(?<![\w])(?:[A-ZÄÖÜ][\wäöüß'’\-]*"
    # OCR reads ß as B: BeispielstraBe.
    r'(?:stra(?:ß|ss|B|b|8)e|str\.|weg|gasse|platz|allee|ring|damm|ufer|gade|vej|gatan|vägen'
    r'|laan|straat|plein|gracht|kade)'
    r"|(?:Calle|Avenida|Rua|Via|Viale|Corso|Piazza|Rue|Carrer|Paseo)[ \t]+[A-Z][\w'’\-]*"
    r"(?:[ \t]+[A-Z][\w'’\-]*){0,2})"
    r',?[ \t]+\d{1,5}[a-zA-Z]?(?![\w.])'
)
_PO_BOX = re.compile(
    r'(?<![\w])(?:P\.?[ \t]?O\.?[ \t]?|Post[ \t]+Office[ \t]+|Postal[ \t]+)?Box[ \t]+#?\d{1,6}(?!\w)'
    r'|(?<![\w])(?:P\.?[ \t]?O\.?|PO)[ \t]*Box[ \t]*#?\d{1,6}(?!\w)',
    re.I,
)

_STATES = (
    'AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH '
    'NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR GU VI'
).split()
_STATE_NAMES = (
    'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida'
    '|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland'
    '|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada'
    '|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio'
    '|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee'
    '|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming'
)
_PROVINCES = 'AB BC MB NB NL NS NT NU ON PE QC SK YT'.split()
_PROVINCE_NAMES = (
    'Alberta|British Columbia|Manitoba|New Brunswick|Newfoundland and Labrador'
    '|Newfoundland|Nova Scotia|Northwest Territories|Nunavut|Ontario'
    '|Prince Edward Island|Quebec|Québec|Saskatchewan|Yukon'
)
# Canada Post leaves out D F I O Q U, and W Z as a first letter: T6G 2R3.
_CA_POSTAL = r'[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ \t\-]?\d[ABCEGHJ-NPRSTV-Z]\d'
_ZIP = r'\d{5}(?:[ \t]*-[ \t]*\d{4})?'
_TOWN = rf"{_CAPITAL_WORD}(?:[ \t]+{_CAPITAL_WORD}){{0,3}}"

_REGION_LINE = re.compile(
    rf'(?<![\w]){_TOWN},?[ \t]+'
    rf'(?:(?:{"|".join(_STATES)}|{_STATE_NAMES})\.?,?[ \t]+{_ZIP}'
    rf'|(?:{"|".join(_PROVINCES)}|{_PROVINCE_NAMES})\.?,?[ \t]+{_CA_POSTAL})(?![\w])'
)
_CA_POSTAL_ALONE = re.compile(rf'(?<![A-Za-z0-9]){_CA_POSTAL}(?![A-Za-z0-9])')
# Royal Mail: SW1A 2AA, M1 1AE. The inward letters leave out C I K M O V, so 8 KB and
# 4 MB can't match; GB and TB are ruled out by hand (an M2 8GB laptop).
_UK_POSTCODE = re.compile(
    r'(?<![A-Za-z0-9])[A-Z]{1,2}\d[A-Z\d]?[ \t]\d[ABD-HJLNP-UW-Z]{2}(?![A-Za-z0-9])'
)
_UK_NOT = re.compile(r'\d(?:GB|TB)$')

# A value after a label: Address: 12 Example Road, Home address 77 Elm Grove.
_LABEL = re.compile(
    r'\b(?P<kind>[A-Za-z\-]+[ \t]+)?'
    r'(?:addr(?:ess)?|adresse|adres|anschrift|direcci[oó]n|endere[cç]o|indirizzo)'
    r'(?:[ \t]+line[ \t]*\d)?[ \t]*[:#=\-]?[ \t]*'
    r'(?P<value>#?\d{1,6}[A-Za-z]?(?:[-–/]\d{1,6}[A-Za-z]?)?,?[ \t]+[^\W\d_][^|;\n]{1,80})',
    re.I,
)
# Addresses that aren't where someone lives.
_NOT_POSTAL = frozenset({
    'ip', 'ipv4', 'ipv6', 'email', 'e-mail', 'mail', 'mac', 'wallet', 'web', 'memory',
    'server', 'bitcoin', 'btc', 'eth', 'ethereum', 'contract', 'network', 'host', 'return',
    'base', 'bind', 'listen', 'local', 'remote', 'broadcast', 'gateway', 'url', 'link',
    'proxy', 'dns', 'socket', 'device', 'bluetooth', 'hardware', 'start', 'end', 'load',
})
_VALUE_WORDS = 8  # words of a label's value covered, so a joined line doesn't cover on
_HINT = re.compile(
    r'\b(?:address|addr|live[sd]?[ \t]+(?:at|on)|living[ \t]+(?:at|on)|located[ \t]+at'
    r'|ship(?:ping)?[ \t]+to|deliver(?:y|ed)?[ \t]+to|send[ \t]+(?:it[ \t]+)?to'
    r'|mail(?:ing)?[ \t]+(?:it[ \t]+)?to|my[ \t]+(?:place|house|home|apartment|apt)'
    r'|come[ \t]+(?:over[ \t]+)?to|pick[ \t]+(?:me[ \t]+)?up[ \t]+at|meet[ \t]+(?:me[ \t]+)?at'
    r'|drop[ \t]+(?:it[ \t]+)?off[ \t]+at|moved[ \t]+to|stay(?:ing)?[ \t]+at)\b',
    re.I,
)


_LOWER_TYPES = frozenset(word.lower() for word in _TYPES)


def _street_ok(match: re.Match) -> bool:
    """False for a name and type that name something else, as in 3 Google Drive files."""
    words = [word.rstrip('.,').lower() for word in match.group().split()]
    for index in range(2, len(words)):
        if words[index] in _LOWER_TYPES:
            return f'{words[index - 1]} {words[index]}' not in _NOT_STREETS
    return True


def address_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each part of the text that is a postal address."""
    return [(start, end) for start, end, _ in address_parts(text)]


def address_parts(text: str) -> list[tuple[int, int, int]]:
    """(start, end, anchor) of each address in the text. anchor is where its evidence
    starts: the label before a labelled value, else the start itself."""
    if not any(char.isdigit() for char in text):
        return []
    spans = []
    for match in _STREET.finditer(text):
        if _street_ok(match):
            spans.append((*match.span(), match.start()))
    if _HINT.search(text):
        for match in _STREET_LOWER.finditer(text):
            if _street_ok(match):
                spans.append((*match.span(), match.start()))
    for pattern in (_PREFIX_TYPES, _NUMBER_AFTER, _PO_BOX, _REGION_LINE, _CA_POSTAL_ALONE):
        spans.extend((*match.span(), match.start()) for match in pattern.finditer(text))
    spans.extend(
        (*match.span(), match.start())
        for match in _UK_POSTCODE.finditer(text) if not _UK_NOT.search(match.group())
    )
    for match in _LABEL.finditer(text):
        kind = (match.group('kind') or '').strip().lower()
        if kind in _NOT_POSTAL:
            continue
        start = match.start('value')
        value = match.group('value')
        words = list(re.finditer(r'\S+', value))[:_VALUE_WORDS]
        end = start + words[-1].end()
        while end > start and text[end - 1] in ' \t,.':
            end -= 1
        label = match.end('kind') if match.group('kind') else match.start()
        spans.append((start, end, label))
    return spans
