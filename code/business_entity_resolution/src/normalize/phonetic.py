"""
Phonetic key generation (Soundex / phonetic encoding).
Stdlib-only implementation to handle transliteration variations and typos.
"""

import re
from business_entity_resolution.src.normalize.text import clean_text


SOUNDEX_MAP = {
    'b': '1', 'f': '1', 'p': '1', 'v': '1',
    'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
    'd': '3', 't': '3',
    'l': '4',
    'm': '5', 'n': '5',
    'r': '6',
}


def soundex(token: str) -> str:
    """
    Computes standard American Soundex for an alphanumeric token.
    Example: 'Robert' -> 'R163', 'Rupert' -> 'R163'
    """
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""

    first_char = token[0].upper()
    encoded = [first_char]

    prev_code = SOUNDEX_MAP.get(token[0], "0")

    for char in token[1:]:
        curr_code = SOUNDEX_MAP.get(char, "0")
        if curr_code != "0":
            if curr_code != prev_code:
                encoded.append(curr_code)
            prev_code = curr_code
        else:
            # Vowels/H/W reset previous code
            prev_code = "0"

    # Pad or truncate to 4 characters
    soundex_code = "".join(encoded).replace("0", "")
    return (soundex_code + "000")[:4]


def compute_phonetic_key(text: str) -> str:
    """
    Computes a joined phonetic string for all tokens in the text.
    """
    cleaned = clean_text(text)
    tokens = [t for t in cleaned.split(" ") if t]
    keys = [soundex(t) for t in tokens if soundex(t)]
    return " ".join(keys)
