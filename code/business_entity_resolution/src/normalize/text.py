"""
Text normalization utilities:
- Unicode NFKD decomposition and accent removal (essential for France and multilingual robustness)
- Punctuation canonicalization (& -> and)
- Whitespace cleanup
"""

import re
import unicodedata
from typing import List


def strip_accents(text: str) -> str:
    """
    Strips accents and diacritics using Unicode NFKD normalization.
    Example: 'Café de la Paix' -> 'Cafe de la Paix'
    """
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """
    General text cleaning:
    1. Strip accents
    2. Lowercase
    3. Canonicalize & / + to 'and'
    4. Remove hyphens/dots in abbreviations (e.g. U.S.A. -> usa)
    5. Clean non-alphanumeric characters to spaces
    6. Normalize whitespace
    """
    if not text:
        return ""

    s = strip_accents(text).lower()

    # Canonicalize symbols
    s = re.sub(r"[&+]|\band\b", " and ", s)
    s = re.sub(r"[@]", " at ", s)

    # Remove dots inside acronyms/initials like U.S.A. -> usa
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)

    # Replace punctuation and special characters with spaces
    s = re.sub(r"[^\w\s]", " ", s)

    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    """Splits cleaned text into alphanumeric tokens."""
    cleaned = clean_text(text)
    return [t for t in cleaned.split(" ") if t]
