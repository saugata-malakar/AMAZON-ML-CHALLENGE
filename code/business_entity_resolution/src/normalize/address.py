"""
Address normalization and parsing:
- Abbreviation expansion (Rd -> Road, St -> Street, etc.)
- Postal code / PIN code extraction (US 5-digit, India 6-digit, France 5-digit)
- Landmark extraction (Near, Opp, Behind)
- Number and building extraction
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml
from business_entity_resolution.src.normalize.text import clean_text


_ABBREVIATIONS_BY_COUNTRY: Dict[str, Dict[str, str]] = {}
_LANDMARKS_PATTERNS = [
    r"\b(?:near|opp|opposite|behind|beside|adjacent to|next to)\s+[^,;]+",
]


def _load_abbreviations():
    global _ABBREVIATIONS_BY_COUNTRY
    if _ABBREVIATIONS_BY_COUNTRY:
        return

    config_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "abbreviations"
    for conf_file in config_dir.glob("*.yaml"):
        c_code = conf_file.stem
        try:
            with open(conf_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "address_abbreviations" in data:
                    _ABBREVIATIONS_BY_COUNTRY[c_code] = data["address_abbreviations"]
        except Exception:
            pass

    if "global" not in _ABBREVIATIONS_BY_COUNTRY:
        _ABBREVIATIONS_BY_COUNTRY["global"] = {
            "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
            "dr": "drive", "ln": "lane", "ste": "suite", "apt": "apartment",
            "fl": "floor", "flr": "floor", "bldg": "building", "no": "number"
        }


def extract_postal_code(text: str, country: str = "") -> str:
    """
    Extracts postal code / PIN code based on country or general pattern.
    - India: 6-digit code
    - US: 5-digit or 5+4 code
    - France: 5-digit code
    """
    c = country.lower().strip() if country else ""
    if c == "in":
        # Indian PIN codes: 6 digits starting with 1-9
        m = re.search(r"\b([1-9][0-9]{5})\b", text)
        if m:
            return m.group(1)
    elif c in ("us", "fr"):
        # US ZIP or France Code Postal: 5 digits
        m = re.search(r"\b([0-9]{5})\b", text)
        if m:
            return m.group(1)

    # General fallback: check for 6-digit or 5-digit
    m6 = re.search(r"\b([1-9][0-9]{5})\b", text)
    if m6:
        return m6.group(1)
    m5 = re.search(r"\b([0-9]{5})\b", text)
    if m5:
        return m5.group(1)

    return ""


def extract_landmarks(text: str) -> List[str]:
    """
    Extracts landmark phrases like 'near sbi atm', 'opposite bus stand'.
    """
    landmarks = []
    for pat in _LANDMARKS_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            cleaned = clean_text(m)
            if cleaned:
                landmarks.append(cleaned)
    return landmarks


def expand_address_abbreviations(text: str, country: str = "") -> str:
    """
    Expands common address abbreviations using country-specific and global rules.
    """
    _load_abbreviations()
    country_key = country.lower().strip() if country else "global"
    mapping = dict(_ABBREVIATIONS_BY_COUNTRY.get("global", {}))
    if country_key in _ABBREVIATIONS_BY_COUNTRY:
        mapping.update(_ABBREVIATIONS_BY_COUNTRY[country_key])

    words = text.split(" ")
    expanded = [mapping.get(w, w) for w in words]
    return " ".join(expanded)


def parse_address(addr_str: str, country: str = "") -> Dict:
    """
    Parses a raw address into structured components.
    """
    cleaned = clean_text(addr_str)
    postal_code = extract_postal_code(cleaned, country)
    landmarks = extract_landmarks(cleaned)
    expanded = expand_address_abbreviations(cleaned, country)

    # Extract all numeric tokens (house numbers, suite numbers, PINs)
    numbers = re.findall(r"\b[0-9]+\b", cleaned)

    tokens = [t for t in expanded.split(" ") if t]

    return {
        "raw_address": addr_str,
        "clean_address": cleaned,
        "expanded_address": expanded,
        "postal_code": postal_code,
        "landmarks": landmarks,
        "numbers": numbers,
        "tokens": tokens,
    }
