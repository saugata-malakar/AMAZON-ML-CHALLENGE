"""
Business name normalization and feature parsing:
- DBA (Doing Business As) and trade name splitting
- Legal suffix extraction (pvt ltd, corp, inc, llc, sarl, etc.)
- Variant generation
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml
from business_entity_resolution.src.normalize.text import clean_text


DBA_PATTERNS = [
    r"\bd/?b/?a\b",
    r"\bt/?a\b",
    r"\btrading as\b",
    r"\bformerly known as\b",
    r"\bf/?k/?a\b",
    r"\bformerly\b",
    r"\baka\b",
]

_LEGAL_SUFFIXES_BY_COUNTRY: Dict[str, List[str]] = {}


def _load_legal_suffixes():
    global _LEGAL_SUFFIXES_BY_COUNTRY
    if _LEGAL_SUFFIXES_BY_COUNTRY:
        return

    config_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "abbreviations"
    for conf_file in config_dir.glob("*.yaml"):
        c_code = conf_file.stem
        try:
            with open(conf_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "legal_suffixes" in data:
                    # Sort suffixes by length descending so multi-word (e.g. 'private limited') matches before 'limited'
                    suffixes = sorted(data["legal_suffixes"], key=lambda x: len(x), reverse=True)
                    _LEGAL_SUFFIXES_BY_COUNTRY[c_code] = suffixes
        except Exception:
            pass

    if "global" not in _LEGAL_SUFFIXES_BY_COUNTRY:
        _LEGAL_SUFFIXES_BY_COUNTRY["global"] = [
            "private limited", "pvt ltd", "corporation", "incorporated",
            "limited", "corp", "inc", "ltd", "llc", "llp", "company", "co"
        ]


def split_dba_variants(name: str) -> List[str]:
    """
    Splits names containing DBA, trade names, or parenthesized alternative names.
    Example: 'Acme Supplies dba Fast Delivery' -> ['acme supplies', 'fast delivery']
    """
    if not name:
        return []

    # Handle text inside parentheses
    variants = []
    parentheses_matches = re.findall(r"\((.*?)\)", name)
    cleaned_without_parens = re.sub(r"\(.*?\)", " ", name).strip()

    variants.append(cleaned_without_parens)
    for p in parentheses_matches:
        if p.strip():
            variants.append(p.strip())

    combined_pattern = "|".join(DBA_PATTERNS)
    final_variants = []
    for var in variants:
        parts = re.split(combined_pattern, var, flags=re.IGNORECASE)
        for part in parts:
            c = clean_text(part)
            if c and c not in final_variants:
                final_variants.append(c)

    return final_variants or [clean_text(name)]


def extract_legal_suffix(name_str: str, country: str = "") -> Tuple[str, str]:
    """
    Strips legal suffix from the end of a business name.
    Returns (core_name, legal_suffix).
    Example: 'amazon data services india pvt ltd' -> ('amazon data services india', 'pvt ltd')
    """
    _load_legal_suffixes()
    cleaned = clean_text(name_str)
    if not cleaned:
        return "", ""

    country_key = country.lower().strip() if country else "global"
    suffix_list = _LEGAL_SUFFIXES_BY_COUNTRY.get(country_key, _LEGAL_SUFFIXES_BY_COUNTRY.get("global", []))
    # Also fallback to global suffixes
    if country_key != "global" and "global" in _LEGAL_SUFFIXES_BY_COUNTRY:
        all_suffixes = list(dict.fromkeys(suffix_list + _LEGAL_SUFFIXES_BY_COUNTRY["global"]))
    else:
        all_suffixes = suffix_list

    for suffix in all_suffixes:
        pattern = r"\b" + re.escape(suffix) + r"\s*$"
        if re.search(pattern, cleaned):
            core = re.sub(pattern, "", cleaned).strip()
            if core:  # only strip if core name remains non-empty
                return core, suffix

    return cleaned, ""


def parse_name(name_str: str, country: str = "") -> Dict:
    """
    Parses a raw business name string into structured normalized attributes.
    """
    variants = split_dba_variants(name_str)
    primary_variant = variants[0] if variants else clean_text(name_str)
    core_name, legal_suffix = extract_legal_suffix(primary_variant, country)

    all_cores = []
    for var in variants:
        c, _ = extract_legal_suffix(var, country)
        if c and c not in all_cores:
            all_cores.append(c)

    return {
        "raw_name": name_str,
        "clean_name": clean_text(name_str),
        "core_name": core_name or clean_text(name_str),
        "legal_suffix": legal_suffix,
        "name_variants": all_cores or [core_name or clean_text(name_str)],
    }
