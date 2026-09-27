"""
Unit tests for text, name, address, and country normalization.
"""

from business_entity_resolution.src.normalize.text import clean_text, strip_accents
from business_entity_resolution.src.normalize.country import normalize_country
from business_entity_resolution.src.normalize.name import parse_name, split_dba_variants, extract_legal_suffix
from business_entity_resolution.src.normalize.address import parse_address, extract_postal_code, expand_address_abbreviations
from business_entity_resolution.src.normalize.phonetic import soundex, compute_phonetic_key


def test_strip_accents_and_clean_text():
    raw = "Café & Bistrot de l'Étoile"
    cleaned = clean_text(raw)
    assert "cafe" in cleaned
    assert "and" in cleaned  # & converted to and
    assert "etoile" in cleaned
    assert "é" not in cleaned


def test_open_set_country():
    assert normalize_country("USA") == "us"
    assert normalize_country("United States") == "us"
    assert normalize_country("India") == "in"
    assert normalize_country("France") == "fr"
    # Open-set unseen country should NOT throw and be normalized cleanly
    assert normalize_country("Germany") == "germany"


def test_split_dba_variants():
    variants = split_dba_variants("Acme Corporation d/b/a Fast Delivery Services")
    assert "acme corporation" in variants
    assert "fast delivery services" in variants


def test_extract_legal_suffix():
    core, suffix = extract_legal_suffix("Acme Supplies Pvt Ltd", country="in")
    assert core == "acme supplies"
    assert "pvt ltd" in suffix or "ltd" in suffix

    core_fr, suffix_fr = extract_legal_suffix("Dupont Consulting SARL", country="fr")
    assert core_fr == "dupont consulting"
    assert suffix_fr == "sarl"


def test_address_parsing():
    addr = "123 Main St., Apt 4B, New York, NY 10001"
    parsed = parse_address(addr, country="us")
    assert parsed["postal_code"] == "10001"
    assert "street" in parsed["expanded_address"]
    assert "apartment" in parsed["expanded_address"]

    addr_in = "Plot 42, Opp SBI Bank, Near Bus Stand, Bengaluru 560001"
    parsed_in = parse_address(addr_in, country="in")
    assert parsed_in["postal_code"] == "560001"
    assert any("near bus stand" in lm or "opp sbi bank" in lm for lm in parsed_in["landmarks"])


def test_phonetic_encoding():
    # Robert and Rupert should produce the same soundex code R163
    assert soundex("Robert") == soundex("Rupert")
    key = compute_phonetic_key("Tech Solutions")
    assert len(key) > 0
