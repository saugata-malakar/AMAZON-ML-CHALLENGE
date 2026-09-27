"""
Country label normalization.

Per challenge specification:
"Treat country as an open set of string labels: do not hard-code, filter, or one-hot
your pipeline to only {US, India}, and remember that every test entity — France
included — must appear in your submission."

This module canonicalizes common variations without restricting or filtering the set of countries.
"""

from business_entity_resolution.src.normalize.text import clean_text

COUNTRY_SYNONYMS = {
    # United States
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    # India
    "in": "in",
    "ind": "in",
    "india": "in",
    "bharat": "in",
    # France
    "fr": "fr",
    "fra": "fr",
    "france": "fr",
    "republique francaise": "fr",
}


def normalize_country(country_str: str) -> str:
    """
    Normalizes a country string label.
    Preserves any unknown country as a cleaned lowercase string.
    """
    cleaned = clean_text(country_str)
    return COUNTRY_SYNONYMS.get(cleaned, cleaned)
