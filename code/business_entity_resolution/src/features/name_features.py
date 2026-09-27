"""
Pairwise Name Feature Engineering:
- Character 3-gram Jaccard similarity
- Token Jaccard, Token Set, Token Sort ratios
- Normalized Levenshtein ratio
- First token and last token exact matches
- Acronym / Initials match
- Substring containment
- Legal suffix consistency
- Phonetic Soundex match
"""

from typing import Dict, List, Set
from difflib import SequenceMatcher
from business_entity_resolution.src.normalize.phonetic import soundex


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes Jaccard similarity over character n-grams."""
    if not s1 or not s2:
        return 0.0
    ngrams1 = {s1[i:i+n] for i in range(max(0, len(s1) - n + 1))}
    ngrams2 = {s2[i:i+n] for i in range(max(0, len(s2) - n + 1))}
    if not ngrams1 or not ngrams2:
        return 1.0 if s1 == s2 else 0.0
    inter = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return inter / union if union > 0 else 0.0


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Computes Jaccard similarity over word tokens."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """Normalized similarity between alphabetically sorted token strings."""
    tokens1 = sorted(s1.split())
    tokens2 = sorted(s2.split())
    sorted_s1 = " ".join(tokens1)
    sorted_s2 = " ".join(tokens2)
    return SequenceMatcher(None, sorted_s1, sorted_s2).ratio()


def token_set_ratio(s1: str, s2: str) -> float:
    """Token set ratio: compares common tokens with remaining tokens."""
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    common = tokens1 & tokens2
    diff1 = tokens1 - common
    diff2 = tokens2 - common

    t0 = " ".join(sorted(common))
    t1 = (" ".join(sorted(common)) + " " + " ".join(sorted(diff1))).strip()
    t2 = (" ".join(sorted(common)) + " " + " ".join(sorted(diff2))).strip()

    scores = [
        SequenceMatcher(None, t0, t1).ratio() if t0 else 0.0,
        SequenceMatcher(None, t0, t2).ratio() if t0 else 0.0,
        SequenceMatcher(None, t1, t2).ratio(),
    ]
    return max(scores)


def compute_name_features(parsed1: Dict, parsed2: Dict) -> Dict[str, float]:
    """
    Computes all pairwise name similarity features between two parsed name dicts.
    """
    core1 = parsed1.get("core_name", "")
    core2 = parsed2.get("core_name", "")

    tokens1 = core1.split()
    tokens2 = core2.split()

    # Best variant match
    best_char_jaccard = char_ngram_jaccard(core1, core2, 3)
    best_lev_ratio = SequenceMatcher(None, core1, core2).ratio()

    # Acronym match: e.g. 'sbi' vs 'state bank of india'
    acronym1 = "".join(t[0] for t in tokens1 if t)
    acronym2 = "".join(t[0] for t in tokens2 if t)
    acronym_match = 1.0 if (acronym1 and (acronym1 == core2 or acronym1 == acronym2)) or (acronym2 and (acronym2 == core1 or acronym2 == acronym1)) else 0.0

    # Substring containment
    containment = 1.0 if (core1 and core2 and (core1 in core2 or core2 in core1)) else 0.0

    # First and last token matches
    first_token_match = 1.0 if (tokens1 and tokens2 and tokens1[0] == tokens2[0]) else 0.0
    last_token_match = 1.0 if (tokens1 and tokens2 and tokens1[-1] == tokens2[-1]) else 0.0

    # Legal suffix match
    suff1 = parsed1.get("legal_suffix", "")
    suff2 = parsed2.get("legal_suffix", "")
    if not suff1 and not suff2:
        suffix_match = 0.5  # both missing
    elif suff1 and suff2:
        suffix_match = 1.0 if suff1 == suff2 else 0.0
    else:
        suffix_match = 0.2  # one missing

    # Phonetic first token match
    soundex1 = soundex(tokens1[0]) if tokens1 else ""
    soundex2 = soundex(tokens2[0]) if tokens2 else ""
    phonetic_match = 1.0 if (soundex1 and soundex1 == soundex2) else 0.0

    # Length ratio
    len1, len2 = len(core1), len(core2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    return {
        "name_char_jaccard_3": best_char_jaccard,
        "name_lev_ratio": best_lev_ratio,
        "name_token_jaccard": token_jaccard(tokens1, tokens2),
        "name_token_sort_ratio": token_sort_ratio(core1, core2),
        "name_token_set_ratio": token_set_ratio(core1, core2),
        "name_containment": containment,
        "name_acronym_match": acronym_match,
        "name_first_token_match": first_token_match,
        "name_last_token_match": last_token_match,
        "name_suffix_match": suffix_match,
        "name_phonetic_match": phonetic_match,
        "name_length_ratio": len_ratio,
    }
