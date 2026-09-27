"""
Pairwise Address Feature Engineering:
- Token and Character similarity
- Postal code / PIN code exact and prefix matching
- Street / Building number overlap
- Landmark token overlap
- Address length ratio
"""

from typing import Dict, List, Set
from difflib import SequenceMatcher
from business_entity_resolution.src.features.name_features import char_ngram_jaccard, token_jaccard, token_sort_ratio, token_set_ratio


def compute_address_features(parsed1: Dict, parsed2: Dict) -> Dict[str, float]:
    """
    Computes all pairwise address similarity features between two parsed address dicts.
    """
    addr1 = parsed1.get("expanded_address", "")
    addr2 = parsed2.get("expanded_address", "")

    tokens1 = parsed1.get("tokens", [])
    tokens2 = parsed2.get("tokens", [])

    # Postal code / PIN comparison
    pin1 = parsed1.get("postal_code", "")
    pin2 = parsed2.get("postal_code", "")

    if pin1 and pin2:
        pin_both_present = 1.0
        if pin1 == pin2:
            pin_match = 1.0
        elif len(pin1) >= 3 and len(pin2) >= 3 and pin1[:3] == pin2[:3]:
            pin_match = 0.5  # regional prefix match
        else:
            pin_match = 0.0
    else:
        pin_both_present = 0.0
        pin_match = -1.0  # missing flag

    # Numeric tokens comparison (building/street numbers)
    nums1 = set(parsed1.get("numbers", []))
    nums2 = set(parsed2.get("numbers", []))
    num_inter = len(nums1 & nums2)
    num_union = len(nums1 | nums2)
    num_jaccard = (num_inter / num_union) if num_union > 0 else (1.0 if not nums1 and not nums2 else 0.0)

    # Landmark overlap
    lms1 = parsed1.get("landmarks", [])
    lms2 = parsed2.get("landmarks", [])
    lm_overlap = 0.0
    if lms1 and lms2:
        lm_tokens1 = set(" ".join(lms1).split())
        lm_tokens2 = set(" ".join(lms2).split())
        inter = len(lm_tokens1 & lm_tokens2)
        union = len(lm_tokens1 | lm_tokens2)
        lm_overlap = inter / union if union > 0 else 0.0

    # Address length ratio
    len1, len2 = len(addr1), len(addr2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    return {
        "addr_char_jaccard_3": char_ngram_jaccard(addr1, addr2, 3),
        "addr_token_jaccard": token_jaccard(tokens1, tokens2),
        "addr_token_sort_ratio": token_sort_ratio(addr1, addr2),
        "addr_token_set_ratio": token_set_ratio(addr1, addr2),
        "addr_pin_match": pin_match,
        "addr_pin_both_present": pin_both_present,
        "addr_num_jaccard": num_jaccard,
        "addr_num_inter_count": float(num_inter),
        "addr_landmark_overlap": lm_overlap,
        "addr_length_ratio": len_ratio,
    }
