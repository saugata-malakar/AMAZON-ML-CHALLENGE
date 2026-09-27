"""
Unit tests for decision and unique assignment modules.
"""

from business_entity_resolution.src.decision.assignment import greedy_unique_assignment
from business_entity_resolution.src.decision.threshold import optimize_macro_f05_threshold
import pandas as pd
import numpy as np


def test_greedy_unique_assignment():
    all_s1 = ["S1-001", "S1-002", "S1-003"]
    # Both S1-001 and S1-002 claim S2-999, but S1-001 has higher probability (0.9 vs 0.7)
    pairs = [
        ("S1-001", "S2-999", 0.90),
        ("S1-002", "S2-999", 0.70),
        ("S1-002", "S3-111", 0.60),
    ]

    assigned = greedy_unique_assignment(pairs, all_s1)

    assert assigned["S1-001"] == ["S2-999"]
    assert assigned["S1-002"] == ["S3-111"]  # S2-999 was already claimed by S1-001
    assert assigned["S1-003"] == []  # singleton guaranteed present in dict
