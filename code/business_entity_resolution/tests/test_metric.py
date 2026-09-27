"""
Unit tests for metric computation.
Validates the official example and edge cases from the challenge specification.
"""

import pytest
import math
from business_entity_resolution.src.metric import compute_entity_f05, compute_macro_f05


def test_pdf_worked_example():
    """
    Example from Page 6 of the PDF:
    Model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
    Ground truth says S1-00001 matches [S2-00047, S3-00812]
    Precision = 2/3, Recall = 2/2 = 1.0
    F_0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
    """
    predicted = ["S2-00047", "S2-00193", "S3-00812"]
    truth = ["S2-00047", "S3-00812"]

    score = compute_entity_f05(predicted, truth)
    # 1.25 * (2/3) * 1.0 / (0.25 * (2/3) + 1.0) = (5/6) / (7/6) = 5/7 = 0.7142857142857143
    expected = 5.0 / 7.0
    assert math.isclose(score, expected, rel_tol=1e-5)
    assert round(score, 3) == 0.714


def test_singleton_correct():
    """Singletons correctly predicted as empty should score 1.0"""
    score = compute_entity_f05([], [])
    assert score == 1.0


def test_singleton_false_positive():
    """Singletons with any predicted match should score 0.0"""
    score = compute_entity_f05(["S2-99999"], [])
    assert score == 0.0


def test_non_singleton_empty_prediction():
    """Non-singleton where model predicts nothing should score 0.0"""
    score = compute_entity_f05([], ["S2-00001"])
    assert score == 0.0


def test_perfect_match():
    """Exact match should score 1.0"""
    score = compute_entity_f05(["S2-00001", "S3-00002"], ["S2-00001", "S3-00002"])
    assert score == 1.0


def test_completely_wrong_match():
    """Completely wrong predictions should score 0.0"""
    score = compute_entity_f05(["S2-99999"], ["S2-00001"])
    assert score == 0.0


def test_macro_average():
    """Test macro average across multiple entities"""
    preds = {
        "S1-001": ["S2-00047", "S2-00193", "S3-00812"], # 5/7 = ~0.7143
        "S1-002": [],                                      # true singleton, predicted empty -> 1.0
        "S1-003": ["S2-11111"],                            # true singleton, predicted false merge -> 0.0
    }
    truth = {
        "S1-001": ["S2-00047", "S3-00812"],
        "S1-002": [],
        "S1-003": [],
    }

    macro = compute_macro_f05(preds, truth)
    expected_macro = (5.0 / 7.0 + 1.0 + 0.0) / 3.0
    assert math.isclose(macro, expected_macro, rel_tol=1e-5)
