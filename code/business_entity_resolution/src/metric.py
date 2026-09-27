"""
Evaluation metric module for Business Entity Resolution Challenge.

Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Metric computation rules per specification:
1. Computed per Source 1 entity, then macro-averaged across all Source 1 entities.
2. Singletons (entities with no true matches):
   - Correctly predicted empty list: score = 1.0
   - Any predicted match: score = 0.0
3. Non-singletons with empty prediction: score = 0.0
4. When true positives = 0: score = 0.0
"""

from typing import Dict, List, Set, Union
import numpy as np


def compute_entity_f05(
    predicted_ids: Union[List[str], Set[str]],
    true_ids: Union[List[str], Set[str]]
) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    """
    pred_set = set(p for p in predicted_ids if p)
    true_set = set(t for t in true_ids if t)

    # Case 1: True singleton (no true matches)
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Case 2: True non-singleton, but prediction is empty
    if len(pred_set) == 0:
        return 0.0

    # Case 3: Both non-empty
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def compute_macro_f05(
    predictions: Dict[str, Union[List[str], Set[str]]],
    ground_truth: Dict[str, Union[List[str], Set[str]]]
) -> float:
    """
    Computes Macro-Averaged F_0.5 score across all Source 1 entities in ground truth.
    
    Args:
        predictions: Dict mapping s1_id -> list/set of predicted matching IDs
        ground_truth: Dict mapping s1_id -> list/set of true matching IDs
        
    Returns:
        Macro-averaged F_0.5 score as a float
    """
    if not ground_truth:
        raise ValueError("Ground truth dictionary is empty.")

    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        score = compute_entity_f05(pred_matches, true_matches)
        scores.append(score)

    return float(np.mean(scores))
