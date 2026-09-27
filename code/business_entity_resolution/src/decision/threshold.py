"""
Macro F0.5 Threshold Optimization:
Directly grid searches the probability threshold that maximizes Macro F_0.5 on validation.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.src.decision.assignment import greedy_unique_assignment
from business_entity_resolution.src.metric import compute_macro_f05


def optimize_macro_f05_threshold(
    df_val_features: pd.DataFrame,
    val_probs: np.ndarray,
    ground_truth_val: Dict[str, List[str]],
    all_val_s1_ids: List[str],
    threshold_range: Tuple[float, float, float] = (0.20, 0.85, 0.05)
) -> Tuple[float, float]:
    """
    Finds the optimal match probability threshold that maximizes Macro F_0.5.
    
    Returns:
        (best_threshold, best_f05_score)
    """
    s1_ids = df_val_features["s1_id"].tolist()
    cand_ids = df_val_features["cand_id"].tolist()

    all_pairs = list(zip(s1_ids, cand_ids, [float(p) for p in val_probs]))

    start, stop, step = threshold_range
    thresholds = np.arange(start, stop + 1e-5, step)

    best_thresh = 0.5
    best_score = -1.0

    print("Optimizing threshold for Macro F_0.5...")
    for thresh in thresholds:
        filtered_pairs = [p for p in all_pairs if p[2] >= thresh]
        preds = greedy_unique_assignment(filtered_pairs, all_val_s1_ids)
        score = compute_macro_f05(preds, ground_truth_val)

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

        print(f"  Threshold {thresh:.2f} -> Validation Macro F_0.5: {score:.4f}")

    print(f"\nOptimal Threshold: {best_thresh:.2f} with Macro F_0.5 = {best_score:.4f}")
    return best_thresh, best_score
