"""
Assignment and Conflict Resolution:
Since Source 1 is deduplicated and each S2/S3 record represents a single real-world entity,
each target record can belong to at most ONE Source 1 entity.
This module enforces competitive greedy assignment based on match confidence probabilities.
"""

from typing import Dict, List, Set, Tuple


def greedy_unique_assignment(
    scored_pairs: List[Tuple[str, str, float]],
    all_s1_ids: List[str]
) -> Dict[str, List[str]]:
    """
    Assigns each target record (S2/S3) to at most ONE S1 entity,
    resolving conflicts in descending order of predicted probability.
    
    Args:
        scored_pairs: List of (s1_id, target_id, probability) that passed the threshold.
        all_s1_ids: List of all S1 entity IDs to guarantee every entity appears in output.
        
    Returns:
        Mapping: s1_id -> list of assigned target IDs (ordered).
    """
    # Sort pairs by probability descending
    sorted_pairs = sorted(scored_pairs, key=lambda x: x[2], reverse=True)

    assigned_mapping: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}
    claimed_targets: Set[str] = set()

    for s1_id, target_id, _ in sorted_pairs:
        if target_id not in claimed_targets:
            claimed_targets.add(target_id)
            if s1_id in assigned_mapping:
                assigned_mapping[s1_id].append(target_id)

    return assigned_mapping
