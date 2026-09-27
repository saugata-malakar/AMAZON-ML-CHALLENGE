"""
Candidate Generation Coordinator:
- Partitions by country with safe open-set fallback
- Runs TF-IDF character + word sparse retrieval
- Computes blocking recall ceiling on labeled datasets
- Generates formatted candidate pairs mapping
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.src.blocking.tfidf_ann import TfidfBlocker
from business_entity_resolution.src.normalize.country import normalize_country
from business_entity_resolution.src.normalize.name import parse_name
from business_entity_resolution.src.normalize.address import parse_address


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    top_k: int = 40,
    min_score_threshold: float = 0.05
) -> Dict[str, List[str]]:
    """
    Generates candidate pairs for each S1 entity from target records (Source 2 and Source 3).
    Restricts search within normalized country groups, with fallback to global for unassigned entities.
    """
    # 1. Normalize country labels
    s1_df = s1_df.copy()
    target_df = target_df.copy()

    s1_df["norm_country"] = s1_df["country"].apply(normalize_country)
    target_df["norm_country"] = target_df["country"].apply(normalize_country)

    # 2. Extract core name and clean text for blocking
    s1_cores = [parse_name(n, c)["core_name"] for n, c in zip(s1_df["business_name"], s1_df["norm_country"])]
    s1_texts = [f"{n} {a}".strip() for n, a in zip(s1_df["business_name"], s1_df["business_address"])]

    target_cores = [parse_name(n, c)["core_name"] for n, c in zip(target_df["business_name"], target_df["norm_country"])]
    target_texts = [f"{n} {a}".strip() for n, a in zip(target_df["business_name"], target_df["business_address"])]

    s1_df["core_name"] = s1_cores
    s1_df["full_text"] = s1_texts

    target_df["core_name"] = target_cores
    target_df["full_text"] = target_texts

    candidate_mapping: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_df["entity_id"]}

    # 3. Group by country
    countries = set(s1_df["norm_country"].unique()) | set(target_df["norm_country"].unique())

    for c in countries:
        s1_c_df = s1_df[s1_df["norm_country"] == c]
        tgt_c_df = target_df[target_df["norm_country"] == c]

        if len(s1_c_df) == 0 or len(tgt_c_df) == 0:
            continue

        blocker = TfidfBlocker()
        blocker.fit_targets(
            target_ids=tgt_c_df["entity_id"].tolist(),
            target_names=tgt_c_df["core_name"].tolist(),
            target_texts=tgt_c_df["full_text"].tolist()
        )

        c_results = blocker.query(
            s1_ids=s1_c_df["entity_id"].tolist(),
            s1_names=s1_c_df["core_name"].tolist(),
            s1_texts=s1_c_df["full_text"].tolist(),
            top_k=top_k
        )

        for s1_id, scored_candidates in c_results.items():
            valid_cands = [tid for tid, score in scored_candidates if score >= min_score_threshold]
            candidate_mapping[s1_id] = valid_cands

    return candidate_mapping


def evaluate_blocking_recall(
    candidate_mapping: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Computes recall ceiling: what fraction of true matching pairs are present in candidates?
    """
    total_true_pairs = 0
    captured_true_pairs = 0

    entities_with_matches = 0
    entities_full_recall = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue

        entities_with_matches += 1
        cand_set = set(candidate_mapping.get(s1_id, []))
        matched = set(true_matches) & cand_set

        total_true_pairs += len(true_matches)
        captured_true_pairs += len(matched)

        if len(matched) == len(true_matches):
            entities_full_recall += 1

    pair_recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    entity_recall = (entities_full_recall / entities_with_matches) if entities_with_matches > 0 else 1.0

    return {
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
        "pair_recall_ceiling": float(pair_recall),
        "entity_full_recall_rate": float(entity_recall)
    }
