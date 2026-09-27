"""
Feature Matrix Builder:
Constructs tabular pairwise features for candidate pairs generated during blocking.
"""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from business_entity_resolution.src.normalize.country import normalize_country
from business_entity_resolution.src.normalize.name import parse_name
from business_entity_resolution.src.normalize.address import parse_address
from business_entity_resolution.src.features.name_features import compute_name_features
from business_entity_resolution.src.features.address_features import compute_address_features


def build_pair_features(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_mapping: Dict[str, List[str]],
    ground_truth: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Constructs feature vectors for all candidate pairs (s1_id, candidate_id).
    
    If ground_truth is provided, attaches binary target column 'label' (1 for match, 0 for negative).
    """
    # 1. Pre-index and parse records for fast lookup
    print(f"Pre-parsing {len(s1_df)} S1 records...")
    s1_parsed = {}
    for _, row in s1_df.iterrows():
        eid = row["entity_id"]
        c = normalize_country(row.get("country", ""))
        s1_parsed[eid] = {
            "name": parse_name(row.get("business_name", ""), c),
            "addr": parse_address(row.get("business_address", ""), c),
            "country": c
        }

    print(f"Pre-parsing {len(target_df)} target records...")
    tgt_parsed = {}
    for _, row in target_df.iterrows():
        eid = row["entity_id"]
        c = normalize_country(row.get("country", ""))
        tgt_parsed[eid] = {
            "name": parse_name(row.get("business_name", ""), c),
            "addr": parse_address(row.get("business_address", ""), c),
            "country": c
        }

    # 2. Iterate through candidate pairs
    rows = []
    total_pairs = sum(len(cands) for cands in candidate_mapping.values())
    print(f"Building features across {total_pairs:,} candidate pairs...")

    for s1_id, cands in candidate_mapping.items():
        if s1_id not in s1_parsed:
            continue
        p1 = s1_parsed[s1_id]
        true_set = set(ground_truth.get(s1_id, [])) if ground_truth is not None else None

        for rank_idx, cand_id in enumerate(cands):
            if cand_id not in tgt_parsed:
                continue
            p2 = tgt_parsed[cand_id]

            # Pair features
            n_feats = compute_name_features(p1["name"], p2["name"])
            a_feats = compute_address_features(p1["addr"], p2["addr"])

            country_match = 1.0 if (p1["country"] and p2["country"] and p1["country"] == p2["country"]) else 0.0
            is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
            is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

            row_data = {
                "s1_id": s1_id,
                "cand_id": cand_id,
                "cand_rank": float(rank_idx),
                "is_s2": is_s2,
                "is_s3": is_s3,
                "country_match": country_match,
                **n_feats,
                **a_feats,
            }

            if true_set is not None:
                row_data["label"] = 1 if cand_id in true_set else 0

            rows.append(row_data)

    df_feats = pd.DataFrame(rows)
    print(f"Feature matrix built successfully: shape {df_feats.shape}")
    return df_feats
