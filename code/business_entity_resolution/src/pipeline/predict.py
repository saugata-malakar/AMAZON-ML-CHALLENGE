"""
Inference Pipeline:
Generates the two mandatory challenge submission files:
1. output/candidate_pairs.tsv (blocking candidates fed to model)
2. output/matching_results.tsv (final precision-calibrated matches)

Ensures:
- Exactly one row per test S1 entity (including singletons and unseen countries like France)
- Final matches are a strict subset of candidate pairs
- No self-matches, no duplicate IDs, no out-of-vocabulary test IDs
"""

import argparse
from pathlib import Path
import json
import joblib
import pandas as pd

from business_entity_resolution.src.io_utils import read_tsv, write_submission_tsv
from business_entity_resolution.src.blocking.candidates import generate_candidate_pairs
from business_entity_resolution.src.features.build import build_pair_features
from business_entity_resolution.src.decision.assignment import greedy_unique_assignment


def run_inference(
    test_dir: Path,
    artifact_dir: Path,
    output_dir: Path
):
    print("=======================================================")
    print("       BUSINESS ENTITY RESOLUTION - INFERENCE PIPELINE ")
    print("=======================================================\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test data
    print("[1/5] Loading test files...")
    s1_test = read_tsv(test_dir / "test_source1.tsv")
    s2_test = read_tsv(test_dir / "test_source2.tsv")
    s3_test = read_tsv(test_dir / "test_source3.tsv")
    target_test = pd.concat([s2_test, s3_test], ignore_index=True)

    all_test_s1_ids = s1_test["entity_id"].tolist()
    valid_target_ids = set(target_test["entity_id"])

    print(f"  Test S1 records: {len(s1_test):,}")
    print(f"  Test Target records: {len(target_test):,}")

    # 2. Load model and metadata
    print("\n[2/5] Loading trained model and artifacts...")
    model_path = artifact_dir / "matcher_model.joblib"
    meta_path = artifact_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing artifacts in {artifact_dir}. Please run train_and_validate.py first.")

    matcher = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("best_threshold", 0.5))
    top_k = int(metadata.get("top_k", 40))
    print(f"  Loaded model. Using optimal threshold: {threshold:.2f}, top_k: {top_k}")

    # 3. Candidate Generation (Blocking)
    print("\n[3/5] Generating candidate pairs...")
    candidate_mapping = generate_candidate_pairs(s1_test, target_test, top_k=top_k)

    # Write output/candidate_pairs.tsv
    cand_out_file = output_dir / "candidate_pairs.tsv"
    write_submission_tsv(cand_out_file, candidate_mapping, is_candidate_file=True)
    print(f"  -> Candidate pairs written to: {cand_out_file}")

    # 4. Feature Extraction & Scoring
    print("\n[4/5] Building pairwise features and scoring...")
    test_features = build_pair_features(s1_test, target_test, candidate_mapping, ground_truth=None)

    if len(test_features) > 0:
        probs = matcher.predict_proba(test_features)
        s1_ids = test_features["s1_id"].tolist()
        cand_ids = test_features["cand_id"].tolist()

        scored_pairs = []
        for s1, c_id, p in zip(s1_ids, cand_ids, probs):
            if p >= threshold and c_id in valid_target_ids:
                scored_pairs.append((s1, c_id, float(p)))

        # 5. Greedy unique assignment
        matches_mapping = greedy_unique_assignment(scored_pairs, all_test_s1_ids)
    else:
        matches_mapping = {s1: [] for s1 in all_test_s1_ids}

    # Verify subset property: matching_results.tsv must be subset of candidate_pairs.tsv
    for s1, m_list in matches_mapping.items():
        cand_set = set(candidate_mapping.get(s1, []))
        for m in m_list:
            if m not in cand_set:
                raise ValueError(f"Pipeline integrity violation: matched entity {m} for {s1} is not in candidate pairs!")

    # Write output/matching_results.tsv
    match_out_file = output_dir / "matching_results.tsv"
    write_submission_tsv(match_out_file, matches_mapping, is_candidate_file=False)
    print(f"  -> Matching results written to: {match_out_file}")

    # Summary
    total_matched = sum(len(v) for v in matches_mapping.values())
    singletons = sum(1 for v in matches_mapping.values() if len(v) == 0)
    print("\n=======================================================")
    print("  INFERENCE COMPLETED SUCCESSFULLY")
    print(f"  Total S1 entities processed: {len(all_test_s1_ids):,}")
    print(f"  Total matched target pairs: {total_matched:,}")
    print(f"  Predicted singletons (no matches): {singletons:,} ({singletons / len(all_test_s1_ids) * 100:.2f}%)")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test data directory")
    parser.add_argument("--artifact-dir", type=str, default="code/business_entity_resolution/artifacts", help="Path to artifacts directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to submission output directory")
    args = parser.parse_args()
    run_inference(Path(args.test_dir), Path(args.artifact_dir), Path(args.output_dir))
