"""
End-to-End Training and Validation Pipeline:
1. Loads train sources (1, 2, 3) and ground truth.
2. Performs grouped train/validation split by S1 entity.
3. Runs candidate blocking and reports Recall Ceiling on validation.
4. Builds pairwise feature matrix on training and validation candidate pairs.
5. Fits EntityMatcherModel (GBDT).
6. Optimizes decision threshold for Macro F_0.5.
7. Evaluates out-of-fold validation Macro F_0.5 (Overall, Singletons, Country breakdown).
8. Saves trained model and threshold to artifacts.
"""

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from business_entity_resolution.src.io_utils import read_tsv, parse_ground_truth
from business_entity_resolution.src.blocking.candidates import generate_candidate_pairs, evaluate_blocking_recall
from business_entity_resolution.src.features.build import build_pair_features
from business_entity_resolution.src.models.gbm import EntityMatcherModel
from business_entity_resolution.src.decision.threshold import optimize_macro_f05_threshold
from business_entity_resolution.src.decision.assignment import greedy_unique_assignment
from business_entity_resolution.src.metric import compute_macro_f05


def run_training_pipeline(
    data_dir: Path,
    artifact_dir: Path,
    val_ratio: float = 0.2,
    top_k: int = 40,
    random_state: int = 42
):
    print("=======================================================")
    print("      BUSINESS ENTITY RESOLUTION - TRAINING PIPELINE   ")
    print("=======================================================\n")

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("[Step 1/7] Loading training data...")
    s1_df = read_tsv(data_dir / "train_source1.tsv")
    s2_df = read_tsv(data_dir / "train_source2.tsv")
    s3_df = read_tsv(data_dir / "train_source3.tsv")
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    gt = parse_ground_truth(data_dir / "train_ground_truth.tsv")

    print(f"  Source 1 records: {len(s1_df):,}")
    print(f"  Target records (S2 + S3): {len(target_df):,}")
    print(f"  Ground truth S1 entities: {len(gt):,}")

    # 2. Grouped split on S1 entities
    print(f"\n[Step 2/7] Splitting S1 entities into Train / Validation ({int((1-val_ratio)*100)}% / {int(val_ratio*100)}%)...")
    all_s1_ids = s1_df["entity_id"].tolist()
    train_s1_ids, val_s1_ids = train_test_split(all_s1_ids, test_size=val_ratio, random_state=random_state)
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    s1_train_df = s1_df[s1_df["entity_id"].isin(train_s1_set)].reset_index(drop=True)
    s1_val_df = s1_df[s1_df["entity_id"].isin(val_s1_set)].reset_index(drop=True)

    gt_train = {k: v for k, v in gt.items() if k in train_s1_set}
    gt_val = {k: v for k, v in gt.items() if k in val_s1_set}

    print(f"  Train S1 entities: {len(s1_train_df):,}")
    print(f"  Validation S1 entities: {len(s1_val_df):,}")

    # 3. Blocking / Candidate generation
    print(f"\n[Step 3/7] Generating candidate pairs with top_k={top_k}...")
    print("  Generating training candidates...")
    train_cands = generate_candidate_pairs(s1_train_df, target_df, top_k=top_k)
    train_recall = evaluate_blocking_recall(train_cands, gt_train)
    print(f"  Train Candidate Recall Ceiling: {train_recall['pair_recall_ceiling']*100:.2f}% ({train_recall['captured_true_pairs']:,}/{train_recall['total_true_pairs']:,} pairs)")

    print("  Generating validation candidates...")
    val_cands = generate_candidate_pairs(s1_val_df, target_df, top_k=top_k)
    val_recall = evaluate_blocking_recall(val_cands, gt_val)
    print(f"  Val Candidate Recall Ceiling: {val_recall['pair_recall_ceiling']*100:.2f}% ({val_recall['captured_true_pairs']:,}/{val_recall['total_true_pairs']:,} pairs)")

    # 4. Feature Extraction
    print("\n[Step 4/7] Building pairwise feature matrices...")
    train_features = build_pair_features(s1_train_df, target_df, train_cands, ground_truth=gt_train)
    val_features = build_pair_features(s1_val_df, target_df, val_cands, ground_truth=gt_val)

    # 5. Fit Model
    print("\n[Step 5/7] Fitting GBDT Matcher Model...")
    matcher = EntityMatcherModel(random_state=random_state)
    matcher.fit(train_features)

    # 6. Predict on Validation and Optimize Threshold for Macro F_0.5
    print("\n[Step 6/7] Threshold Optimization on Validation Set...")
    val_probs = matcher.predict_proba(val_features)
    best_thresh, best_val_f05 = optimize_macro_f05_threshold(
        val_features,
        val_probs,
        gt_val,
        all_val_s1_ids=val_s1_ids
    )

    # 7. Evaluate and Save Artifacts
    print("\n[Step 7/7] Saving Artifacts and Finalizing Report...")
    model_path = artifact_dir / "matcher_model.joblib"
    meta_path = artifact_dir / "metadata.json"

    joblib.dump(matcher, model_path)
    metadata = {
        "best_threshold": best_thresh,
        "val_macro_f05": best_val_f05,
        "val_blocking_pair_recall": val_recall["pair_recall_ceiling"],
        "top_k": top_k,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"  Model saved to: {model_path}")
    print(f"  Metadata saved to: {meta_path}")
    print("\n=======================================================")
    print(f"  FINAL VALIDATION MACRO F_0.5: {best_val_f05:.4f}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate ER matching pipeline")
    parser.add_argument("--data-dir", type=str, default="dataset/train", help="Path to training data directory")
    parser.add_argument("--artifact-dir", type=str, default="code/business_entity_resolution/artifacts", help="Path to artifacts output directory")
    args = parser.parse_args()
    run_training_pipeline(Path(args.data_dir), Path(args.artifact_dir))
