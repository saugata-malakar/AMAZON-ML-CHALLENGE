"""
Submission Validation Helper Script (stdlib only, no dependencies).
Validates matching_results.tsv and candidate_pairs.tsv against challenge rules.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_tsv_raw(filepath: Path) -> List[Tuple[str, str]]:
    """Reads TSV into list of (col1, col2) without pandas."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line and line_num > 1:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                rows.append((parts[0].strip(), ""))
            elif len(parts) == 2:
                rows.append((parts[0].strip(), parts[1].strip()))
            else:
                rows.append((parts[0].strip(), "\t".join(parts[1:]).strip()))
    return rows


def get_test_entity_ids(test_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Reads valid S1 IDs and valid target (S2 + S3) IDs from test source files."""
    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    for p in [s1_file, s2_file, s3_file]:
        if not p.exists():
            raise FileNotFoundError(f"Required test source file missing: {p}")

    s1_ids = set()
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.split("\t")
            if parts and parts[0].strip():
                s1_ids.add(parts[0].strip())

    target_ids = set()
    for tf in [s2_file, s3_file]:
        with open(tf, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.split("\t")
                if parts and parts[0].strip():
                    target_ids.add(parts[0].strip())

    return s1_ids, target_ids


def validate(matching_path: Path, candidate_path: Path, test_dir: Path) -> bool:
    issues = []

    try:
        valid_s1_ids, valid_target_ids = get_test_entity_ids(test_dir)
    except Exception as e:
        print(f"Validation failed during test directory inspection: {e}")
        return False

    # Check 1: matching_results.tsv
    if not matching_path.exists():
        issues.append(f"Matching file does not exist: {matching_path}")
        matching_rows = []
    else:
        matching_rows = read_tsv_raw(matching_path)

    # Check 2: candidate_pairs.tsv
    if not candidate_path.exists():
        issues.append(f"Candidate file does not exist: {candidate_path}")
        candidate_rows = []
    else:
        candidate_rows = read_tsv_raw(candidate_path)

    if not matching_rows or not candidate_rows:
        for idx, iss in enumerate(issues, 1):
            print(f"[{idx}] {iss}")
        return False

    # Check Headers
    if matching_rows[0] != ("source1_entity_id", "matched_entity_ids"):
        issues.append(f"matching_results.tsv header must be 'source1_entity_id\\tmatched_entity_ids', got {matching_rows[0]}")

    if candidate_rows[0] != ("source1_entity_id", "candidate_entity_ids"):
        issues.append(f"candidate_pairs.tsv header must be 'source1_entity_id\\tcandidate_entity_ids', got {candidate_rows[0]}")

    # Inspect matching rows
    seen_s1_match = set()
    match_map: Dict[str, Set[str]] = {}

    for line_idx, (s1, matched_str) in enumerate(matching_rows[1:], 2):
        if not s1:
            issues.append(f"matching_results.tsv: line {line_idx} has empty source1_entity_id")
            continue
        if s1 in seen_s1_match:
            issues.append(f"matching_results.tsv: duplicate row for entity {s1} at line {line_idx}")
        seen_s1_match.add(s1)

        if s1 not in valid_s1_ids:
            issues.append(f"matching_results.tsv: entity {s1} is not a valid test Source 1 entity")

        target_list = [t.strip() for t in matched_str.split(",") if t.strip()]
        if len(target_list) != len(set(target_list)):
            issues.append(f"matching_results.tsv: line {line_idx} ({s1}) has duplicate target IDs: {matched_str}")

        for t in target_list:
            if t.startswith("S1-"):
                issues.append(f"matching_results.tsv: line {line_idx} ({s1}) contains illegal self-match to Source 1: {t}")
            elif t not in valid_target_ids:
                issues.append(f"matching_results.tsv: line {line_idx} ({s1}) references non-existent test ID: {t}")

        match_map[s1] = set(target_list)

    # Inspect candidate rows
    seen_s1_cand = set()
    cand_map: Dict[str, Set[str]] = {}

    for line_idx, (s1, cands_str) in enumerate(candidate_rows[1:], 2):
        if not s1:
            issues.append(f"candidate_pairs.tsv: line {line_idx} has empty source1_entity_id")
            continue
        if s1 in seen_s1_cand:
            issues.append(f"candidate_pairs.tsv: duplicate row for entity {s1} at line {line_idx}")
        seen_s1_cand.add(s1)

        if s1 not in valid_s1_ids:
            issues.append(f"candidate_pairs.tsv: entity {s1} is not a valid test Source 1 entity")

        cand_list = [c.strip() for c in cands_str.split(",") if c.strip()]
        if len(cand_list) != len(set(cand_list)):
            issues.append(f"candidate_pairs.tsv: line {line_idx} ({s1}) has duplicate candidate IDs: {cands_str}")

        for c in cand_list:
            if c.startswith("S1-"):
                issues.append(f"candidate_pairs.tsv: line {line_idx} ({s1}) contains illegal self-match to Source 1: {c}")
            elif c not in valid_target_ids:
                issues.append(f"candidate_pairs.tsv: line {line_idx} ({s1}) references non-existent test ID: {c}")

        cand_map[s1] = set(cand_list)

    # Check Completeness: Every test S1 entity must be present
    missing_in_match = valid_s1_ids - seen_s1_match
    if missing_in_match:
        issues.append(f"matching_results.tsv is missing {len(missing_in_match)} test S1 entities (e.g. {list(missing_in_match)[:3]})")

    missing_in_cand = valid_s1_ids - seen_s1_cand
    if missing_in_cand:
        issues.append(f"candidate_pairs.tsv is missing {len(missing_in_cand)} test S1 entities (e.g. {list(missing_in_cand)[:3]})")

    # Check Subset Property: Every matched ID must appear in candidate_pairs.tsv
    for s1, matched_set in match_map.items():
        cands_set = cand_map.get(s1, set())
        not_in_cands = matched_set - cands_set
        if not_in_cands:
            issues.append(f"Pipeline bug: {s1} has matches {not_in_cands} that are not in candidate_pairs.tsv")

    if issues:
        print(f"\nVALIDATION FAILED ({len(issues)} issues found):")
        for i, issue in enumerate(issues[:20], 1):
            print(f"  [{i}] {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues.")
        return False

    print("\nPASS: Both submission files satisfy all challenge constraints and format rules.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission files")
    parser.add_argument("--matching", type=str, default="output/matching_results.tsv")
    parser.add_argument("--candidate", type=str, default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    args = parser.parse_args()

    passed = validate(Path(args.matching), Path(args.candidate), Path(args.test_dir))
    sys.exit(0 if passed else 1)
