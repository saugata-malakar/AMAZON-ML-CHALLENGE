"""
Unit tests for submission validator script.
"""

import tempfile
from pathlib import Path
from business_entity_resolution.utils.validate_submission import validate


def test_validator_passes_on_valid_submission():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        test_dir = root / "dataset" / "test"
        output_dir = root / "output"
        test_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        # Create dummy test files
        (test_dir / "test_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-01\tA\tB\tUS\nS1-02\tC\tD\tFR\n", encoding="utf-8")
        (test_dir / "test_source2.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-01\tA\tB\tUS\n", encoding="utf-8")
        (test_dir / "test_source3.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-01\tC\tD\tFR\n", encoding="utf-8")

        # Valid outputs: S1-01 -> S2-01, S1-02 is singleton
        matching_file = output_dir / "matching_results.tsv"
        candidate_file = output_dir / "candidate_pairs.tsv"

        matching_file.write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-01\nS1-02\t\n", encoding="utf-8")
        candidate_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\tS2-01,S3-01\nS1-02\tS3-01\n", encoding="utf-8")

        assert validate(matching_file, candidate_file, test_dir) is True


def test_validator_fails_when_matched_not_in_candidates():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        test_dir = root / "dataset" / "test"
        output_dir = root / "output"
        test_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        (test_dir / "test_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-01\tA\tB\tUS\n", encoding="utf-8")
        (test_dir / "test_source2.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-01\tA\tB\tUS\n", encoding="utf-8")
        (test_dir / "test_source3.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")

        matching_file = output_dir / "matching_results.tsv"
        candidate_file = output_dir / "candidate_pairs.tsv"

        # Matching has S2-01, but candidates does NOT have S2-01 -> pipeline bug
        matching_file.write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-01\n", encoding="utf-8")
        candidate_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\t\n", encoding="utf-8")

        assert validate(matching_file, candidate_file, test_dir) is False
