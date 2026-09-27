"""
Unit tests for strict TSV I/O utilities.
"""

import tempfile
from pathlib import Path
from business_entity_resolution.src.io_utils import read_tsv, parse_ground_truth, write_submission_tsv


def test_read_tsv_preserves_na_strings():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.tsv"
        content = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-01\tNA\t123 Main St\tUS\nS1-02\t\tNULL\tIN\n"
        test_file.write_text(content, encoding="utf-8")

        df = read_tsv(test_file)
        assert df.loc[0, "business_name"] == "NA"
        assert df.loc[1, "business_name"] == ""
        assert df.loc[1, "business_address"] == "NULL"


def test_parse_ground_truth():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_file = Path(tmpdir) / "train_ground_truth.tsv"
        content = "source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00047,S3-00812\nS1-00002\t\n"
        gt_file.write_text(content, encoding="utf-8")

        parsed = parse_ground_truth(gt_file)
        assert parsed["S1-00001"] == ["S2-00047", "S3-00812"]
        assert parsed["S1-00002"] == []


def test_write_submission_tsv():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "matching_results.tsv"
        preds = {
            "S1-00001": ["S2-00047", "S3-00812"],
            "S1-00002": [],
            "S1-00003": ["S3-00004"]
        }
        write_submission_tsv(out_file, preds, is_candidate_file=False)

        raw = out_file.read_text(encoding="utf-8")
        lines = [line.strip("\r\n") for line in raw.splitlines()]
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-00001\tS2-00047,S3-00812"
        assert lines[2] == "S1-00002\t"
        assert lines[3] == "S1-00003\tS3-00004"
