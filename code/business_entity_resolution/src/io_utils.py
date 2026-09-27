"""
I/O utilities for Business Entity Resolution Challenge.

Per challenge specification:
- All source, ground truth, and submission files are TAB-SEPARATED (.tsv).
- We read with dtype=str, keep_default_na=False to ensure business names like 'NA'
  or 'NULL' and missing values are safely loaded as strings without silent corruption.
"""

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def read_tsv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a TSV file with strict string dtypes and no default NaN casting.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )
    return df


def parse_ground_truth(df_or_path: Union[pd.DataFrame, str, Path]) -> Dict[str, List[str]]:
    """
    Parses train_ground_truth.tsv into a dictionary:
    {
        'S1-00001': ['S2-00047', 'S3-00812'],
        'S1-00002': []  # singleton
    }
    """
    if isinstance(df_or_path, (str, Path)):
        df = read_tsv(df_or_path)
    else:
        df = df_or_path

    if "source1_entity_id" not in df.columns or "matched_entity_ids" not in df.columns:
        raise ValueError(f"Ground truth dataframe missing required columns: {GROUND_TRUTH_COLUMNS}")

    truth_dict: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matched_str = row["matched_entity_ids"].strip()
        if matched_str:
            truth_dict[s1_id] = [m.strip() for m in matched_str.split(",") if m.strip()]
        else:
            truth_dict[s1_id] = []

    return truth_dict


def write_submission_tsv(
    output_path: Union[str, Path],
    id_mapping: Dict[str, List[str]],
    is_candidate_file: bool = False
) -> None:
    """
    Writes predictions or candidates to TSV according to challenge format:
    - No quoting
    - Tab-separated
    - S1 entities ordered or provided
    - Comma-separated target IDs (empty string for singletons)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    col1 = "source1_entity_id"
    col2 = "candidate_entity_ids" if is_candidate_file else "matched_entity_ids"

    records = []
    for s1_id, target_ids in id_mapping.items():
        # Remove any empty strings and deduplicate preserving order
        deduped = []
        seen = set()
        for tid in target_ids:
            clean_tid = tid.strip()
            if clean_tid and clean_tid not in seen:
                seen.add(clean_tid)
                deduped.append(clean_tid)
        records.append({
            col1: s1_id.strip(),
            col2: ",".join(deduped)
        })

    df_out = pd.DataFrame(records, columns=[col1, col2])
    df_out.to_csv(
        output_path,
        sep="\t",
        index=False,
        encoding="utf-8"
    )
