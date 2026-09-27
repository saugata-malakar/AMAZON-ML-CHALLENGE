# Amazon ML Challenge - Business Entity Resolution Pipeline

This repository contains an end-to-end, reproducible Machine Learning pipeline for multi-source Business Entity Resolution (ER), designed and calibrated specifically for the competition's macro-averaged $F_{0.5}$ metric.

---

## 1. Pipeline Architecture

1. **Text & Address Normalization (`src/normalize/`)**:
   - Unicode NFKD decomposition and accent stripping (crucial for French addresses in test data).
   - DBA / trade name splitting (e.g., `dba`, `t/a`, `formerly`).
   - Country-aware legal suffix extraction (`pvt ltd`, `inc`, `corp`, `sarl`, `llc`) into distinct core names.
   - Address abbreviation expansions (US, India, France, and Global fallback).
   - Regex-based postal code / PIN extraction (US 5-digit, India 6-digit, France 5-digit).
   - Open-set country normalization (never filters unseen countries like France).

2. **Blocking / Candidate Generation (`src/blocking/`)**:
   - Country-partitioned sparse TF-IDF character n-grams (3-4) and word n-grams.
   - Multi-channel candidate retrieval with $\ge 98\%$ recall ceiling.
   - Directly outputs `output/candidate_pairs.tsv`.

3. **Pairwise Feature Engineering (`src/features/`)**:
   - High-dimensional similarity signals: Jaro-Winkler, Levenshtein ratio, token-sort ratio, token-set ratio, character 3-gram Jaccard, acronym matching, legal suffix matching, Soundex phonetic matching.
   - Address similarities: Postal/PIN code matching (exact, prefix, missing), building/house number intersection, landmark phrase overlap.
   - Relational features: Candidate rank, target source indicator (`is_s2`, `is_s3`), country match.

4. **Pairwise Scoring & Macro $F_{0.5}$ Calibration (`src/models/`, `src/decision/`)**:
   - Gradient Boosted Decision Tree (GBDT) classifier (`HistGradientBoostingClassifier`, scikit-learn, BSD-3-Clause license, $< 8\text{B}$ parameters, zero pretrained weights).
   - Grid-search threshold optimization targeting the competition's macro-averaged $F_{0.5}$ metric on held-out validation.
   - Strict deterministic tie-breaking on compound keys `(-probability, s1_id, target_id)` ensuring exact reproducibility across runs independent of `PYTHONHASHSEED`.
   - Greedy unique target assignment enforcing 1-to-1 matching constraint (each S2/S3 record is assigned to at most one S1 entity).
   - Directly outputs `output/matching_results.tsv`.

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── README.md
├── requirements.txt
├── configs/
│   └── abbreviations/
│       ├── global.yaml
│       ├── us.yaml
│       ├── in.yaml
│       └── fr.yaml
├── src/
│   ├── io_utils.py               # Strict TSV reading/writing
│   ├── metric.py                 # Macro F0.5 per challenge specification
│   ├── eda.py                    # Comprehensive dataset diagnostics
│   ├── normalize/                # Unicode, name, address, country, phonetic
│   ├── blocking/                 # Sparse TF-IDF candidate retrieval
│   ├── features/                 # Pairwise name and address feature engineering
│   ├── models/                   # GBDT pairwise matching model
│   ├── decision/                 # Unique assignment and F0.5 threshold search
│   └── pipeline/
│       ├── train_and_validate.py # End-to-end training and CV validation
│       └── predict.py            # End-to-end test inference and TSV output
├── tests/                        # 19 passing unit tests
└── utils/
    └── validate_submission.py    # Organizer validation compliance check
```

---

## 3. How to Run End-to-End

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```bash
python -m pytest tests -v
```

### Step 3: Run EDA on Training Data
```bash
python -m business_entity_resolution.src.eda --data-dir dataset/train
```

### Step 4: Train Model & Optimize Threshold
```bash
python -m business_entity_resolution.src.pipeline.train_and_validate \
    --data-dir dataset/train \
    --artifact-dir code/business_entity_resolution/artifacts
```

### Step 5: Run Inference on Test Set
```bash
python -m business_entity_resolution.src.pipeline.predict \
    --test-dir dataset/test \
    --artifact-dir code/business_entity_resolution/artifacts \
    --output-dir output
```

### Step 6: Validate Submission Files
```bash
python code/business_entity_resolution/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
