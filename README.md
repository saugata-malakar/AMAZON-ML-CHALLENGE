# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the complete, production-grade, reproducible Machine Learning pipeline for multi-source Business Entity Resolution (ER) at extreme scale (~22.4M records across noisy business databases), calibrated for the competition's macro-averaged $F_{0.5}$ metric.

---

## 🌟 Key Highlights

- **Pure Classical ML & Zero Pretrained Models:** Built from scratch using `HistGradientBoostingClassifier` (BSD-3-Clause) and C-accelerated token/n-gram similarity features.
- **Extreme Scale & Throughput:** Evaluated across **10,320,219 target records** and **1,732,544 test entities** in under 55 minutes (<1 GB RAM streaming).
- **Portable JSON Model (Zero-Retraining):** The complete trained tree ensemble (split thresholds, feature indices, leaf values) is exported into standalone JSON files (`model_config.json`, `model_parameters.json`) with an exact pure Python/NumPy inference engine (`predict_from_json.py`).
- **Country-Agnostic & Unseen Country Generalization:** Evaluated via bidirectional Leave-One-Country-Out (LOCO) cross-validation and French linguistic stress testing (88.89% true recall).
- **Official Validator Compliance:** Passes all official format and integrity rules (`PASS — no blocking issues found. Safe to submit.`).

---

## 📁 Repository Structure

```text
.
├── code/
│   └── business_entity_resolution/
│       ├── artifacts/                # Exported JSON models (config, parameters, trees)
│       │   ├── model_config.json     # Hyperparameters, threshold tau, feature definitions
│       │   └── model_parameters.json # Complete tree weights and split nodes (zero retraining)
│       ├── configs/                  # Hand-authored abbreviation dictionaries (fr, in, us, global)
│       ├── src/                      # Production pipeline source modules
│       │   ├── normalize/            # Unicode, diacritics, legal suffixes, addresses
│       │   ├── blocking/             # Inverted index candidate generation with IDF
│       │   ├── features/             # Vectorized 14-dimensional feature extraction
│       │   ├── models/               # GBDT pairwise matching models
│       │   ├── decision/             # Greedy 1-to-1 priority locking & thresholding
│       │   ├── predict_from_json.py  # Zero-retraining portable inference engine
│       │   └── run_full_production.py# End-to-end full-scale production script
│       ├── tests/                    # Passing pytest suite
│       ├── utils/                    # Organizer submission validator
│       ├── README.md                 # Detailed code README
│       └── requirements.txt          # Minimal Python dependencies
│
├── Documentation_template.md         # Comprehensive methodology and audit report
├── run_full_production.py            # Canonical full-scale production runner
└── export_model_to_json.py           # Model parameter exporter and numerical verifier
```

---

## 🚀 Portable JSON Inference (Zero Retraining)

You can run forward inference on any machine without installing heavy ML frameworks or retraining:

```python
from code.business_entity_resolution.src.predict_from_json import (
    load_json_model,
    PortableGBDTPredictor,
)

# 1. Load model config and parameters directly from JSON
config, params = load_json_model("code/business_entity_resolution/artifacts")
predictor = PortableGBDTPredictor(config, params)

# 2. Score a 14-dimensional feature vector
prob = predictor.predict_proba_single(feature_vector)
is_match = prob >= predictor.optimal_tau
```

Numerical parity between scikit-learn and the pure JSON predictor is verified at machine precision ($\Delta < 2.22 \times 10^{-16}$).

---

## 🛠️ How to Run Full Pipeline

```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run unit tests
python -m pytest code/business_entity_resolution/tests -v

# 3. Run production pipeline
python code/business_entity_resolution/src/run_full_production.py

# 4. Validate output files
python code/business_entity_resolution/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 📜 License
Distributed under the BSD-3-Clause License. See code headers for details.
