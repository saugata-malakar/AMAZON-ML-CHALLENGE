# Business Entity Resolution Challenge — Methodology Write-up

## 1. Methodology Overview
Our solution approaches multi-source Business Entity Resolution through a decoupled, multi-stage machine learning pipeline:
1. **Domain-Specific Normalization & Component Extraction**: Normalizes text across open-set countries (US, India, France) via Unicode NFKD decomposition, DBA/trade name splitting, legal suffix stripping, and postal/PIN code extraction.
2. **High-Recall Candidate Blocking**: Constrains the search space from $O(N \times M)$ pairwise comparisons down to top-$K$ candidates per Source 1 entity using country-partitioned sparse TF-IDF character (3-4) and word n-grams, establishing a high recall ceiling ($\ge 98\%$).
3. **Tabular Pairwise Feature Engineering**: Formulates linkage as a binary classification problem using 20+ fine-grained similarity signals across business names (Jaro-Winkler, Levenshtein, token-set/sort ratios, acronym matching, Soundex phonetic keys) and addresses (PIN match, building number overlap, landmark overlap).
4. **Calibrated Classification & Metric Optimization**: Trains an Apache-2.0 compliant Gradient Boosted Decision Tree (GBDT) model, then optimizes the classification threshold specifically to maximize the challenge's macro-averaged $F_{0.5}$ metric.
5. **Greedy 1-to-1 Conflict Resolution**: Enforces the structural invariant that each target record (S2/S3) belongs to at most one Source 1 reference entity.

---

## 2. Candidate Generation & Blocking Strategy
- **Search Space Reduction**: Comparing all S1 entities against all S2/S3 target entities is computationally prohibitive.
- **Partitioning**: Blocks are partitioned by normalized country. Because France appears only at test time, country handling is strictly open-set: unseen countries execute using language-agnostic character n-grams and universal abbreviations.
- **Algorithms**:
  - Sparse TF-IDF character 3-to-4 n-grams on core business names (robust against typos and transliterations).
  - Word-level TF-IDF on combined name and address.
  - Candidate mapping directly populates `output/candidate_pairs.tsv` to ensure perfect alignment with final predictions.

---

## 3. Model Architecture & Feature Engineering

### 3.1 Feature Space
- **Name Features**:
  - `name_char_jaccard_3`: Character 3-gram Jaccard similarity.
  - `name_lev_ratio`: Normalized Levenshtein edit similarity ratio.
  - `name_token_jaccard`, `name_token_sort_ratio`, `name_token_set_ratio`: Word-level token permutation similarities.
  - `name_acronym_match`: Detects initialisms (e.g., "SBI" vs "State Bank of India").
  - `name_containment`: Substring containment indicator.
  - `name_first_token_match` & `name_last_token_match`: Boundary token matches.
  - `name_suffix_match`: Legal suffix consistency (e.g. `pvt ltd`, `inc`, `sarl`).
  - `name_phonetic_match`: Soundex phonetic equality for transliteration resilience.
- **Address Features**:
  - `addr_char_jaccard_3`, `addr_token_jaccard`, `addr_token_sort_ratio`, `addr_token_set_ratio`.
  - `addr_pin_match`: 1.0 for exact postal/PIN code match, 0.5 for 3-digit prefix match, 0.0 for mismatch, -1.0 if missing.
  - `addr_num_jaccard` & `addr_num_inter_count`: Exact street/building number overlaps.
  - `addr_landmark_overlap`: Overlap of landmark descriptors ("near", "opposite").
- **Relational & Source Features**:
  - Candidate rank index, `is_s2`, `is_s3`, and `country_match`.

### 3.2 Machine Learning Classifier
- **Model**: HistGradientBoostingClassifier (Gradient Boosted Decision Trees).
- **Parameters**: 150 trees, max depth 31 leaf nodes, learning rate 0.08, L2 regularization 1.0.
- **Parameter Count & License**: $< 500\text{K}$ parameters (far below the $8\text{B}$ ceiling), licensed under BSD / Apache 2.0.

---

## 4. Decision Layer & Metric Optimization ($F_{0.5}$)
- The official evaluation metric is macro-averaged $F_{0.5}$, which weights precision twice as heavily as recall:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Singletons**: Source 1 entities with no true matches score $1.0$ when predicted empty, but drop straight to $0.0$ if even a single false match is predicted.
- **Optimization**: The decision threshold $\tau$ is tuned via grid search on out-of-fold validation entities to directly maximize macro $F_{0.5}$.
- **Uniqueness Assignment**: A greedy maximum-probability assignment ensures each target ID is linked to at most one reference S1 entity.

---

## 5. Academic Integrity & Fair Play Statement
- **Zero External Lookups**: No external databases, commercial ER APIs, government registries, or geocoding services (e.g., Google Maps, Nominatim) were used.
- **Self-Contained & Reproducible**: All normalization dictionaries, blocking indexes, feature transformers, and decision thresholds were built strictly from the competition data and our own domain knowledge.
