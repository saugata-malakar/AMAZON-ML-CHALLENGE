import time, os, sys, re, gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci', 'france', 'de', 'la',
    'le', 'du', 'des', 'les', 'en', 'rue', 'bd', 'boulevard', 'av', 'impasse'
}

def clean_toks(name, addr):
    text = (name + " " + addr).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return set(w for w in text.split() if len(w) >= 3 and w not in STOPWORDS)

def get_words(text):
    return set(re.findall(r'\w+', text.lower()))

def get_3grams(text):
    t = text.lower()
    return set(t[i:i+3] for i in range(len(t)-2))

def fast_jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def extract_features(s1_n, s1_a, t_n, t_a, shared_tok_count):
    n_tok1 = get_words(s1_n)
    n_tok2 = get_words(t_n)
    n_jacc = fast_jaccard(n_tok1, n_tok2)
    
    n_3g1 = get_3grams(s1_n)
    n_3g2 = get_3grams(t_n)
    n_3g_jacc = fast_jaccard(n_3g1, n_3g2)
    
    a_tok1 = get_words(s1_a)
    a_tok2 = get_words(t_a)
    a_jacc = fast_jaccard(a_tok1, a_tok2)
    
    num1 = set(re.findall(r'\d+', s1_a))
    num2 = set(re.findall(r'\d+', t_a))
    num_jacc = fast_jaccard(num1, num2)
    
    len1 = len(s1_n)
    len2 = len(t_n)
    len_ratio = (min(len1, len2) + 1) / (max(len1, len2) + 1)
    
    # Combined text jaccard
    c_jacc = fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2)
    
    return [n_jacc, n_3g_jacc, a_jacc, num_jacc, len_ratio, shared_tok_count, c_jacc]

def macro_f05(gt_dict, pred_dict):
    scores = []
    for s1_id, true_targets in gt_dict.items():
        preds = set(pred_dict.get(s1_id, []))
        if not true_targets:
            # Singleton
            scores.append(1.0 if not preds else 0.0)
        else:
            if not preds:
                scores.append(0.0)
            else:
                tp = len(preds & true_targets)
                precision = tp / len(preds)
                recall = tp / len(true_targets)
                denom = 0.25 * precision + recall
                f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
                scores.append(f05)
    return np.mean(scores) if scores else 0.0

print("Testing training and threshold optimization on train set...")
base = r'c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train'

# Load GT for 30,000 entities
print("Reading ground truth...")
gt_map = {}
with open(f'{base}/train_ground_truth.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_id = p[0]
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
        gt_map[s1_id] = set(m)
        if len(gt_map) >= 30000:
            break

s1_eval_ids = set(gt_map.keys())

# Load S1 entities
s1_records = {}
with open(f'{base}/train_source1.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_eval_ids:
            s1_records[p[0]] = (p[1], p[2], p[3])
            if len(s1_records) == len(s1_eval_ids):
                break

print(f"Loaded {len(s1_records)} S1 entities")

# Split into 20k train S1, 10k val S1
s1_list = list(s1_records.keys())
train_s1_ids = set(s1_list[:20000])
val_s1_ids = set(s1_list[20000:])

# Collect needed target IDs for evaluation
all_needed_targets = set()
for sid in s1_eval_ids:
    all_needed_targets |= gt_map[sid]

print(f"Loading targets from S2 and S3...")
# Let's load 1,000,000 targets from S2 and S3 + all true targets
tgt_records = {}
for sf in ['train_source2.tsv', 'train_source3.tsv']:
    with open(f'{base}/{sf}', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip('\n').split('\t')
            if p[0] in all_needed_targets or i < 400000:
                tgt_records[p[0]] = (p[1], p[2], p[3])

print(f"Total targets loaded: {len(tgt_records):,}")

# Build distinctive token index on targets
print("Building index on targets...")
t0 = time.time()
inv_index = defaultdict(list)
tgt_id_list = list(tgt_records.keys())
for i, tid in enumerate(tgt_id_list):
    n, a, c = tgt_records[tid]
    for tok in clean_toks(n, a):
        inv_index[tok].append(i)

# Filter high freq tokens (> 300)
inv_index = {k: v for k, v in inv_index.items() if len(v) <= 300}
print(f"Index built in {time.time()-t0:.2f}s with {len(inv_index)} tokens")

# Query train S1 and build training features
print("Generating training candidate pairs...")
X_train = []
y_train = []

for sid in train_s1_ids:
    s1_n, s1_a, s1_c = s1_records[sid]
    toks = clean_toks(s1_n, s1_a)
    counts = Counter()
    for tok in toks:
        if tok in inv_index:
            for idx in inv_index[tok]:
                counts[idx] += 1
    
    top = counts.most_common(20)
    true_tgt = gt_map[sid]
    
    for idx, shared_c in top:
        tid = tgt_id_list[idx]
        t_n, t_a, t_c = tgt_records[tid]
        if s1_c != t_c: continue
        feat = extract_features(s1_n, s1_a, t_n, t_a, shared_c)
        X_train.append(feat)
        y_train.append(1 if tid in true_tgt else 0)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"Training pairs: {len(X_train)} (Positive: {sum(y_train)}, Negative: {len(y_train)-sum(y_train)})")

clf = HistGradientBoostingClassifier(random_state=42, max_iter=200, min_samples_leaf=30)
clf.fit(X_train, y_train)
print("Classifier trained successfully!")

# Validate on 10,000 val S1 entities
print("\nEvaluating on 10,000 validation entities...")
val_candidates = {}
val_features = []
val_pairs_meta = [] # (s1_id, target_id)

for sid in val_s1_ids:
    s1_n, s1_a, s1_c = s1_records[sid]
    toks = clean_toks(s1_n, s1_a)
    counts = Counter()
    for tok in toks:
        if tok in inv_index:
            for idx in inv_index[tok]:
                counts[idx] += 1
    top = counts.most_common(20)
    
    val_candidates[sid] = []
    for idx, shared_c in top:
        tid = tgt_id_list[idx]
        t_n, t_a, t_c = tgt_records[tid]
        if s1_c != t_c: continue
        val_candidates[sid].append(tid)
        val_features.append(extract_features(s1_n, s1_a, t_n, t_a, shared_c))
        val_pairs_meta.append((sid, tid))

val_features = np.array(val_features)
probs = clf.predict_proba(val_features)[:, 1]

# Test different thresholds
gt_val = {sid: gt_map[sid] for sid in val_s1_ids}

print("\nThreshold Search for Macro F0.5:")
best_t = 0.5
best_score = 0.0

for t in np.arange(0.50, 0.95, 0.05):
    # Greedy 1-to-1 assignment
    scored_pairs = []
    for (sid, tid), p in zip(val_pairs_meta, probs):
        if p >= t:
            scored_pairs.append((p, sid, tid))
    
    scored_pairs.sort(reverse=True)
    assigned_tgt = set()
    preds = defaultdict(list)
    for p, sid, tid in scored_pairs:
        if tid not in assigned_tgt:
            preds[sid].append(tid)
            assigned_tgt.add(tid)
    
    score = macro_f05(gt_val, preds)
    print(f"  Threshold {t:.2f} -> Macro F0.5 = {score:.4f} (Matched {len(preds)}/{len(val_s1_ids)} S1)")
    if score > best_score:
        best_score = score
        best_t = t

print(f"\nBEST THRESHOLD: {best_t:.2f} with Macro F0.5 = {best_score:.4f}")
