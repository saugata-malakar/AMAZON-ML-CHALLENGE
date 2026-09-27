#!/usr/bin/env python3
"""
Test validation score of the 14 pure-C-backed features on 50k train / 20k val
"""
import time, os, sys, re, gc, math
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")

TOP_K = 15
TRAIN_SAMPLE = 50_000
VAL_SAMPLE   = 20_000

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci', 'france', 'de', 'la',
    'le', 'du', 'des', 'les', 'en', 'rue', 'bd', 'boulevard', 'av', 'impasse',
    'new', 'old', 'east', 'west', 'north', 'south', 'main', 'cross'
}

def clean_toks(text):
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in clean.split() if len(w) >= 3 and w not in STOPWORDS]

def clean_toks_set(text):
    return set(clean_toks(text))

def get_words(text):
    return set(re.findall(r'\w+', text.lower()))

def get_2grams(text):
    t = text.lower()
    if len(t) < 2: return set()
    return set(t[i:i+2] for i in range(len(t)-1))

def get_3grams(text):
    t = text.lower()
    if len(t) < 3: return set()
    return set(t[i:i+3] for i in range(len(t)-2))

def fast_jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def extract_features(s1_n, s1_a, t_n, t_a, idf_score):
    s1_nc = re.sub(r'[^\w\s]', ' ', s1_n.lower()).strip()
    t_nc  = re.sub(r'[^\w\s]', ' ', t_n.lower()).strip()
    s1_ac = re.sub(r'[^\w\s]', ' ', s1_a.lower()).strip()
    t_ac  = re.sub(r'[^\w\s]', ' ', t_a.lower()).strip()

    n_tok1, n_tok2 = get_words(s1_n), get_words(t_n)
    n_2g1,  n_2g2  = get_2grams(s1_n), get_2grams(t_n)
    n_3g1,  n_3g2  = get_3grams(s1_n), get_3grams(t_n)
    a_tok1, a_tok2 = get_words(s1_a), get_words(t_a)
    a_2g1,  a_2g2  = get_2grams(s1_a), get_2grams(t_a)
    a_3g1,  a_3g2  = get_3grams(s1_a), get_3grams(t_a)

    num1 = set(re.findall(r'\d+', s1_a))
    num2 = set(re.findall(r'\d+', t_a))

    min_len = min(len(s1_nc), len(t_nc))
    pfx_len = 0
    if min_len > 0:
        for i in range(min_len):
            if s1_nc[i] == t_nc[i]: pfx_len += 1
            else: break
        pfx_ratio = pfx_len / min_len
    else:
        pfx_ratio = 0.0

    return [
        fast_jaccard(n_tok1, n_tok2),
        fast_jaccard(n_2g1, n_2g2),
        fast_jaccard(n_3g1, n_3g2),
        pfx_ratio,
        (min(len(s1_nc), len(t_nc)) + 1) / (max(len(s1_nc), len(t_nc)) + 1),
        len(n_tok1 & n_tok2) / (max(len(n_tok1), 1)),

        fast_jaccard(a_tok1, a_tok2),
        fast_jaccard(a_2g1, a_2g2),
        fast_jaccard(a_3g1, a_3g2),
        fast_jaccard(num1, num2),
        1.0 if (not s1_ac or not t_ac) else 0.0,

        fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2),
        fast_jaccard(n_3g1 | a_3g1, n_3g2 | a_3g2),
        idf_score,
    ]

def macro_f05(gt_dict, pred_dict):
    scores = []
    for s1_id, true_targets in gt_dict.items():
        preds = set(pred_dict.get(s1_id, []))
        if not true_targets:
            scores.append(1.0 if not preds else 0.0)
        else:
            if not preds:
                scores.append(0.0)
            else:
                tp = len(preds & true_targets)
                p = tp / len(preds)
                r = tp / len(true_targets)
                d = 0.25 * p + r
                scores.append((1.25 * p * r) / d if d > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

print("Loading GT...")
gt_map = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
        gt_map[p[0]] = set(m)
        if len(gt_map) >= TRAIN_SAMPLE + VAL_SAMPLE:
            break

s1_recs = {}
s1_needed = set(gt_map.keys())
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed:
            s1_recs[p[0]] = (p[1], p[2], p[3])
            if len(s1_recs) == len(s1_needed): break

s1_list = list(s1_recs.keys())
train_ids = set(s1_list[:TRAIN_SAMPLE])
val_ids = set(s1_list[TRAIN_SAMPLE:])

all_needed = set()
for sid in s1_needed:
    all_needed |= gt_map[sid]

print("Loading Targets...")
tgt_recs = {}
for sf in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, sf), encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip('\n').split('\t')
            if p[0] in all_needed or i < 400000:
                tgt_recs[p[0]] = (p[1], p[2], p[3])

tgt_id_list = list(tgt_recs.keys())
tgt_names = [tgt_recs[t][0] for t in tgt_id_list]
tgt_addrs = [tgt_recs[t][1] for t in tgt_id_list]
tgt_ctry  = {tgt_id_list[i]: tgt_recs[tgt_id_list[i]][2] for i in range(len(tgt_id_list))}
N = len(tgt_id_list)

print(f"Building fast inverted index on {N:,} targets...")
t0 = time.time()
inv_index = defaultdict(list)
for i in range(N):
    toks = clean_toks_set(tgt_names[i] + " " + tgt_addrs[i])
    for tok in toks:
        if len(inv_index[tok]) < 500:
            inv_index[tok].append(i)

# IDF weights
idf = {tok: math.log(N / (len(postings) + 1)) for tok, postings in inv_index.items()}
print(f"Index built in {time.time()-t0:.1f}s: {len(inv_index):,} tokens")

def query_blocker(name, addr):
    toks = clean_toks_set(name + " " + addr)
    scores = Counter()
    for tok in toks:
        if tok in inv_index:
            w = idf[tok]
            for idx in inv_index[tok]:
                scores[idx] += w
    return scores.most_common(TOP_K)

print("Generating training candidate pairs...")
t0 = time.time()
X_train, y_train = [], []
for sid in train_ids:
    s1_n, s1_a, s1_c = s1_recs[sid]
    true_tgt = gt_map[sid]
    for idx, idf_w in query_blocker(s1_n, s1_a):
        tid = tgt_id_list[idx]
        if tgt_ctry.get(tid) != s1_c: continue
        feat = extract_features(s1_n, s1_a, tgt_names[idx], tgt_addrs[idx], idf_w)
        X_train.append(feat)
        y_train.append(1 if tid in true_tgt else 0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"Training pairs: {len(X_train):,} in {time.time()-t0:.1f}s (Pos: {y_train.sum():,})")

print("Training HistGradientBoostingClassifier...")
clf = HistGradientBoostingClassifier(
    max_iter=300, max_depth=7, learning_rate=0.08,
    min_samples_leaf=30, l2_regularization=0.5, random_state=42
)
clf.fit(X_train, y_train)

print("Evaluating on Validation set...")
t0 = time.time()
val_feats, val_meta = [], []
for sid in val_ids:
    s1_n, s1_a, s1_c = s1_recs[sid]
    for idx, idf_w in query_blocker(s1_n, s1_a):
        tid = tgt_id_list[idx]
        if tgt_ctry.get(tid) != s1_c: continue
        feat = extract_features(s1_n, s1_a, tgt_names[idx], tgt_addrs[idx], idf_w)
        val_feats.append(feat)
        val_meta.append((sid, tid))

val_feats = np.array(val_feats, dtype=np.float32)
probs = clf.predict_proba(val_feats)[:, 1]
print(f"Validation pairs: {len(val_feats):,} scored in {time.time()-t0:.1f}s")

gt_val = {sid: gt_map[sid] for sid in val_ids}
best_t, best_f05 = 0.5, 0.0
for t in np.arange(0.50, 0.90, 0.02):
    scored = [(p, sid, tid) for (sid, tid), p in zip(val_meta, probs) if p >= t]
    scored.sort(reverse=True)
    assigned = set()
    preds = defaultdict(list)
    for p, sid, tid in scored:
        if tid not in assigned:
            preds[sid].append(tid)
            assigned.add(tid)
    f05 = macro_f05(gt_val, preds)
    if f05 > best_f05:
        best_f05 = f05
        best_t = float(t)
    print(f"  Threshold {t:.2f} -> Macro F0.5 = {f05:.4f}")

print(f"\n>>> BEST THRESHOLD: {best_t:.2f} with Macro F0.5 = {best_f05:.4f} <<<")
