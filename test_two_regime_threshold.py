#!/usr/bin/env python3
"""
Test Idea #1: Two-Regime Thresholding.
Trains v5 on 20k entities, evaluates on 10k val entities.
Sweeps tau1 (1st match threshold) and tau2 (2nd+ match threshold).
"""
import sys, os, re, math, time
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')
BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train"

STOPWORDS = {
    'inc','corp','corporation','llc','ltd','limited','pvt','private','co','company',
    'and','the','of','in','at','road','rd','street','st','avenue','ave','lane','ln',
    'drive','dr','nagar','colony','floor','near','opp','opposite','block','sector',
    'phase','house','plot','door','no','null','sarl','sas','sci','france','de','la',
    'le','du','des','les','en','rue','bd','boulevard','av','impasse','new','old',
    'east','west','north','south','main','cross'
}

def clean_toks(n, a):
    text = (n + " " + a).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) >= 3 and w not in STOPWORDS]

def get_words(text): return set(re.findall(r'\w+', text.lower()))
def get_2grams(text):
    t = text.lower()
    return set(t[i:i+2] for i in range(len(t)-1)) if len(t) >= 2 else set()
def get_3grams(text):
    t = text.lower()
    return set(t[i:i+3] for i in range(len(t)-2)) if len(t) >= 3 else set()
def fast_jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2) if (s1 and s2) else 0.0

def precompute_s1(n, a):
    nc = re.sub(r'[^\w\s]', ' ', n.lower()).strip()
    ac = re.sub(r'[^\w\s]', ' ', a.lower()).strip()
    return (get_words(n), get_2grams(n), get_3grams(n),
            get_words(a), get_2grams(a), get_3grams(a),
            set(re.findall(r'\d+', a)), nc, ac)

def extract_features(s1_pre, t_n, t_a, idf_score):
    (n_tok1, n_2g1, n_3g1, a_tok1, a_2g1, a_3g1, num1, s1_nc, s1_ac) = s1_pre
    t_nc = re.sub(r'[^\w\s]', ' ', t_n.lower()).strip()
    t_ac = re.sub(r'[^\w\s]', ' ', t_a.lower()).strip()
    n_tok2 = get_words(t_n); n_2g2 = get_2grams(t_n); n_3g2 = get_3grams(t_n)
    a_tok2 = get_words(t_a); a_2g2 = get_2grams(t_a); a_3g2 = get_3grams(t_a)
    num2 = set(re.findall(r'\d+', t_a))
    min_len = min(len(s1_nc), len(t_nc))
    pfx_len = 0
    if min_len > 0:
        for i in range(min_len):
            if s1_nc[i] == t_nc[i]: pfx_len += 1
            else: break
        pfx_ratio = pfx_len / min_len
    else: pfx_ratio = 0.0
    return [
        fast_jaccard(n_tok1, n_tok2), fast_jaccard(n_2g1, n_2g2), fast_jaccard(n_3g1, n_3g2),
        pfx_ratio, (min(len(s1_nc), len(t_nc)) + 1) / (max(len(s1_nc), len(t_nc)) + 1),
        len(n_tok1 & n_tok2) / (max(len(n_tok1), 1)),
        fast_jaccard(a_tok1, a_tok2), fast_jaccard(a_2g1, a_2g2), fast_jaccard(a_3g1, a_3g2),
        fast_jaccard(num1, num2), 1.0 if (not s1_ac or not t_ac) else 0.0,
        fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2),
        fast_jaccard(n_3g1 | a_3g1, n_3g2 | a_3g2), idf_score,
    ]

def macro_f05(gt_dict, pred_dict):
    scores = []
    for sid, tt in gt_dict.items():
        pp = set(pred_dict.get(sid, []))
        if not tt: scores.append(1.0 if not pp else 0.0)
        else:
            if not pp: scores.append(0.0)
            else:
                tp = len(pp & tt); p = tp / len(pp); r = tp / len(tt)
                d = 0.25 * p + r
                scores.append((1.25 * p * r) / d if d > 0 else 0.0)
    return float(np.mean(scores))

print("Loading data...", flush=True)
TOTAL_N = 30000; TRAIN_N = 20000; VAL_N = 10000

gt_map = {}
with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
        gt_map[p[0]] = set(m)
        if len(gt_map) >= TOTAL_N: break

s1_recs = {}
s1_needed = set(gt_map.keys())
with open(os.path.join(BASE, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed:
            s1_recs[p[0]] = (p[1], p[2], p[3])
            if len(s1_recs) == len(s1_needed): break

s1_list = list(s1_recs.keys())
train_ids = s1_list[:TRAIN_N]
val_ids   = s1_list[TRAIN_N:TOTAL_N]

all_needed = set()
for sid in s1_needed: all_needed |= gt_map[sid]

tgt_recs = {}
for sf in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(BASE, sf), encoding='utf-8') as f:
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

raw_inv = defaultdict(list)
for i in range(N):
    for tok in set(clean_toks(tgt_names[i], tgt_addrs[i])):
        raw_inv[tok].append(i)
inv_index = {tok: postings for tok, postings in raw_inv.items() if len(postings) <= 300}
del raw_inv
idf = {tok: math.log(N / (len(postings) + 1)) for tok, postings in inv_index.items()}

def query_blocker(name, addr):
    toks = clean_toks(name, addr)
    scores = Counter()
    for tok in toks:
        if tok in inv_index:
            w = idf[tok]
            for idx in inv_index[tok]:
                scores[idx] += w
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:15]

print("Training model...", flush=True)
X_train, y_train = [], []
for sid in train_ids:
    s1_n, s1_a, s1_c = s1_recs[sid]
    s1_pre = precompute_s1(s1_n, s1_a)
    true_tgt = gt_map[sid]
    for idx, idf_w in query_blocker(s1_n, s1_a):
        tid = tgt_id_list[idx]
        if tgt_ctry.get(tid) != s1_c: continue
        X_train.append(extract_features(s1_pre, tgt_names[idx], tgt_addrs[idx], idf_w))
        y_train.append(1 if tid in true_tgt else 0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

clf = HistGradientBoostingClassifier(
    max_iter=250, max_depth=6, learning_rate=0.08,
    min_samples_leaf=25, l2_regularization=0.5, random_state=42
)
clf.fit(X_train, y_train)

print("Scoring validation set...", flush=True)
val_feats, val_meta = [], []
for sid in val_ids:
    s1_n, s1_a, s1_c = s1_recs[sid]
    s1_pre = precompute_s1(s1_n, s1_a)
    for idx, idf_w in query_blocker(s1_n, s1_a):
        tid = tgt_id_list[idx]
        if tgt_ctry.get(tid) != s1_c: continue
        val_feats.append(extract_features(s1_pre, tgt_names[idx], tgt_addrs[idx], idf_w))
        val_meta.append((sid, tid))

val_feats = np.array(val_feats, dtype=np.float32)
probs = clf.predict_proba(val_feats)[:, 1]

gt_val = {sid: gt_map[sid] for sid in val_ids}

print("\n--- BASELINE: Single Global Threshold ---", flush=True)
best_single_f05 = 0.0
best_single_t = 0.0
for t in np.arange(0.50, 0.90, 0.05):
    scored = [(float(p), sid, tid) for (sid, tid), p in zip(val_meta, probs) if p >= t]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    assigned = set()
    preds = defaultdict(list)
    for p, sid, tid in scored:
        if tid not in assigned:
            preds[sid].append(tid)
            assigned.add(tid)
    f05 = macro_f05(gt_val, preds)
    if f05 > best_single_f05:
        best_single_f05 = f05
        best_single_t = t
    print(f"  Single tau = {t:.2f} -> Macro F0.5 = {f05:.4f}", flush=True)

print(f"\nBest Single Threshold: tau={best_single_t:.2f} -> F0.5 = {best_single_f05:.4f}", flush=True)

print("\n--- EXPERIMENT: Two-Regime Thresholding ---", flush=True)
print("  tau1 = threshold for 1st match (strict gate for singletons)")
print("  tau2 = threshold for 2nd+ matches (looser to capture multi-matches)")

best_f05 = 0.0
best_comb = (0, 0)

for tau1 in np.arange(0.65, 0.90, 0.05):
    for tau2 in np.arange(0.40, 0.80, 0.05):
        if tau2 > tau1: continue
        
        # We consider any pair >= tau2
        scored = [(float(p), sid, tid) for (sid, tid), p in zip(val_meta, probs) if p >= tau2]
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        
        assigned = set()
        preds = defaultdict(list)
        
        for p, sid, tid in scored:
            if tid in assigned:
                continue
            if len(preds[sid]) == 0:
                # First match must clear tau1
                if p >= tau1:
                    preds[sid].append(tid)
                    assigned.add(tid)
            else:
                # Subsequent matches only need to clear tau2
                if p >= tau2:
                    preds[sid].append(tid)
                    assigned.add(tid)
                    
        f05 = macro_f05(gt_val, preds)
        marker = "  ◄ NEW BEST" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_comb = (tau1, tau2)
        print(f"  tau1={tau1:.2f}, tau2={tau2:.2f} -> Macro F0.5 = {f05:.4f}{marker}", flush=True)

print(f"\n>>> COMPARISON RESULT <<<")
print(f"Single Threshold : tau={best_single_t:.2f} -> F0.5 = {best_single_f05:.4f}")
print(f"Two-Regime       : tau1={best_comb[0]:.2f}, tau2={best_comb[1]:.2f} -> F0.5 = {best_f05:.4f}")
diff = best_f05 - best_single_f05
print(f"Net Difference   : {diff:+.4f} F0.5")
