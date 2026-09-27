#!/usr/bin/env python3
"""
v6 Production Pipeline: Dual-Blocker Union
= IDF token blocker (K=15) ∪ Char 3-gram TF-IDF blocker (K=15)
Union candidate set fed to same HistGBDT classifier.
Same greedy 1-to-1 assignment, same threshold τ=0.70.
"""
import sys, os, re, math, time, gc
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR  = os.path.join(BASE, "dataset", "test")
OUT_DIR   = os.path.join(BASE, "output")
TEMP_DIR  = os.path.join(r"c:\Users\Administrator\Downloads\AMAZON", "temp_v6")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

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

def get_3grams_str(text):
    t = re.sub(r'[^\w]', '', text.lower())
    return list(t[i:i+3] for i in range(len(t)-2)) if len(t) >= 3 else []

def get_words(text):
    return set(re.findall(r'\w+', text.lower()))
def get_2grams(text):
    t = text.lower()
    return set(t[i:i+2] for i in range(len(t)-1)) if len(t)>=2 else set()
def get_3grams(text):
    t = text.lower()
    return set(t[i:i+3] for i in range(len(t)-2)) if len(t)>=3 else set()
def fast_jaccard(s1, s2):
    return len(s1&s2)/len(s1|s2) if (s1 and s2) else 0.0

def precompute_s1(n, a):
    nc = re.sub(r'[^\w\s]',' ',n.lower()).strip()
    ac = re.sub(r'[^\w\s]',' ',a.lower()).strip()
    return (get_words(n),get_2grams(n),get_3grams(n),
            get_words(a),get_2grams(a),get_3grams(a),
            set(re.findall(r'\d+',a)),nc,ac)

def extract_features(s1_pre, t_n, t_a, idf_score):
    (n_tok1,n_2g1,n_3g1,a_tok1,a_2g1,a_3g1,num1,s1_nc,s1_ac) = s1_pre
    t_nc=re.sub(r'[^\w\s]',' ',t_n.lower()).strip()
    t_ac=re.sub(r'[^\w\s]',' ',t_a.lower()).strip()
    n_tok2=get_words(t_n);n_2g2=get_2grams(t_n);n_3g2=get_3grams(t_n)
    a_tok2=get_words(t_a);a_2g2=get_2grams(t_a);a_3g2=get_3grams(t_a)
    num2=set(re.findall(r'\d+',t_a))
    min_len=min(len(s1_nc),len(t_nc)); pfx_len=0
    if min_len>0:
        for i in range(min_len):
            if s1_nc[i]==t_nc[i]: pfx_len+=1
            else: break
        pfx_ratio=pfx_len/min_len
    else: pfx_ratio=0.0
    return [
        fast_jaccard(n_tok1,n_tok2),fast_jaccard(n_2g1,n_2g2),fast_jaccard(n_3g1,n_3g2),
        pfx_ratio,(min(len(s1_nc),len(t_nc))+1)/(max(len(s1_nc),len(t_nc))+1),
        len(n_tok1&n_tok2)/(max(len(n_tok1),1)),
        fast_jaccard(a_tok1,a_tok2),fast_jaccard(a_2g1,a_2g2),fast_jaccard(a_3g1,a_3g2),
        fast_jaccard(num1,num2),1.0 if (not s1_ac or not t_ac) else 0.0,
        fast_jaccard(n_tok1|a_tok1,n_tok2|a_tok2),
        fast_jaccard(n_3g1|a_3g1,n_3g2|a_3g2),idf_score,
    ]

def macro_f05(gt_dict, pred_dict):
    scores=[]
    for s1_id,true_targets in gt_dict.items():
        preds=set(pred_dict.get(s1_id,[]))
        if not true_targets: scores.append(1.0 if not preds else 0.0)
        else:
            if not preds: scores.append(0.0)
            else:
                tp=len(preds&true_targets); p=tp/len(preds); r=tp/len(true_targets)
                d=0.25*p+r; scores.append((1.25*p*r)/d if d>0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

# ============================================================
# TRAINING
# ============================================================
print("="*60, flush=True)
print("v6 Dual-Blocker Pipeline", flush=True)
print("="*60, flush=True)

TRAIN_N = 40000
VAL_N   = 20000

print(f"\n[1/5] Loading training data...", flush=True)
t0 = time.time()
gt_map = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p)>1 else []
        gt_map[p[0]] = set(m)
        if len(gt_map) >= TRAIN_N+VAL_N: break

s1_recs = {}
s1_needed = set(gt_map.keys())
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed:
            s1_recs[p[0]] = (p[1], p[2], p[3])
            if len(s1_recs)==len(s1_needed): break

all_needed = set()
for m in gt_map.values(): all_needed |= m
tgt_recs = {}
for sf in ["train_source2.tsv","train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, sf), encoding='utf-8') as f:
        f.readline()
        for i,line in enumerate(f):
            p = line.rstrip('\n').split('\t')
            if p[0] in all_needed or i < 500000:
                tgt_recs[p[0]] = (p[1],p[2],p[3])

tgt_id_list = list(tgt_recs.keys())
N = len(tgt_id_list)
tgt_names = [tgt_recs[t][0] for t in tgt_id_list]
tgt_addrs = [tgt_recs[t][1] for t in tgt_id_list]
tgt_ctry  = {t: tgt_recs[t][2] for t in tgt_id_list}
print(f"  Target pool: {N:,} | elapsed: {time.time()-t0:.1f}s", flush=True)

print(f"\n[2/5] Building dual index on training target pool...", flush=True)
t0 = time.time()
# IDF token index
raw_inv = defaultdict(list)
for i in range(N):
    for tok in set(clean_toks(tgt_names[i], tgt_addrs[i])):
        raw_inv[tok].append(i)
inv_index = {tok: v for tok,v in raw_inv.items() if len(v)<=300}
idf = {tok: math.log(N/(len(v)+1)) for tok,v in inv_index.items()}
del raw_inv; gc.collect()

# Char 3-gram index
cg_inv = defaultdict(list)
for i in range(N):
    for g in set(get_3grams_str(tgt_names[i]+" "+tgt_addrs[i])):
        cg_inv[g].append(i)
cg_inv_f = {g: v for g,v in cg_inv.items() if 2<=len(v)<=500}
cg_idf   = {g: math.log(N/(len(v)+1)) for g,v in cg_inv_f.items()}
del cg_inv; gc.collect()
print(f"  IDF index: {len(inv_index):,} tokens | 3-gram index: {len(cg_inv_f):,} grams | {time.time()-t0:.1f}s", flush=True)

def query_dual(name, addr, country, k=15):
    """Return union of IDF token + char 3-gram candidates, country-filtered."""
    # Blocker 1: IDF token
    scores1 = Counter()
    for tok in clean_toks(name, addr):
        if tok in inv_index:
            w = idf[tok]
            for idx in inv_index[tok]: scores1[idx] += w
    cands1 = {(tgt_id_list[idx], w) for idx,w in scores1.most_common(k) if tgt_ctry.get(tgt_id_list[idx])==country}

    # Blocker 2: Char 3-gram
    scores2 = Counter()
    for g in set(get_3grams_str(name+" "+addr)):
        if g in cg_inv_f:
            w = cg_idf[g]
            for idx in cg_inv_f[g]: scores2[idx] += w
    cands2 = {(tgt_id_list[idx], w) for idx,w in scores2.most_common(k) if tgt_ctry.get(tgt_id_list[idx])==country}

    # Union: keep best IDF score per entity (for features), prefer blocker1 score
    merged = {}
    for tid, w in cands2:
        merged[tid] = w
    for tid, w in cands1:
        merged[tid] = w  # blocker1 overwrites — its score used for IDF feature
    return list(merged.items())  # [(tid, idf_w), ...]

print(f"\n[3/5] Building training pairs...", flush=True)
t0 = time.time()
s1_list = list(s1_recs.keys())
train_ids = set(s1_list[:TRAIN_N])

X_train, y_train = [], []
for sid in s1_list[:TRAIN_N]:
    s1_n, s1_a, s1_c = s1_recs[sid]
    s1_pre = precompute_s1(s1_n, s1_a)
    true_tgt = gt_map[sid]
    for tid, idf_w in query_dual(s1_n, s1_a, s1_c, k=15):
        feat = extract_features(s1_pre, tgt_recs[tid][0], tgt_recs[tid][1], idf_w)
        X_train.append(feat); y_train.append(1 if tid in true_tgt else 0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"  Training pairs: {len(X_train):,} (pos={y_train.sum():,}) | {time.time()-t0:.1f}s", flush=True)

print(f"\n[4/5] Training classifier...", flush=True)
t0 = time.time()
pos_w = (y_train==0).sum() / max((y_train==1).sum(), 1)
sw = np.where(y_train==1, pos_w, 1.0)
clf = HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=0.08,
                                      min_samples_leaf=25, l2_regularization=0.5, random_state=42)
clf.fit(X_train, y_train, sample_weight=sw)
print(f"  Trained in {time.time()-t0:.1f}s", flush=True)

# Validation
print(f"\n[4b] Validation F0.5 on {VAL_N} held-out entities...", flush=True)
t0 = time.time()
val_feats, val_labels, val_meta = [], [], []
for sid in s1_list[TRAIN_N:TRAIN_N+VAL_N]:
    s1_n, s1_a, s1_c = s1_recs[sid]
    s1_pre = precompute_s1(s1_n, s1_a)
    for tid, idf_w in query_dual(s1_n, s1_a, s1_c, k=15):
        feat = extract_features(s1_pre, tgt_recs[tid][0], tgt_recs[tid][1], idf_w)
        val_feats.append(feat); val_labels.append(1 if tid in gt_map[sid] else 0)
        val_meta.append((sid, tid))

val_feats = np.array(val_feats, dtype=np.float32)
probs = clf.predict_proba(val_feats)[:,1]

best_tau, best_f05 = 0.70, 0.0
for tau in np.arange(0.40, 0.90, 0.02):
    scored = sorted(zip(probs, [m[0] for m in val_meta], [m[1] for m in val_meta]), reverse=True)
    assigned=set(); preds=defaultdict(list)
    for p, sid, tid in scored:
        if p < tau: break
        if tid not in assigned:
            preds[sid].append(tid); assigned.add(tid)
    gt_val = {sid: gt_map[sid] for sid in s1_list[TRAIN_N:TRAIN_N+VAL_N]}
    f05 = macro_f05(gt_val, preds)
    if f05 > best_f05: best_f05=f05; best_tau=tau
    print(f"  tau={tau:.2f}: F0.5={f05:.4f}", flush=True)

print(f"\n  >>> Best tau={best_tau:.2f}, F0.5={best_f05:.4f} (v5 was 0.7910) <<<", flush=True)
print(f"  Validation took {time.time()-t0:.1f}s", flush=True)

print("\n[5/5] Test inference — saving model and index for test run", flush=True)
with open(os.path.join(TEMP_DIR, "clf_v6.pkl"), "wb") as f:
    pickle.dump((clf, best_tau), f)
print("Model saved to temp_v6/clf_v6.pkl", flush=True)
print("Run run_test_v6.py to complete test inference.", flush=True)
print(f"\nDONE. Best F0.5={best_f05:.4f} at tau={best_tau:.2f}", flush=True)
