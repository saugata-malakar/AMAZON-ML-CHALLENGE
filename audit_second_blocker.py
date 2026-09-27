#!/usr/bin/env python3
"""
Point 1: Second blocking generator — char 3-gram TF-IDF cosine blocker.
Union with existing IDF token blocker. Measure recall ceiling at K=15 per generator.
"""
import sys, os, re, math, time
from collections import defaultdict, Counter
import numpy as np

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

def get_3grams_str(text):
    t = re.sub(r'[^\w]', '', text.lower())
    return list(t[i:i+3] for i in range(len(t)-2)) if len(t) >= 3 else []

TRAIN_N = 30000
print("Loading GT + data...", flush=True)
gt_map = {}
with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p)>1 else []
        gt_map[p[0]] = set(m)
        if len(gt_map) >= TRAIN_N: break

s1_recs = {}
s1_needed = set(gt_map.keys())
with open(os.path.join(BASE, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed:
            s1_recs[p[0]] = (p[1], p[2], p[3])
            if len(s1_recs) == len(s1_needed): break

all_needed = set()
for m in gt_map.values(): all_needed |= m

tgt_recs = {}
for sf in ["train_source2.tsv","train_source3.tsv"]:
    with open(os.path.join(BASE, sf), encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip('\n').split('\t')
            if p[0] in all_needed or i < 500000:
                tgt_recs[p[0]] = (p[1], p[2], p[3])

tgt_id_list = list(tgt_recs.keys())
N = len(tgt_id_list)
print(f"Target pool: {N:,}", flush=True)

# ===== BLOCKER 1: Distinctive IDF token blocker (existing) =====
print("Building IDF token index...", flush=True)
raw_inv = defaultdict(list)
for i, tid in enumerate(tgt_id_list):
    for tok in set(clean_toks(tgt_recs[tid][0], tgt_recs[tid][1])):
        raw_inv[tok].append(i)
inv_index = {tok: postings for tok, postings in raw_inv.items() if len(postings) <= 300}
idf = {tok: math.log(N / (len(v)+1)) for tok, v in inv_index.items()}
del raw_inv
print(f"  IDF index: {len(inv_index):,} tokens", flush=True)

def query_idf(name, addr, k):
    scores = Counter()
    for tok in clean_toks(name, addr):
        if tok in inv_index:
            w = idf[tok]
            for idx in inv_index[tok]:
                scores[idx] += w
    return {tgt_id_list[idx] for idx, _ in scores.most_common(k)}

# ===== BLOCKER 2: Char 3-gram TF-IDF cosine (new) =====
print("Building char 3-gram TF-IDF index...", flush=True)
t0 = time.time()
cg_inv = defaultdict(list)
for i, tid in enumerate(tgt_id_list):
    n, a, _ = tgt_recs[tid]
    grams = set(get_3grams_str(n + " " + a))
    for g in grams:
        cg_inv[g].append(i)
    if i % 100000 == 0:
        print(f"  {i:,}/{N:,}...", flush=True)

# keep only grams appearing in [2, 500] targets
cg_inv_f = {g: v for g, v in cg_inv.items() if 2 <= len(v) <= 500}
cg_idf = {g: math.log(N / (len(v)+1)) for g, v in cg_inv_f.items()}
del cg_inv
print(f"  Char 3-gram index: {len(cg_inv_f):,} grams ({time.time()-t0:.1f}s)", flush=True)

def query_cgram(name, addr, k):
    text = name + " " + addr
    grams = set(get_3grams_str(text))
    scores = Counter()
    for g in grams:
        if g in cg_inv_f:
            w = cg_idf[g]
            for idx in cg_inv_f[g]:
                scores[idx] += w
    return {tgt_id_list[idx] for idx, _ in scores.most_common(k)}

# ===== Evaluate: recall at each K for both blockers and their union =====
print("\nMeasuring recall ceiling...", flush=True)

# Use val entities (last 5000 of s1_list)
s1_list = list(s1_recs.keys())
val_ids = s1_list[25000:30000]

for K in [10, 15, 20]:
    tok_hits = 0; cg_hits = 0; union_hits = 0; total_with_match = 0
    tok_sizes = []; cg_sizes = []; union_sizes = []

    for sid in val_ids:
        s1_n, s1_a, s1_c = s1_recs[sid]
        true_tgts = gt_map[sid]
        if not true_tgts: continue
        total_with_match += 1

        # blocker 1
        tok_cands = {t for t in query_idf(s1_n, s1_a, K)
                     if tgt_recs[t][2] == s1_c}
        # blocker 2
        cg_cands = {t for t in query_cgram(s1_n, s1_a, K)
                    if tgt_recs[t][2] == s1_c}
        union_cands = tok_cands | cg_cands

        tok_sizes.append(len(tok_cands))
        cg_sizes.append(len(cg_cands))
        union_sizes.append(len(union_cands))

        if true_tgts & tok_cands: tok_hits += 1
        if true_tgts & cg_cands: cg_hits += 1
        if true_tgts & union_cands: union_hits += 1

    print(f"\nK={K} per generator, union candidate set:", flush=True)
    print(f"  IDF token blocker     : recall={tok_hits/total_with_match*100:.2f}%  avg_cands={np.mean(tok_sizes):.1f}", flush=True)
    print(f"  Char 3-gram blocker   : recall={cg_hits/total_with_match*100:.2f}%  avg_cands={np.mean(cg_sizes):.1f}", flush=True)
    print(f"  UNION (both combined) : recall={union_hits/total_with_match*100:.2f}%  avg_cands={np.mean(union_sizes):.1f}", flush=True)
    print(f"  Recall gain over IDF alone: +{(union_hits-tok_hits)/total_with_match*100:.2f} pp", flush=True)
    print(f"  Candidate size increase: +{np.mean(union_sizes)-np.mean(tok_sizes):.1f} avg cands", flush=True)

# ===== Diagnose the unrecovered 15% =====
print("\n\nDiagnosing entities missed by BOTH blockers at K=15...", flush=True)
missed_examples = []
for sid in val_ids:
    s1_n, s1_a, s1_c = s1_recs[sid]
    true_tgts = gt_map[sid]
    if not true_tgts: continue
    tok_cands = {t for t in query_idf(s1_n, s1_a, 15) if tgt_recs[t][2] == s1_c}
    cg_cands  = {t for t in query_cgram(s1_n, s1_a, 15) if tgt_recs[t][2] == s1_c}
    union = tok_cands | cg_cands
    if not (true_tgts & union):
        for tgt_id in list(true_tgts)[:1]:
            if tgt_id in tgt_recs:
                tgt_n, tgt_a, _ = tgt_recs[tgt_id]
                missed_examples.append({
                    's1_name': s1_n, 's1_addr': s1_a,
                    'tgt_name': tgt_n, 'tgt_addr': tgt_a
                })
    if len(missed_examples) >= 20: break

print(f"Sample of missed pairs (both blockers fail):", flush=True)
for ex in missed_examples[:10]:
    print(f"  S1:  '{ex['s1_name']}' | '{ex['s1_addr']}'", flush=True)
    print(f"  TGT: '{ex['tgt_name']}' | '{ex['tgt_addr']}'", flush=True)
    print(f"  ---", flush=True)
