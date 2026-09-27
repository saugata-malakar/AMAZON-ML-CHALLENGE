#!/usr/bin/env python3
"""
Test Idea #3: Adaptive Candidate Retrieval Depth.
Instead of flat Top-15:
Keep top candidate always.
Keep candidate i (up to K=15) only if score[i] >= alpha * score[0].
Measure:
1. Blocking Recall Ceiling
2. Average Candidates per Entity
for alpha in [0.0 (flat 15), 0.3, 0.4, 0.5, 0.6, 0.7].
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

print("Loading data for Idea #3 test...", flush=True)
gt_map = {}
with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
        gt_map[p[0]] = set(m)
        if len(gt_map) >= 30000: break

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
val_ids = s1_list[20000:30000]

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

val_with_match = [sid for sid in val_ids if gt_map[sid]]
print(f"Evaluating {len(val_with_match):,} validation entities with true matches...", flush=True)

print(f"\n{'Rule':<30} {'Recall Ceiling':>15} {'Avg Cands/S1':>15} {'Reduction Ratio':>18}")
print("-" * 80)

# Evaluate different alpha thresholds
# Candidate i kept if score_i >= alpha * score_0, capped at max_k=15
for alpha in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
    hits = 0
    all_cand_sizes = []
    
    for sid in val_with_match:
        s1_n, s1_a, s1_c = s1_recs[sid]
        true_tgts = gt_map[sid]
        
        toks = clean_toks(s1_n, s1_a)
        scores = Counter()
        for tok in toks:
            if tok in inv_index:
                w = idf[tok]
                for idx in inv_index[tok]:
                    scores[idx] += w
        
        sorted_cands = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:15]
        
        # Filter same-country
        country_cands = [(idx, w) for idx, w in sorted_cands if tgt_ctry.get(tgt_id_list[idx]) == s1_c]
        
        if not country_cands:
            all_cand_sizes.append(0)
            continue
            
        top_score = country_cands[0][1]
        threshold_score = alpha * top_score
        
        selected_tids = [tgt_id_list[idx] for idx, w in country_cands if w >= threshold_score]
        all_cand_sizes.append(len(selected_tids))
        
        if true_tgts & set(selected_tids):
            hits += 1
            
    recall = hits / len(val_with_match) * 100
    avg_cands = np.mean(all_cand_sizes)
    reduction = N / max(avg_cands, 0.001)
    label = f"Flat Top-15 (alpha=0.0)" if alpha == 0.0 else f"Adaptive alpha={alpha:.1f}"
    print(f"{label:<30} {recall:>14.2f}% {avg_cands:>15.2f} {reduction:>16.0f}:1")
