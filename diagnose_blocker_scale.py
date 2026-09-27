#!/usr/bin/env python3
"""
Diagnostic: Measure Blocker Recall vs MAX_POSTING on FULL Target Pool.
Tests on 5,000 held-out US validation entities against ALL 6,186,873 US targets.
Finds exact recall ceiling for MAX_POSTING in [300, 1000, 3000, 5000, 10000].
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
    return [w for w in re.sub(r'[^\w\s]', ' ', text).split() if len(w) >= 3 and w not in STOPWORDS]

print("1. Loading US Ground Truth...", flush=True)
gt_map = {}
s1_needed = set()
with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
        if m:
            gt_map[p[0]] = set(m)
            s1_needed.add(p[0])
            if len(gt_map) >= 20000: break

s1_recs = {}
with open(os.path.join(BASE, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed and len(p) >= 4 and p[3] == 'US':
            s1_recs[p[0]] = (p[1], p[2])
            if len(s1_recs) >= 5000: break

us_val_ids = list(s1_recs.keys())[:5000]
print(f"Loaded {len(us_val_ids):,} US validation entities with ground truth matches.", flush=True)

print("2. Loading ALL US targets (streamed)...", flush=True)
tgt_ids = []
tgt_names = []
tgt_addrs = []
t0 = time.time()
for sf in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(BASE, sf), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4 and p[3] == 'US':
                tgt_ids.append(p[0])
                tgt_names.append(p[1])
                tgt_addrs.append(p[2])

N_tgt = len(tgt_ids)
print(f"Loaded {N_tgt:,} US targets in {time.time()-t0:.1f}s", flush=True)

print("3. Building inverted index on US targets...", flush=True)
t0 = time.time()
raw_inv = defaultdict(list)
for i in range(N_tgt):
    for tok in set(clean_toks(tgt_names[i], tgt_addrs[i])):
        raw_inv[tok].append(i)
print(f"Raw inverted index: {len(raw_inv):,} unique tokens in {time.time()-t0:.1f}s", flush=True)

# Test various cutoffs
for max_p in [300, 1000, 3000, 5000, 10000]:
    t_start = time.time()
    # Filter
    inv = {tok: v for tok, v in raw_inv.items() if len(v) <= max_p}
    idf = {tok: math.log(N_tgt / (len(v) + 1)) for tok, v in inv.items()}
    vocab_size = len(inv)
    
    hits = 0
    total_tgts_found = 0
    all_cands = []
    
    for sid in us_val_ids:
        s1_n, s1_a = s1_recs[sid]
        true_tgts = gt_map[sid]
        
        scores = Counter()
        for tok in clean_toks(s1_n, s1_a):
            if tok in inv:
                w = idf[tok]
                for idx in inv[tok]:
                    scores[idx] += w
                    
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:15]
        retrieved = {tgt_ids[idx] for idx, w in top}
        all_cands.append(len(retrieved))
        
        # Check recall
        matched_in_gt = true_tgts & retrieved
        if matched_in_gt:
            hits += 1
            total_tgts_found += len(matched_in_gt)
            
    total_true_tgts = sum(len(gt_map[sid]) for sid in us_val_ids)
    entity_recall = hits / len(us_val_ids) * 100
    target_recall = total_tgts_found / total_true_tgts * 100
    avg_c = np.mean(all_cands)
    elapsed = time.time() - t_start
    
    print(f"MAX_POSTING = {max_p:>5} | Vocab: {vocab_size:>8,} | Entity Recall: {entity_recall:>6.2f}% | Target Recall: {target_recall:>6.2f}% | Avg Cands: {avg_c:>5.1f} | {elapsed:.1f}s", flush=True)
