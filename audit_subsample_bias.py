#!/usr/bin/env python3
"""
Subsample origin + representativeness check.
1. What order does train_ground_truth.tsv use? Is it sorted by ID, country, random?
2. What's the country distribution of first-40k vs a random 40k vs full GT?
3. Re-measure recall ceiling on a properly random sample of 5000 val entities
   drawn from a much larger pool (use reservoir sampling over full GT).
"""
import sys, os, re, math, random
from collections import defaultdict, Counter
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train"
random.seed(42)
np.random.seed(42)

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

# ================================================================
# STEP 1: Scan full train_ground_truth.tsv for structure
# ================================================================
print("Scanning full train_ground_truth.tsv...", flush=True)

all_s1_ids = []
match_counts = []
has_match = 0
with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_id = p[0]
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p)>1 else []
        all_s1_ids.append(s1_id)
        match_counts.append(len(m))
        if m: has_match += 1

total = len(all_s1_ids)
print(f"  Total S1 entities in GT: {total:,}", flush=True)
print(f"  Entities with ≥1 match: {has_match:,} ({has_match/total*100:.1f}%)", flush=True)
print(f"  Singletons: {total-has_match:,} ({(total-has_match)/total*100:.1f}%)", flush=True)
print(f"  First 10 IDs: {all_s1_ids[:10]}", flush=True)
print(f"  Last 10 IDs:  {all_s1_ids[-10:]}", flush=True)

# Check if sorted by ID (lexicographic)
is_sorted = all(all_s1_ids[i] <= all_s1_ids[i+1] for i in range(min(10000, len(all_s1_ids)-1)))
print(f"  First 10k IDs appear sorted: {is_sorted}", flush=True)

# Country distribution of first 40k vs random 40k
print("\nLoading S1 country labels...", flush=True)
s1_country = {}
with open(os.path.join(BASE, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4:
            s1_country[p[0]] = p[3]

# First 40k
first40k_ctry = Counter(s1_country.get(sid, 'unknown') for sid in all_s1_ids[:40000])
# Random 40k (reservoir sample)
rng_sample = random.sample(range(total), min(40000, total))
rand40k_ctry = Counter(s1_country.get(all_s1_ids[i], 'unknown') for i in rng_sample)
# Full GT
full_ctry = Counter(s1_country.get(sid, 'unknown') for sid in all_s1_ids)

print("\nCountry distribution comparison:", flush=True)
print(f"{'Country':<10} {'First 40k':>12} {'Random 40k':>12} {'Full GT':>12}")
for c in sorted(full_ctry.keys()):
    f40 = first40k_ctry.get(c,0)/400
    r40 = rand40k_ctry.get(c,0)/400
    fg = full_ctry.get(c,0)/total*100
    print(f"  {c:<8} {f40:>11.1f}%  {r40:>11.1f}%  {fg:>11.1f}%", flush=True)

# Match-rate distribution: first 40k vs random 40k
first40k_match_rate = sum(1 for i in range(40000) if match_counts[i]>0) / 400
rand40k_match_rate  = sum(1 for i in rng_sample if match_counts[i]>0) / 400
full_match_rate = has_match / total * 100
print(f"\nMatch rate (% with ≥1 match):", flush=True)
print(f"  First 40k: {first40k_match_rate:.1f}%", flush=True)
print(f"  Random 40k: {rand40k_match_rate:.1f}%", flush=True)
print(f"  Full GT: {full_match_rate:.1f}%", flush=True)

# ================================================================
# STEP 2: Recall ceiling on a properly RANDOM 5000-entity val sample
# drawn from a much larger pool (200k random) to avoid first-N bias
# ================================================================
print("\n\nDrawing random val sample from full GT...", flush=True)
RAND_POOL = 200000
RAND_VAL  = 5000

# Draw 200k random IDs, build GT for them
rng_pool_idx = sorted(random.sample(range(total), min(RAND_POOL, total)))
rng_pool_ids = [all_s1_ids[i] for i in rng_pool_idx]
rng_gt_needed_ids = set(rng_pool_ids)

gt_map = {}
needed_tgts = set()
with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in rng_gt_needed_ids:
            m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p)>1 else []
            gt_map[p[0]] = set(m)
            needed_tgts |= set(m)

print(f"  Pool: {len(gt_map):,} entities, {len(needed_tgts):,} needed targets", flush=True)

# Load S1 recs for the pool
s1_recs = {}
with open(os.path.join(BASE, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in rng_gt_needed_ids:
            s1_recs[p[0]] = (p[1], p[2], p[3])

# Load target recs (needed + first 500k for index)
tgt_recs = {}
for sf in ["train_source2.tsv","train_source3.tsv"]:
    with open(os.path.join(BASE, sf), encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip('\n').split('\t')
            if p[0] in needed_tgts or i < 500000:
                tgt_recs[p[0]] = (p[1], p[2], p[3])

tgt_id_list = list(tgt_recs.keys())
N = len(tgt_id_list)
tgt_ctry_map = {t: tgt_recs[t][2] for t in tgt_id_list}
print(f"  Target pool for index: {N:,}", flush=True)

# Build IDF token index
raw_inv = defaultdict(list)
for i, tid in enumerate(tgt_id_list):
    for tok in set(clean_toks(tgt_recs[tid][0], tgt_recs[tid][1])):
        raw_inv[tok].append(i)
inv_index = {tok: v for tok,v in raw_inv.items() if len(v)<=300}
idf = {tok: math.log(N/(len(v)+1)) for tok,v in inv_index.items()}
del raw_inv
print(f"  IDF index: {len(inv_index):,} tokens", flush=True)

def query_idf(name, addr, k):
    scores = Counter()
    for tok in clean_toks(name, addr):
        if tok in inv_index:
            w = idf[tok]
            for idx in inv_index[tok]: scores[idx] += w
    return {tgt_id_list[idx] for idx, _ in scores.most_common(k)}

# Val: random 5000 from the random pool (with matches only)
val_with_match = [sid for sid in rng_pool_ids[:RAND_POOL] if gt_map.get(sid) and sid in s1_recs]
random.shuffle(val_with_match)
val_ids = val_with_match[:RAND_VAL]

print(f"\nRecall ceiling on RANDOM {len(val_ids):,}-entity val sample:", flush=True)
for k in [10, 15, 20]:
    hits = 0
    cand_sizes = []
    for sid in val_ids:
        s1_n, s1_a, s1_c = s1_recs[sid]
        cands = {t for t in query_idf(s1_n, s1_a, k) if tgt_ctry_map.get(t)==s1_c}
        cand_sizes.append(len(cands))
        if gt_map[sid] & cands: hits += 1
    recall = hits / len(val_ids) * 100
    print(f"  K={k}: recall={recall:.2f}%  avg_cands={np.mean(cand_sizes):.1f}", flush=True)

print("\nCountry distribution of random val sample:", flush=True)
val_ctry = Counter(s1_recs[sid][2] for sid in val_ids if sid in s1_recs)
for c,n in val_ctry.most_common():
    print(f"  {c}: {n} ({n/len(val_ids)*100:.1f}%)", flush=True)
