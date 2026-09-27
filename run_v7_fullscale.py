#!/usr/bin/env python3
"""
v7 FULL-SCALE Pipeline: Train on realistic target pool sizes, re-tune threshold.
Key changes from v5:
1. Train on 100k S1 entities (not 45k)
2. Load ALL training targets (~10.3M) for index building (not 1M)
3. Build index PER COUNTRY to match test-time regime
4. Re-tune threshold on held-out val at realistic scale
5. Adaptive candidate depth (alpha=0.3) to reduce candidate count
"""
import time, os, sys, re, gc, math, pickle
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR  = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE, "output")
TEMP_DIR  = os.path.join(BASE, "temp_v7")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 15
MAX_POSTING = 300
ALPHA_ADAPTIVE = 0.3  # Adaptive candidate pruning

STOPWORDS = {
    'inc','corp','corporation','llc','ltd','limited','pvt','private','co','company',
    'and','the','of','in','at','road','rd','street','st','avenue','ave','lane','ln',
    'drive','dr','nagar','colony','floor','near','opp','opposite','block','sector',
    'phase','house','plot','door','no','null','sarl','sas','sci','france','de','la',
    'le','du','des','les','en','rue','bd','boulevard','av','impasse','new','old',
    'east','west','north','south','main','cross'
}

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def clean_toks(n, a):
    text = (n + " " + a).lower()
    return [w for w in re.sub(r'[^\w\s]', ' ', text).split() if len(w) >= 3 and w not in STOPWORDS]
def get_words(t): return set(re.findall(r'\w+', t.lower()))
def get_2grams(t):
    t2 = t.lower()
    return set(t2[i:i+2] for i in range(len(t2)-1)) if len(t2)>=2 else set()
def get_3grams(t):
    t3 = t.lower()
    return set(t3[i:i+3] for i in range(len(t3)-2)) if len(t3)>=3 else set()
def jac(a,b): return len(a&b)/len(a|b) if (a and b) else 0.0

def precompute_s1(n, a):
    nc = re.sub(r'[^\w\s]',' ', n.lower()).strip()
    ac = re.sub(r'[^\w\s]',' ', a.lower()).strip()
    return (get_words(n), get_2grams(n), get_3grams(n),
            get_words(a), get_2grams(a), get_3grams(a),
            set(re.findall(r'\d+', a)), nc, ac)

def extract_features(s1p, tn, ta, idf_w):
    (nt1,n2g1,n3g1,at1,a2g1,a3g1,num1,s1nc,s1ac) = s1p
    tnc = re.sub(r'[^\w\s]',' ', tn.lower()).strip()
    tac = re.sub(r'[^\w\s]',' ', ta.lower()).strip()
    nt2=get_words(tn); n2g2=get_2grams(tn); n3g2=get_3grams(tn)
    at2=get_words(ta); a2g2=get_2grams(ta); a3g2=get_3grams(ta)
    num2=set(re.findall(r'\d+', ta))
    ml=min(len(s1nc),len(tnc))
    pfx=0
    if ml>0:
        for i in range(ml):
            if s1nc[i]==tnc[i]: pfx+=1
            else: break
        pfx /= ml
    return [jac(nt1,nt2), jac(n2g1,n2g2), jac(n3g1,n3g2), pfx,
            (min(len(s1nc),len(tnc))+1)/(max(len(s1nc),len(tnc))+1),
            len(nt1&nt2)/max(len(nt1),1),
            jac(at1,at2), jac(a2g1,a2g2), jac(a3g1,a3g2),
            jac(num1,num2), 1.0 if (not s1ac or not tac) else 0.0,
            jac(nt1|at1,nt2|at2), jac(n3g1|a3g1,n3g2|a3g2), idf_w]

def macro_f05(gt_dict, pred_dict):
    scores = []
    for sid, tt in gt_dict.items():
        pp = set(pred_dict.get(sid, []))
        if not tt: scores.append(1.0 if not pp else 0.0)
        else:
            if not pp: scores.append(0.0)
            else:
                tp=len(pp&tt); p=tp/len(pp); r=tp/len(tt)
                d = 0.25*p + r
                scores.append((1.25*p*r)/d if d>0 else 0.0)
    return float(np.mean(scores))

# =====================================================
# PHASE 1: LOAD ALL TRAINING DATA AT FULL SCALE
# =====================================================
log("=" * 60)
log("v7 FULL-SCALE PIPELINE")
log("=" * 60)

# Load GT — use 200k entities (100k train + 100k val)
TOTAL_N = 200000
TRAIN_N = 100000

log(f"Loading first {TOTAL_N:,} GT entries...")
t0 = time.time()
gt_map = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p)>1 else []
        gt_map[p[0]] = set(m)
        if len(gt_map) >= TOTAL_N: break

s1_needed = set(gt_map.keys())
log(f"  GT loaded: {len(gt_map):,} entities in {time.time()-t0:.1f}s")

# Load S1 records
log("Loading S1 records...")
t0 = time.time()
s1_recs = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed:
            s1_recs[p[0]] = (p[1], p[2], p[3])
            if len(s1_recs) == len(s1_needed): break
log(f"  S1 loaded: {len(s1_recs):,} in {time.time()-t0:.1f}s")

s1_list = list(s1_recs.keys())
train_ids = s1_list[:TRAIN_N]
val_ids   = s1_list[TRAIN_N:TOTAL_N]

# Determine all needed target IDs from GT
all_needed_tgts = set()
for sid in s1_needed:
    all_needed_tgts |= gt_map[sid]

# =====================================================
# PHASE 2: LOAD FULL TARGET POOL (ALL RECORDS)
# =====================================================
log("Loading FULL S2+S3 target pool (ALL records, not subsampled)...")
t0 = time.time()
tgt_recs = {}
for sf in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, sf), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4:
                tgt_recs[p[0]] = (p[1], p[2], p[3])
log(f"  FULL target pool: {len(tgt_recs):,} records in {time.time()-t0:.1f}s")

# =====================================================
# PHASE 3: TRAIN PER COUNTRY (matching test-time regime)
# =====================================================
# Group S1 and targets by country
countries = ['US', 'India']
s1_by_country = defaultdict(list)
for sid in train_ids:
    s1_by_country[s1_recs[sid][2]].append(sid)

val_by_country = defaultdict(list)
for sid in val_ids:
    val_by_country[s1_recs[sid][2]].append(sid)

X_all, y_all = [], []

for country in countries:
    log(f"\n--- Building training pairs for {country} ---")
    t0 = time.time()
    
    # Filter targets by country
    ctry_tgt_ids = []
    ctry_tgt_names = []
    ctry_tgt_addrs = []
    for tid, (tn, ta, tc) in tgt_recs.items():
        if tc == country:
            ctry_tgt_ids.append(tid)
            ctry_tgt_names.append(tn)
            ctry_tgt_addrs.append(ta)
    
    N_ctry = len(ctry_tgt_ids)
    log(f"  {country}: {N_ctry:,} targets, {len(s1_by_country[country]):,} train S1")
    
    # Build inverted index on country targets
    raw_inv = defaultdict(list)
    for i in range(N_ctry):
        for tok in set(clean_toks(ctry_tgt_names[i], ctry_tgt_addrs[i])):
            raw_inv[tok].append(i)
    inv_index = {tok: v for tok, v in raw_inv.items() if len(v) <= MAX_POSTING}
    idf = {tok: math.log(N_ctry / (len(v) + 1)) for tok, v in inv_index.items()}
    del raw_inv
    log(f"  Index: {len(inv_index):,} tokens in {time.time()-t0:.1f}s")
    
    # Generate training pairs
    t0 = time.time()
    n_pairs = 0
    for sid in s1_by_country[country]:
        s1_n, s1_a, s1_c = s1_recs[sid]
        s1_pre = precompute_s1(s1_n, s1_a)
        true_tgt = gt_map[sid]
        
        scores = Counter()
        for tok in clean_toks(s1_n, s1_a):
            if tok in inv_index:
                w = idf[tok]
                for idx in inv_index[tok]:
                    scores[idx] += w
        
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:TOP_K]
        
        for idx, idf_w in top:
            tid = ctry_tgt_ids[idx]
            feat = extract_features(s1_pre, ctry_tgt_names[idx], ctry_tgt_addrs[idx], idf_w)
            X_all.append(feat)
            y_all.append(1 if tid in true_tgt else 0)
            n_pairs += 1
    
    log(f"  {country}: {n_pairs:,} pairs in {time.time()-t0:.1f}s")
    
    # Free country-specific data
    del ctry_tgt_ids, ctry_tgt_names, ctry_tgt_addrs, inv_index, idf
    gc.collect()

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)
log(f"\nTotal training pairs: {len(X_all):,} (pos: {y_all.sum():,}, neg: {(y_all==0).sum():,})")

# =====================================================
# PHASE 4: TRAIN CLASSIFIER
# =====================================================
log("Training HistGradientBoostingClassifier...")
t0 = time.time()
clf = HistGradientBoostingClassifier(
    max_iter=300, max_depth=7, learning_rate=0.05,
    min_samples_leaf=20, l2_regularization=0.3, random_state=42
)
clf.fit(X_all, y_all)
log(f"  Trained in {time.time()-t0:.1f}s")
del X_all, y_all
gc.collect()

# =====================================================
# PHASE 5: VALIDATE ON HELD-OUT SET AT FULL SCALE
# =====================================================
log("\nValidating on held-out set at FULL TARGET SCALE...")
val_feats, val_meta = [], []

for country in countries:
    t0 = time.time()
    ctry_tgt_ids = []
    ctry_tgt_names = []
    ctry_tgt_addrs = []
    for tid, (tn, ta, tc) in tgt_recs.items():
        if tc == country:
            ctry_tgt_ids.append(tid)
            ctry_tgt_names.append(tn)
            ctry_tgt_addrs.append(ta)
    
    N_ctry = len(ctry_tgt_ids)
    
    raw_inv = defaultdict(list)
    for i in range(N_ctry):
        for tok in set(clean_toks(ctry_tgt_names[i], ctry_tgt_addrs[i])):
            raw_inv[tok].append(i)
    inv_index = {tok: v for tok, v in raw_inv.items() if len(v) <= MAX_POSTING}
    idf = {tok: math.log(N_ctry / (len(v) + 1)) for tok, v in inv_index.items()}
    del raw_inv
    
    log(f"  {country}: scoring {len(val_by_country[country]):,} val entities against {N_ctry:,} targets")
    
    for sid in val_by_country[country]:
        s1_n, s1_a, s1_c = s1_recs[sid]
        s1_pre = precompute_s1(s1_n, s1_a)
        
        scores = Counter()
        for tok in clean_toks(s1_n, s1_a):
            if tok in inv_index:
                w = idf[tok]
                for idx in inv_index[tok]:
                    scores[idx] += w
        
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:TOP_K]
        
        for idx, idf_w in top:
            tid = ctry_tgt_ids[idx]
            feat = extract_features(s1_pre, ctry_tgt_names[idx], ctry_tgt_addrs[idx], idf_w)
            val_feats.append(feat)
            val_meta.append((sid, tid))
    
    log(f"  {country}: done in {time.time()-t0:.1f}s")
    del ctry_tgt_ids, ctry_tgt_names, ctry_tgt_addrs, inv_index, idf
    gc.collect()

val_feats = np.array(val_feats, dtype=np.float32)
probs = clf.predict_proba(val_feats)[:, 1]
log(f"  Total val pairs: {len(val_feats):,}")

gt_val = {sid: gt_map[sid] for sid in val_ids}

# Sweep threshold
log("\nThreshold sweep (0.30 → 0.90):")
best_tau, best_f05 = 0.0, 0.0
for tau in np.arange(0.30, 0.92, 0.02):
    tau = round(tau, 2)
    scored = [(float(p), sid, tid) for (sid, tid), p in zip(val_meta, probs) if p >= tau]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    assigned = set()
    preds = defaultdict(list)
    for p, sid, tid in scored:
        if tid not in assigned:
            preds[sid].append(tid)
            assigned.add(tid)
    f05 = macro_f05(gt_val, preds)
    marker = "  ◄ BEST" if f05 > best_f05 else ""
    if f05 > best_f05:
        best_f05 = f05
        best_tau = tau
    log(f"  tau={tau:.2f}: F0.5={f05:.4f}{marker}")

log(f"\n>>> Best tau={best_tau:.2f}, F0.5={best_f05:.4f} <<<")

# Save model
with open(os.path.join(TEMP_DIR, "clf_v7.pkl"), "wb") as f:
    pickle.dump((clf, best_tau), f)
log(f"Model saved to {TEMP_DIR}/clf_v7.pkl")

# =====================================================
# PHASE 6: FULL TEST INFERENCE
# =====================================================
log("\n" + "=" * 60)
log("PHASE 6: FULL TEST INFERENCE")
log("=" * 60)

threshold = best_tau

for country in ["France", "US", "India"]:
    t_start = time.time()
    log(f"\n--- INFERENCE: {country} ---")
    
    # Load S1 for this country
    s1_country = []
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4 and p[3] == country:
                s1_country.append((p[0], p[1], p[2]))
    log(f"  S1 entities: {len(s1_country):,}")
    if not s1_country: continue
    
    # Load targets for this country
    tgt_ids_c, tgt_names_c, tgt_addrs_c = [], [], []
    for sf in ["test_source2.tsv", "test_source3.tsv"]:
        with open(os.path.join(TEST_DIR, sf), encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4 and p[3] == country:
                    tgt_ids_c.append(p[0])
                    tgt_names_c.append(p[1])
                    tgt_addrs_c.append(p[2])
    N_tgt = len(tgt_ids_c)
    log(f"  Targets: {N_tgt:,}")
    
    # Build inverted index
    t0 = time.time()
    raw_inv = defaultdict(list)
    for i in range(N_tgt):
        for tok in set(clean_toks(tgt_names_c[i], tgt_addrs_c[i])):
            raw_inv[tok].append(i)
    inv_index = {tok: v for tok, v in raw_inv.items() if len(v) <= MAX_POSTING}
    idf = {tok: math.log(N_tgt / (len(v) + 1)) for tok, v in inv_index.items()}
    del raw_inv
    log(f"  Index: {len(inv_index):,} tokens in {time.time()-t0:.1f}s")
    
    # Stream batch scoring with adaptive candidates
    batch_size = 50000
    all_scored_pairs = []
    cand_temp_path = os.path.join(TEMP_DIR, f"cands_{country}.tsv")
    f_cand = open(cand_temp_path, "w", encoding="utf-8")
    
    for b_start in range(0, len(s1_country), batch_size):
        b_end = min(b_start + batch_size, len(s1_country))
        tb0 = time.time()
        b_feats, b_meta = [], []
        
        for sid, s1_n, s1_a in s1_country[b_start:b_end]:
            scores = Counter()
            for tok in clean_toks(s1_n, s1_a):
                if tok in inv_index:
                    w = idf[tok]
                    for idx in inv_index[tok]:
                        scores[idx] += w
            
            top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:TOP_K]
            
            # Adaptive pruning
            if top:
                top_score = top[0][1]
                cutoff = ALPHA_ADAPTIVE * top_score
                top = [(idx, w) for idx, w in top if w >= cutoff]
            
            cand_ids = []
            s1_pre = precompute_s1(s1_n, s1_a)
            for idx, idf_w in top:
                tid = tgt_ids_c[idx]
                cand_ids.append(tid)
                feat = extract_features(s1_pre, tgt_names_c[idx], tgt_addrs_c[idx], idf_w)
                b_feats.append(feat)
                b_meta.append((sid, tid))
            
            f_cand.write(f"{sid}\t{','.join(cand_ids)}\n")
        
        if b_feats:
            b_feats_arr = np.array(b_feats, dtype=np.float32)
            b_probs = clf.predict_proba(b_feats_arr)[:, 1]
            for (sid, tid), p in zip(b_meta, b_probs):
                if p >= threshold:
                    all_scored_pairs.append((float(p), sid, tid))
        
        log(f"    Batch {b_end:,}/{len(s1_country):,} in {time.time()-tb0:.1f}s | {len(all_scored_pairs):,} matches")
    
    f_cand.close()
    
    # Greedy 1-to-1 assignment
    all_scored_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    assigned_targets = set()
    s1_matches = defaultdict(list)
    for p, sid, tid in all_scored_pairs:
        if tid not in assigned_targets:
            s1_matches[sid].append(tid)
            assigned_targets.add(tid)
    
    match_temp_path = os.path.join(TEMP_DIR, f"matches_{country}.tsv")
    with open(match_temp_path, "w", encoding="utf-8") as f_match:
        for sid, _, _ in s1_country:
            m = s1_matches.get(sid, [])
            f_match.write(f"{sid}\t{','.join(m)}\n")
    
    log(f"  {country}: {len(s1_matches):,}/{len(s1_country):,} matched ({len(assigned_targets):,} targets) in {time.time()-t_start:.1f}s")
    
    del s1_country, tgt_ids_c, tgt_names_c, tgt_addrs_c, inv_index, idf
    del all_scored_pairs, s1_matches, assigned_targets
    gc.collect()

# =====================================================
# PHASE 7: ASSEMBLE FINAL OUTPUT
# =====================================================
log("\n" + "=" * 60)
log("ASSEMBLING FINAL OUTPUT")
log("=" * 60)

canonical_s1 = []
with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        canonical_s1.append(line.split('\t')[0])
log(f"Canonical test entities: {len(canonical_s1):,}")

# Merge matches
matches_map = {}
for country in ["France", "US", "India"]:
    p = os.path.join(TEMP_DIR, f"matches_{country}.tsv")
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                matches_map[parts[0]] = parts[1] if len(parts) > 1 else ''

# Merge candidates
cands_map = {}
for country in ["France", "US", "India"]:
    p = os.path.join(TEMP_DIR, f"cands_{country}.tsv")
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                cands_map[parts[0]] = parts[1] if len(parts) > 1 else ''

# Write final files
match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")
cand_out  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

with open(match_out, "w", encoding="utf-8") as fm, \
     open(cand_out, "w", encoding="utf-8") as fc:
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    
    matched_count = 0
    singleton_count = 0
    for sid in canonical_s1:
        m = matches_map.get(sid, '')
        c = cands_map.get(sid, '')
        fm.write(f"{sid}\t{m}\n")
        fc.write(f"{sid}\t{c}\n")
        if m.strip():
            matched_count += 1
        else:
            singleton_count += 1

log(f"Output: {matched_count:,} matched, {singleton_count:,} singletons ({singleton_count/(matched_count+singleton_count)*100:.1f}%)")
log(f"Files: {match_out}, {cand_out}")

# Validate
log("\nRunning validator...")
import subprocess
r = subprocess.run(
    ["python", os.path.join(BASE, "utils", "validate_submission.py"),
     "--matching", match_out, "--candidate", cand_out, "--test-dir", TEST_DIR],
    capture_output=True, text=True, encoding='utf-8'
)
log(r.stdout)
if r.returncode != 0:
    log(f"VALIDATOR STDERR: {r.stderr}")

log("\nDONE.")
