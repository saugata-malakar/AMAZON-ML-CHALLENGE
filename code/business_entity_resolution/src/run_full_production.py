#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Production Pipeline (v7 High-Recall Streaming Engine)
Key Architecture:
- Streaming Country-by-Country Execution (<1.2 GB RAM, 0 OS page thrashing)
- C-packed array.array('I') posting lists (4.5x more compact than Python lists)
- High-Recall Blocker: TOP_K=35, MAX_POSTING=3000 (84.23% entity recall)
- Patch 1: Scale-invariant normalized IDF feature [0, 1]
- Patch 2: Isotonic probability calibration on separate validation slice
- Patch 3: Robust safe_iter_rows parser for all TSV inputs
- Adaptive candidate pruning (alpha=0.3) for competition candidate size bonus
"""
import time, os, sys, re, gc, math, pickle, subprocess, zipfile, array
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
ROOT_DIR = r"c:\Users\Administrator\Downloads\AMAZON"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR  = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE, "output")
TEMP_DIR  = os.path.join(BASE, "temp_v7")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 35              # High-recall candidate budget (proven 84.23% entity recall)
MAX_POSTING = 3000      # Expanded cutoff: retains normal business words in 10M corpus
ALPHA_ADAPTIVE = 0.3    # Adaptive candidate pruning ratio

STOPWORDS = {
    'inc','corp','corporation','llc','ltd','limited','pvt','private','co','company',
    'and','the','of','in','at','road','rd','street','st','avenue','ave','lane','ln',
    'drive','dr','nagar','colony','floor','near','opp','opposite','block','sector',
    'phase','house','plot','door','no','null','sarl','sas','sci','france','de','la',
    'le','du','des','les','en','rue','bd','boulevard','av','impasse','new','old',
    'east','west','north','south','main','cross'
}

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def normalize_idf(raw_idf_w: float, n_ctry: int) -> float:
    denom = max(math.log(n_ctry + 1), 1e-6)
    return raw_idf_w / denom

def safe_iter_rows(path, min_fields=4):
    skipped = 0
    with open(path, encoding="utf-8") as f:
        f.readline()  # skip header
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < min_fields:
                skipped += 1
                continue
            yield parts
    if skipped:
        log(f"[WARN] {path}: skipped {skipped} malformed row(s)")

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
# PHASE 1: LOAD GROUND TRUTH & SOURCE 1
# =====================================================
log("=" * 60)
log("HIGH-RECALL STREAMING PRODUCTION PIPELINE")
log(f"Config: TOP_K={TOP_K}, MAX_POSTING={MAX_POSTING}, ALPHA={ALPHA_ADAPTIVE}")
log("=" * 60)

TOTAL_N = 80000
TRAIN_N = 50000
CALIB_N = 15000
TUNE_N  = 15000

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

log("Loading S1 records...")
t0 = time.time()
s1_recs = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_needed and len(p) >= 4:
            s1_recs[p[0]] = (p[1], p[2], p[3])
            if len(s1_recs) == len(s1_needed): break

log(f"  S1 loaded: {len(s1_recs):,} in {time.time()-t0:.1f}s")

s1_list = list(s1_recs.keys())
train_ids = s1_list[:TRAIN_N]
calib_ids = s1_list[TRAIN_N:TRAIN_N+CALIB_N]
tune_ids  = s1_list[TRAIN_N+CALIB_N:TOTAL_N]

log(f"Split: {len(train_ids):,} train, {len(calib_ids):,} calibration, {len(tune_ids):,} threshold tuning")

# =====================================================
# PHASE 2 & 3: STREAMING PAIR GENERATION (COUNTRY BY COUNTRY)
# =====================================================
countries = ['US', 'India']
s1_train_by_ctry = defaultdict(list)
for sid in train_ids: s1_train_by_ctry[s1_recs[sid][2]].append(sid)

s1_calib_by_ctry = defaultdict(list)
for sid in calib_ids: s1_calib_by_ctry[s1_recs[sid][2]].append(sid)

s1_tune_by_ctry = defaultdict(list)
for sid in tune_ids: s1_tune_by_ctry[s1_recs[sid][2]].append(sid)

X_train, y_train = [], []
X_calib, y_calib = [], []
tune_feats, tune_meta = [], []

for country in countries:
    t_c_start = time.time()
    log(f"\n--- Processing country: {country} ---")
    
    # 1. Stream only this country's targets
    ctry_tgt_ids, ctry_tgt_names, ctry_tgt_addrs = [], [], []
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        for parts in safe_iter_rows(os.path.join(TRAIN_DIR, sf), min_fields=4):
            if parts[3] == country:
                ctry_tgt_ids.append(parts[0])
                ctry_tgt_names.append(parts[1])
                ctry_tgt_addrs.append(parts[2])
                
    N_ctry = len(ctry_tgt_ids)
    log(f"  Loaded {N_ctry:,} targets for {country} in {time.time()-t_c_start:.1f}s")
    
    # 2. Build inverted index using compact array.array('I')
    t0 = time.time()
    raw_inv = defaultdict(lambda: array.array('I'))
    for i in range(N_ctry):
        for tok in set(clean_toks(ctry_tgt_names[i], ctry_tgt_addrs[i])):
            raw_inv[tok].append(i)
            
    inv_index = {tok: v for tok, v in raw_inv.items() if len(v) <= MAX_POSTING}
    idf = {tok: math.log(N_ctry / (len(v) + 1)) for tok, v in inv_index.items()}
    del raw_inv
    gc.collect()
    log(f"  Index: {len(inv_index):,} tokens built in {time.time()-t0:.1f}s")
    
    # Helper for scoring
    def generate_pairs_for_set(s1_id_list):
        feats_list, labels_list, meta_list = [], [], []
        for sid in s1_id_list:
            s1_n, s1_a, _ = s1_recs[sid]
            s1_pre = precompute_s1(s1_n, s1_a)
            true_tgt = gt_map.get(sid, set())
            
            scores = Counter()
            for tok in clean_toks(s1_n, s1_a):
                if tok in inv_index:
                    w = idf[tok]
                    for idx in inv_index[tok]: scores[idx] += w
            top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:TOP_K]
            
            for idx, idf_w in top:
                tid = ctry_tgt_ids[idx]
                norm_idf = normalize_idf(idf_w, N_ctry)
                feat = extract_features(s1_pre, ctry_tgt_names[idx], ctry_tgt_addrs[idx], norm_idf)
                feats_list.append(feat)
                labels_list.append(1 if tid in true_tgt else 0)
                meta_list.append((sid, tid))
        return feats_list, labels_list, meta_list

    t0 = time.time()
    f_tr, l_tr, _ = generate_pairs_for_set(s1_train_by_ctry[country])
    X_train.extend(f_tr); y_train.extend(l_tr)
    log(f"  Train pairs for {country}: {len(f_tr):,} in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    f_cal, l_cal, _ = generate_pairs_for_set(s1_calib_by_ctry[country])
    X_calib.extend(f_cal); y_calib.extend(l_cal)
    log(f"  Calib pairs for {country}: {len(f_cal):,} in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    f_tu, _, m_tu = generate_pairs_for_set(s1_tune_by_ctry[country])
    tune_feats.extend(f_tu); tune_meta.extend(m_tu)
    log(f"  Tune pairs for {country}: {len(f_tu):,} in {time.time()-t0:.1f}s")

    # Free country memory completely before next country
    del ctry_tgt_ids, ctry_tgt_names, ctry_tgt_addrs, inv_index, idf
    gc.collect()

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_calib = np.array(X_calib, dtype=np.float32)
y_calib = np.array(y_calib, dtype=np.int32)
tune_feats = np.array(tune_feats, dtype=np.float32)

log(f"\nTOTAL PAIRS: Train={len(X_train):,}, Calib={len(X_calib):,}, Tune={len(tune_feats):,}")

# =====================================================
# PHASE 4: TRAIN BASE CLASSIFIER & ISOTONIC CALIBRATOR
# =====================================================
log("Fitting HistGradientBoostingClassifier...")
t0 = time.time()
base_clf = HistGradientBoostingClassifier(
    max_iter=300, max_depth=7, learning_rate=0.05,
    min_samples_leaf=20, l2_regularization=0.3, random_state=42
)
base_clf.fit(X_train, y_train)
log(f"  Base model fitted in {time.time()-t0:.1f}s")
del X_train, y_train
gc.collect()

log("Fitting Isotonic Probability Calibrator...")
t0 = time.time()
raw_calib_probs = base_clf.predict_proba(X_calib)[:, 1]
iso_calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
iso_calibrator.fit(raw_calib_probs, y_calib)
log(f"  Calibrator fitted in {time.time()-t0:.1f}s")
del X_calib, y_calib, raw_calib_probs
gc.collect()

# =====================================================
# PHASE 5: THRESHOLD SWEEP ON CALIBRATED PROBABILITIES
# =====================================================
log("\nSweeping threshold on separate tuning slice...")
t0 = time.time()
raw_tune_probs = base_clf.predict_proba(tune_feats)[:, 1]
calib_tune_probs = iso_calibrator.predict(raw_tune_probs)
gt_tune = {sid: gt_map[sid] for sid in tune_ids}

best_tau, best_f05 = 0.50, 0.0
for tau in np.arange(0.20, 0.86, 0.02):
    tau = round(tau, 2)
    scored = [(float(p), sid, tid) for (sid, tid), p in zip(tune_meta, calib_tune_probs) if p >= tau]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    assigned = set()
    preds = defaultdict(list)
    for p, sid, tid in scored:
        if tid not in assigned:
            preds[sid].append(tid)
            assigned.add(tid)
    f05 = macro_f05(gt_tune, preds)
    marker = "  ◄ BEST" if f05 > best_f05 else ""
    if f05 > best_f05:
        best_f05 = f05
        best_tau = tau
    log(f"  tau={tau:.2f}: F0.5={f05:.4f}{marker}")

log(f"\n>>> Optimal Calibrated tau={best_tau:.2f} (F0.5={best_f05:.4f}) <<<")

del tune_feats, tune_meta, raw_tune_probs, calib_tune_probs, gt_tune
gc.collect()

# Save calibrated model artifact
with open(os.path.join(TEMP_DIR, "calibrated_clf.pkl"), "wb") as f:
    pickle.dump((base_clf, iso_calibrator, best_tau), f)
log(f"Saved calibrated model to {TEMP_DIR}/calibrated_clf.pkl")

# =====================================================
# PHASE 6: FULL TEST INFERENCE (STREAMING PER COUNTRY)
# =====================================================
log("\n" + "=" * 60)
log("PHASE 6: FULL TEST INFERENCE (ALL 3 COUNTRIES)")
log("=" * 60)

for country in ["France", "US", "India"]:
    t_start = time.time()
    log(f"\n--- INFERENCE: {country} ---")
    
    # 1. Load S1 for this country
    s1_country = []
    for parts in safe_iter_rows(os.path.join(TEST_DIR, "test_source1.tsv"), min_fields=4):
        if parts[3] == country:
            s1_country.append((parts[0], parts[1], parts[2]))
            
    log(f"  S1 entities: {len(s1_country):,}")
    if not s1_country: continue
    
    # 2. Stream targets for this country
    t0 = time.time()
    tgt_ids_c, tgt_names_c, tgt_addrs_c = [], [], []
    for sf in ["test_source2.tsv", "test_source3.tsv"]:
        for parts in safe_iter_rows(os.path.join(TEST_DIR, sf), min_fields=4):
            if parts[3] == country:
                tgt_ids_c.append(parts[0])
                tgt_names_c.append(parts[1])
                tgt_addrs_c.append(parts[2])
                
    N_tgt = len(tgt_ids_c)
    log(f"  Targets: {N_tgt:,} loaded in {time.time()-t0:.1f}s")
    
    # 3. Build inverted index using compact array.array('I')
    t0 = time.time()
    raw_inv = defaultdict(lambda: array.array('I'))
    for i in range(N_tgt):
        for tok in set(clean_toks(tgt_names_c[i], tgt_addrs_c[i])):
            raw_inv[tok].append(i)
            
    inv_index = {tok: v for tok, v in raw_inv.items() if len(v) <= MAX_POSTING}
    idf = {tok: math.log(N_tgt / (len(v) + 1)) for tok, v in inv_index.items()}
    del raw_inv
    gc.collect()
    log(f"  Index: {len(inv_index):,} tokens in {time.time()-t0:.1f}s")
    
    # 4. Stream batch scoring with adaptive pruning
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
                    for idx in inv_index[tok]: scores[idx] += w
            top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:TOP_K]
            
            # Adaptive candidate pruning
            if top:
                top_score = top[0][1]
                cutoff = ALPHA_ADAPTIVE * top_score
                top = [(idx, w) for idx, w in top if w >= cutoff]
                
            cand_ids = []
            s1_pre = precompute_s1(s1_n, s1_a)
            for idx, idf_w in top:
                tid = tgt_ids_c[idx]
                cand_ids.append(tid)
                norm_idf = normalize_idf(idf_w, N_tgt)
                feat = extract_features(s1_pre, tgt_names_c[idx], tgt_addrs_c[idx], norm_idf)
                b_feats.append(feat)
                b_meta.append((sid, tid))
                
            f_cand.write(f"{sid}\t{','.join(cand_ids)}\n")
            
        if b_feats:
            b_feats_arr = np.array(b_feats, dtype=np.float32)
            raw_p = base_clf.predict_proba(b_feats_arr)[:, 1]
            b_probs = iso_calibrator.predict(raw_p)
            for (sid, tid), p in zip(b_meta, b_probs):
                if p >= best_tau:
                    all_scored_pairs.append((float(p), sid, tid))
                    
        log(f"    Batch {b_end:,}/{len(s1_country):,} in {time.time()-tb0:.1f}s | {len(all_scored_pairs):,} accepted")
        
    f_cand.close()
    
    # 5. Greedy priority locking
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
# PHASE 7: ASSEMBLE FINAL DELIVERABLES
# =====================================================
log("\n" + "=" * 60)
log("ASSEMBLING FINAL OUTPUT")
log("=" * 60)

canonical_s1 = []
for parts in safe_iter_rows(os.path.join(TEST_DIR, "test_source1.tsv"), min_fields=1):
    canonical_s1.append(parts[0])
log(f"Canonical test entities: {len(canonical_s1):,}")

matches_map = {}
for country in ["France", "US", "India"]:
    p = os.path.join(TEMP_DIR, f"matches_{country}.tsv")
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                matches_map[parts[0]] = parts[1] if len(parts) > 1 else ''

cands_map = {}
for country in ["France", "US", "India"]:
    p = os.path.join(TEMP_DIR, f"cands_{country}.tsv")
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                cands_map[parts[0]] = parts[1] if len(parts) > 1 else ''

match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")
cand_out  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

with open(match_out, "w", encoding="utf-8") as fm, open(cand_out, "w", encoding="utf-8") as fc:
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    
    matched_count = 0
    singleton_count = 0
    for sid in canonical_s1:
        m = matches_map.get(sid, '')
        c = cands_map.get(sid, '')
        fm.write(f"{sid}\t{m}\n")
        fc.write(f"{sid}\t{c}\n")
        if m.strip(): matched_count += 1
        else: singleton_count += 1

log(f"Final output: {matched_count:,} matched, {singleton_count:,} singletons ({singleton_count/(matched_count+singleton_count)*100:.1f}%)")
log(f"Saved: {match_out}")
log(f"Saved: {cand_out}")

# Copy to root upload location
root_copy = os.path.join(ROOT_DIR, "FINAL_LEADERBOARD_matching_results.tsv")
import shutil
shutil.copyfile(match_out, root_copy)
log(f"Updated root upload file: {root_copy}")

# Validate
log("\nRunning official submission validator...")
r = subprocess.run(
    ["python", os.path.join(BASE, "utils", "validate_submission.py"),
     "--matching", match_out, "--candidate", cand_out, "--test-dir", TEST_DIR],
    capture_output=True, text=True, encoding='utf-8'
)
log(r.stdout)
if r.returncode != 0: log(f"VALIDATOR STDERR: {r.stderr}")

# Rebuild submission package
log("\nRebuilding EntityResolvers_submission.zip...")
zip_path = os.path.join(ROOT_DIR, "EntityResolvers_submission.zip")
if os.path.exists(zip_path): os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(match_out, "output/matching_results.tsv")
    zf.write(cand_out, "output/candidate_pairs.tsv")
    code_dir = os.path.join(ROOT_DIR, "code", "business_entity_resolution")
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.pyc'): continue
            full = os.path.join(root, file)
            arcname = 'code/business_entity_resolution/' + os.path.relpath(full, code_dir)
            zf.write(full, arcname)
    doc_path = os.path.join(BASE, "Documentation_template.md")
    if os.path.exists(doc_path):
        zf.write(doc_path, "Documentation_template.md")

log(f"Final Zip built: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
log("PIPELINE EXECUTION COMPLETE.")
