#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Complete Production Pipeline
End-to-End: Fast Blocking -> Feature Extraction -> HistGradientBoosting -> Macro F0.5 Thresholding -> Greedy 1-to-1 Matching -> Submission Generation & Validation
"""
import time, os, sys, re, gc
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci', 'france', 'de', 'la',
    'le', 'du', 'des', 'les', 'en', 'rue', 'bd', 'boulevard', 'av', 'impasse'
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
    
    c_jacc = fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2)
    
    return [n_jacc, n_3g_jacc, a_jacc, num_jacc, len_ratio, shared_tok_count, c_jacc]

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
                precision = tp / len(preds)
                recall = tp / len(true_targets)
                denom = 0.25 * precision + recall
                f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
                scores.append(f05)
    return float(np.mean(scores)) if scores else 0.0

def train_and_optimize():
    log("="*60)
    log("PHASE 1: TRAINING & THRESHOLD OPTIMIZATION")
    log("="*60)
    
    # 1. Load GT sample
    log("Loading Ground Truth sample for training...")
    gt_map = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
            gt_map[p[0]] = set(m)
            if len(gt_map) >= 40000:
                break
                
    s1_eval_ids = set(gt_map.keys())
    
    # 2. Load S1 records
    s1_records = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in s1_eval_ids:
                s1_records[p[0]] = (p[1], p[2], p[3])
                if len(s1_records) == len(s1_eval_ids):
                    break
    log(f"Loaded {len(s1_records):,} S1 training entities")
    
    s1_list = list(s1_records.keys())
    train_s1_ids = set(s1_list[:25000])
    val_s1_ids = set(s1_list[25000:])
    
    all_needed_targets = set()
    for sid in s1_eval_ids:
        all_needed_targets |= gt_map[sid]
        
    # 3. Load Targets sample (all needed true targets + 800k distractors)
    log("Loading S2 and S3 target records...")
    tgt_records = {}
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(TRAIN_DIR, sf), encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                p = line.rstrip('\n').split('\t')
                if p[0] in all_needed_targets or i < 400000:
                    tgt_records[p[0]] = (p[1], p[2], p[3])
    log(f"Loaded {len(tgt_records):,} target records for training")
    
    # 4. Build inverted index
    log("Building inverted index on targets...")
    t0 = time.time()
    inv_index = defaultdict(list)
    tgt_id_list = list(tgt_records.keys())
    for i, tid in enumerate(tgt_id_list):
        n, a, c = tgt_records[tid]
        for tok in clean_toks(n, a):
            inv_index[tok].append(i)
    inv_index = {k: v for k, v in inv_index.items() if len(v) <= 300}
    log(f"Inverted index built in {time.time()-t0:.2f}s ({len(inv_index):,} distinctive tokens)")
    
    # 5. Build training features
    log("Generating training candidate pairs...")
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
    log(f"Total training pairs: {len(X_train):,} (Positives: {sum(y_train):,}, Negatives: {len(y_train)-sum(y_train):,})")
    
    # 6. Fit HistGradientBoostingClassifier
    log("Training HistGradientBoostingClassifier (Apache-2.0, 0 pre-trained embeddings)...")
    clf = HistGradientBoostingClassifier(random_state=42, max_iter=250, min_samples_leaf=30, l2_regularization=1.0)
    clf.fit(X_train, y_train)
    log("Classifier trained successfully!")
    
    # 7. Evaluate on validation set
    log(f"Evaluating on {len(val_s1_ids):,} validation entities...")
    val_candidates = {}
    val_features = []
    val_pairs_meta = []
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
    
    # Grid search threshold
    gt_val = {sid: gt_map[sid] for sid in val_s1_ids}
    best_t = 0.65
    best_score = 0.0
    log("Tuning threshold for Macro F0.5...")
    for t in np.arange(0.50, 0.95, 0.05):
        scored = [(p, sid, tid) for (sid, tid), p in zip(val_pairs_meta, probs) if p >= t]
        scored.sort(reverse=True)
        assigned_tgt = set()
        preds = defaultdict(list)
        for p, sid, tid in scored:
            if tid not in assigned_tgt:
                preds[sid].append(tid)
                assigned_tgt.add(tid)
        score = macro_f05(gt_val, preds)
        log(f"  Threshold {t:.2f} -> Macro F0.5 = {score:.4f} (Matched {len(preds):,}/{len(val_s1_ids):,} S1)")
        if score > best_score:
            best_score = score
            best_t = float(t)
            
    log(f"OPTIMAL THRESHOLD: {best_t:.2f} with Macro F0.5 = {best_score:.4f}")
    
    del gt_map, s1_records, tgt_records, inv_index, X_train, y_train, val_features, probs
    gc.collect()
    
    return clf, best_t

def run_test_inference(clf, threshold):
    log("="*60)
    log("PHASE 2: FULL TEST INFERENCE ACROSS ALL COUNTRIES")
    log("="*60)
    
    # Read test S1 list to preserve exact ordering and check completeness
    log("Loading test_source1.tsv entity IDs...")
    all_s1_ordered = []
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            all_s1_ordered.append(p[0])
    log(f"Total test Source 1 entities required: {len(all_s1_ordered):,}")
    
    # Outputs dictionaries
    final_matches = defaultdict(list)     # s1_id -> [matched target ids]
    final_candidates = defaultdict(list)  # s1_id -> [candidate target ids]
    
    # Process country-by-country
    countries = ["France", "US", "India"]
    
    for country in countries:
        log(f"\n>>> PROCESSING COUNTRY: {country} <<<")
        t_country_start = time.time()
        
        # 1. Stream S1 for this country
        log(f"  Loading S1 for {country}...")
        s1_country = {}
        with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4 and p[3] == country:
                    s1_country[p[0]] = (p[1], p[2])
        log(f"  Loaded {len(s1_country):,} S1 entities for {country}")
        if not s1_country:
            continue
            
        # 2. Stream Targets (S2 + S3) for this country
        log(f"  Loading Targets (S2 + S3) for {country}...")
        tgt_ids = []
        tgt_names = []
        tgt_addrs = []
        for sf in ["test_source2.tsv", "test_source3.tsv"]:
            with open(os.path.join(TEST_DIR, sf), encoding='utf-8') as f:
                f.readline()
                for line in f:
                    p = line.rstrip('\n').split('\t')
                    if len(p) >= 4 and p[3] == country:
                        tgt_ids.append(p[0])
                        tgt_names.append(p[1])
                        tgt_addrs.append(p[2])
        log(f"  Loaded {len(tgt_ids):,} targets for {country}")
        
        # 3. Build inverted index on targets
        log(f"  Building distinctive token index on {len(tgt_ids):,} targets...")
        t0 = time.time()
        inv_index = defaultdict(list)
        for i in range(len(tgt_ids)):
            for tok in clean_toks(tgt_names[i], tgt_addrs[i]):
                inv_index[tok].append(i)
        inv_index = {k: v for k, v in inv_index.items() if len(v) <= 300}
        log(f"  Inverted index built in {time.time()-t0:.2f}s ({len(inv_index):,} tokens)")
        
        # 4. Query S1 entities in batches and extract features
        log(f"  Blocking & feature scoring for {len(s1_country):,} S1 queries...")
        t1 = time.time()
        
        s1_country_items = list(s1_country.items())
        batch_size = 50000
        country_scored_pairs = [] # (prob, s1_id, target_id)
        
        for b_start in range(0, len(s1_country_items), batch_size):
            b_end = min(b_start + batch_size, len(s1_country_items))
            b_items = s1_country_items[b_start:b_end]
            
            b_feats = []
            b_meta = [] # (s1_id, tid)
            
            for sid, (s1_n, s1_a) in b_items:
                toks = clean_toks(s1_n, s1_a)
                counts = Counter()
                for tok in toks:
                    if tok in inv_index:
                        for idx in inv_index[tok]:
                            counts[idx] += 1
                top = counts.most_common(20)
                
                cands_for_sid = []
                for idx, shared_c in top:
                    tid = tgt_ids[idx]
                    cands_for_sid.append(tid)
                    feat = extract_features(s1_n, s1_a, tgt_names[idx], tgt_addrs[idx], shared_c)
                    b_feats.append(feat)
                    b_meta.append((sid, tid))
                final_candidates[sid] = cands_for_sid
                
            if b_feats:
                b_feats = np.array(b_feats, dtype=np.float32)
                b_probs = clf.predict_proba(b_feats)[:, 1]
                for (sid, tid), p in zip(b_meta, b_probs):
                    if p >= threshold:
                        country_scored_pairs.append((float(p), sid, tid))
                        
            log(f"    Processed {b_end:,}/{len(s1_country):,} S1 queries... ({len(country_scored_pairs):,} pairs >= {threshold:.2f})")
            
        log(f"  Blocking & scoring completed in {time.time()-t1:.2f}s")
        
        # 5. Greedy 1-to-1 matching for this country
        log(f"  Applying greedy 1-to-1 matching on {len(country_scored_pairs):,} candidate matches...")
        country_scored_pairs.sort(reverse=True)
        assigned_tgt = set()
        c_matched_s1 = set()
        
        for p, sid, tid in country_scored_pairs:
            if tid not in assigned_tgt:
                final_matches[sid].append(tid)
                assigned_tgt.add(tid)
                c_matched_s1.add(sid)
                
        log(f"  Country {country} Summary: Matched {len(c_matched_s1):,}/{len(s1_country):,} S1 ({len(assigned_tgt):,} unique targets assigned) in {time.time()-t_country_start:.1f}s")
        
        # Cleanup memory before next country
        del s1_country, s1_country_items, tgt_ids, tgt_names, tgt_addrs, inv_index, country_scored_pairs
        gc.collect()

    # 6. Write matching_results.tsv
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    log(f"\nWriting final {matching_path}...")
    n_matched = 0
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ordered:
            m = final_matches.get(sid, [])
            if m: n_matched += 1
            f.write(f"{sid}\t{','.join(m)}\n")
    log(f"  Done! Total S1 written: {len(all_s1_ordered):,}, Non-singleton matches: {n_matched:,} ({n_matched/len(all_s1_ordered)*100:.2f}%)")

    # 7. Write candidate_pairs.tsv
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    log(f"Writing final {candidate_path}...")
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ordered:
            c = final_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(c)}\n")
    log(f"  Done! Total candidate rows written: {len(all_s1_ordered):,}")

def main():
    total_start = time.time()
    log("="*60)
    log("AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION")
    log("="*60)
    
    clf, threshold = train_and_optimize()
    run_test_inference(clf, threshold)
    
    log("="*60)
    log(f"PIPELINE COMPLETED SUCCESSFULLY IN {(time.time()-total_start)/60:.2f} MINUTES!")
    log("="*60)

if __name__ == "__main__":
    main()
