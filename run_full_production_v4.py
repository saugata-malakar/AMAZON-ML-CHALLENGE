#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Entity Resolution Pipeline v4
FIX: Cache tokenization during index build (v3 was tokenizing 3.8M records TWICE)
Other: Same IDF-weighted blocking + 13 features as v3
"""
import time, os, sys, re, gc, math, difflib
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR  = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 15
TRAIN_SAMPLE = 50_000

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci', 'france', 'de', 'la',
    'le', 'du', 'des', 'les', 'en', 'rue', 'bd', 'boulevard', 'av', 'impasse',
    'new', 'old', 'east', 'west', 'north', 'south', 'main', 'cross', 'near'
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def clean_toks(name, addr):
    text = (name + " " + addr).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) >= 3 and w not in STOPWORDS]

def get_words(text):
    return set(re.findall(r'\w+', text.lower()))

def get_3grams(text):
    t = text.lower()
    if len(t) < 3: return set()
    return set(t[i:i+3] for i in range(len(t)-2))

def fast_jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def seq_ratio(a, b):
    if not a or not b: return 0.0
    if a == b: return 1.0
    return difflib.SequenceMatcher(None, a[:200], b[:200]).ratio()

def extract_features(s1_n, s1_a, t_n, t_a, idf_score):
    """13 features per pair."""
    s1_nc = re.sub(r'[^\w\s]', ' ', s1_n.lower()).strip()
    t_nc  = re.sub(r'[^\w\s]', ' ', t_n.lower()).strip()
    s1_ac = re.sub(r'[^\w\s]', ' ', s1_a.lower()).strip()
    t_ac  = re.sub(r'[^\w\s]', ' ', t_a.lower()).strip()

    n_tok1, n_tok2 = get_words(s1_n), get_words(t_n)
    n_3g1,  n_3g2  = get_3grams(s1_n), get_3grams(t_n)
    a_tok1, a_tok2 = get_words(s1_a), get_words(t_a)
    num1 = set(re.findall(r'\d+', s1_a))
    num2 = set(re.findall(r'\d+', t_a))

    return [
        fast_jaccard(n_tok1, n_tok2),
        fast_jaccard(n_3g1, n_3g2),
        seq_ratio(s1_nc, t_nc),
        (min(len(s1_nc), len(t_nc)) + 1) / (max(len(s1_nc), len(t_nc)) + 1),
        1.0 if s1_nc.split()[:1] == t_nc.split()[:1] and s1_nc.split() else 0.0,
        len(n_tok1 & n_tok2) / (max(len(n_tok1), 1)),
        fast_jaccard(a_tok1, a_tok2),
        fast_jaccard(get_3grams(s1_a), get_3grams(t_a)),
        seq_ratio(s1_ac, t_ac),
        fast_jaccard(num1, num2),
        1.0 if (not s1_ac or not t_ac) else 0.0,
        fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2),
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

# ── IDF-Weighted Inverted Index (FAST — single-pass tokenization) ────
class IDFBlocker:
    def __init__(self, max_posting_size=500):
        self.inv_index = defaultdict(list)
        self.idf = {}
        self.max_posting_size = max_posting_size

    def fit(self, tgt_ids, tgt_names, tgt_addrs):
        N = len(tgt_ids)
        log(f"  Building IDF-weighted inverted index on {N:,} targets...")
        t0 = time.time()

        # SINGLE-PASS: tokenize once, count doc freq, and store token sets
        doc_freq = Counter()
        tok_cache = [None] * N  # cache token sets per record

        for i in range(N):
            toks = set(clean_toks(tgt_names[i], tgt_addrs[i]))
            tok_cache[i] = toks
            for tok in toks:
                doc_freq[tok] += 1

            if i > 0 and i % 500000 == 0:
                log(f"    Tokenized {i:,}/{N:,}...")

        # Compute IDF and build index (skip overly common tokens)
        max_freq = self.max_posting_size
        for tok, freq in doc_freq.items():
            if freq <= max_freq:
                self.idf[tok] = math.log(N / (freq + 1))

        # Build posting lists from cached tokens (no re-tokenization!)
        for i in range(N):
            for tok in tok_cache[i]:
                if tok in self.idf:
                    self.inv_index[tok].append(i)

        del tok_cache, doc_freq
        log(f"  Index built in {time.time()-t0:.1f}s: {len(self.idf):,} tokens (max_posting={max_freq})")

    def query(self, s1_name, s1_addr, top_k):
        toks = clean_toks(s1_name, s1_addr)
        scores = Counter()
        for tok in toks:
            if tok in self.inv_index:
                w = self.idf[tok]
                for idx in self.inv_index[tok]:
                    scores[idx] += w
        return scores.most_common(top_k)


def train_and_optimize():
    log("=" * 60)
    log("PHASE 1: TRAINING & THRESHOLD OPTIMIZATION (v4)")
    log("=" * 60)

    log("Loading Ground Truth...")
    gt_map = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            m = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 else []
            gt_map[p[0]] = set(m)
            if len(gt_map) >= TRAIN_SAMPLE + 20000:
                break
    log(f"  GT loaded: {len(gt_map):,} entities")

    s1_recs = {}
    s1_needed = set(gt_map.keys())
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in s1_needed:
                s1_recs[p[0]] = (p[1], p[2], p[3])
                if len(s1_recs) == len(s1_needed): break
    log(f"  S1 loaded: {len(s1_recs):,}")

    s1_list = list(s1_recs.keys())
    train_ids = set(s1_list[:TRAIN_SAMPLE])
    val_ids = set(s1_list[TRAIN_SAMPLE:])
    log(f"  Train: {len(train_ids):,} | Val: {len(val_ids):,}")

    all_needed = set()
    for sid in s1_needed:
        all_needed |= gt_map[sid]

    log("Loading S2 and S3 targets...")
    tgt_recs = {}
    for sf in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(TRAIN_DIR, sf), encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                p = line.rstrip('\n').split('\t')
                if p[0] in all_needed or i < 500000:
                    tgt_recs[p[0]] = (p[1], p[2], p[3])
    log(f"  Targets loaded: {len(tgt_recs):,}")

    tgt_id_list = list(tgt_recs.keys())
    tgt_names = [tgt_recs[t][0] for t in tgt_id_list]
    tgt_addrs = [tgt_recs[t][1] for t in tgt_id_list]
    tgt_ctry  = {tgt_id_list[i]: tgt_recs[tgt_id_list[i]][2] for i in range(len(tgt_id_list))}

    blocker = IDFBlocker()
    blocker.fit(tgt_id_list, tgt_names, tgt_addrs)

    log("Generating training candidate pairs with IDF blocking...")
    X_train, y_train = [], []
    train_recall_found, train_recall_total = 0, 0

    for sid in train_ids:
        s1_n, s1_a, s1_c = s1_recs[sid]
        true_tgt = gt_map[sid]
        train_recall_total += len(true_tgt)
        results = blocker.query(s1_n, s1_a, top_k=TOP_K)
        for idx, idf_score in results:
            tid = tgt_id_list[idx]
            t_n, t_a = tgt_names[idx], tgt_addrs[idx]
            if tgt_ctry.get(tid) != s1_c: continue
            feat = extract_features(s1_n, s1_a, t_n, t_a, idf_score)
            is_match = 1 if tid in true_tgt else 0
            X_train.append(feat)
            y_train.append(is_match)
            if is_match: train_recall_found += 1

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    log(f"  Training pairs: {len(X_train):,} (Pos: {y_train.sum():,}, Neg: {len(y_train)-y_train.sum():,})")
    log(f"  Training blocking recall: {train_recall_found}/{train_recall_total} ({train_recall_found/train_recall_total*100:.2f}%)")

    log("Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=7, learning_rate=0.08,
        min_samples_leaf=30, l2_regularization=0.5, random_state=42
    )
    clf.fit(X_train, y_train)
    log("  Trained!")

    log(f"Evaluating on {len(val_ids):,} validation entities...")
    val_feats, val_meta = [], []
    val_recall_found, val_recall_total = 0, 0

    for sid in val_ids:
        s1_n, s1_a, s1_c = s1_recs[sid]
        true_tgt = gt_map[sid]
        val_recall_total += len(true_tgt)
        results = blocker.query(s1_n, s1_a, top_k=TOP_K)
        for idx, idf_score in results:
            tid = tgt_id_list[idx]
            t_n, t_a = tgt_names[idx], tgt_addrs[idx]
            if tgt_ctry.get(tid) != s1_c: continue
            feat = extract_features(s1_n, s1_a, t_n, t_a, idf_score)
            val_feats.append(feat)
            val_meta.append((sid, tid))
            if tid in true_tgt: val_recall_found += 1

    log(f"  Validation blocking recall: {val_recall_found}/{val_recall_total} ({val_recall_found/val_recall_total*100:.2f}%)")

    val_feats = np.array(val_feats, dtype=np.float32)
    probs = clf.predict_proba(val_feats)[:, 1]

    gt_val = {sid: gt_map[sid] for sid in val_ids}

    best_t, best_f05 = 0.5, 0.0
    log("Threshold search (0.01 steps)...")
    for t in np.arange(0.40, 0.95, 0.01):
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

    log(f"  BEST THRESHOLD: {best_t:.2f} with Macro F0.5 = {best_f05:.4f}")

    del X_train, y_train, val_feats, probs, tgt_recs, s1_recs, gt_map
    gc.collect()

    return clf, best_t


def run_test_inference(clf, threshold):
    log("=" * 60)
    log("PHASE 2: FULL TEST INFERENCE (v4)")
    log("=" * 60)

    log("Loading test S1 IDs...")
    all_s1_ordered = []
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            all_s1_ordered.append(p[0])
    log(f"  Total test S1: {len(all_s1_ordered):,}")

    final_matches = defaultdict(list)
    final_candidates = defaultdict(list)

    countries = ["France", "US", "India"]

    for country in countries:
        log(f"\n>>> PROCESSING COUNTRY: {country} <<<")
        tc = time.time()

        s1_country = {}
        with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4 and p[3] == country:
                    s1_country[p[0]] = (p[1], p[2])
        log(f"  S1 {country}: {len(s1_country):,}")
        if not s1_country: continue

        tgt_ids, tgt_names, tgt_addrs = [], [], []
        for sf in ["test_source2.tsv", "test_source3.tsv"]:
            with open(os.path.join(TEST_DIR, sf), encoding='utf-8') as f:
                f.readline()
                for line in f:
                    p = line.rstrip('\n').split('\t')
                    if len(p) >= 4 and p[3] == country:
                        tgt_ids.append(p[0])
                        tgt_names.append(p[1])
                        tgt_addrs.append(p[2])
        log(f"  Targets {country}: {len(tgt_ids):,}")

        blocker = IDFBlocker()
        blocker.fit(tgt_ids, tgt_names, tgt_addrs)

        s1_items = list(s1_country.items())
        batch_size = 50000
        country_scored = []

        for b_start in range(0, len(s1_items), batch_size):
            b_end = min(b_start + batch_size, len(s1_items))
            b_feats, b_meta = [], []

            for sid, (s1_n, s1_a) in s1_items[b_start:b_end]:
                results = blocker.query(s1_n, s1_a, top_k=TOP_K)
                cands = []
                for idx, idf_score in results:
                    tid = tgt_ids[idx]
                    cands.append(tid)
                    feat = extract_features(s1_n, s1_a, tgt_names[idx], tgt_addrs[idx], idf_score)
                    b_feats.append(feat)
                    b_meta.append((sid, tid))
                final_candidates[sid] = cands

            if b_feats:
                b_feats_arr = np.array(b_feats, dtype=np.float32)
                b_probs = clf.predict_proba(b_feats_arr)[:, 1]
                for (sid, tid), p in zip(b_meta, b_probs):
                    if p >= threshold:
                        country_scored.append((float(p), sid, tid))

            log(f"    Processed {b_end:,}/{len(s1_items):,} S1, {len(country_scored):,} pairs >= {threshold:.2f}")

        country_scored.sort(reverse=True)
        assigned = set()
        matched_s1 = set()
        for p, sid, tid in country_scored:
            if tid not in assigned:
                final_matches[sid].append(tid)
                assigned.add(tid)
                matched_s1.add(sid)

        log(f"  {country}: Matched {len(matched_s1):,}/{len(s1_country):,} S1 ({len(assigned):,} targets) in {time.time()-tc:.1f}s")
        del s1_country, tgt_ids, tgt_names, tgt_addrs, blocker, country_scored
        gc.collect()

    log("\nWriting matching_results.tsv...")
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    n_matched = 0
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ordered:
            m = final_matches.get(sid, [])
            if m: n_matched += 1
            f.write(f"{sid}\t{','.join(m)}\n")
    log(f"  Matched: {n_matched:,}/{len(all_s1_ordered):,} ({n_matched/len(all_s1_ordered)*100:.1f}%)")

    log("Writing candidate_pairs.tsv...")
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ordered:
            c = final_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(c)}\n")

    log(f"Output: {match_path}")
    log(f"Output: {cand_path}")


def main():
    t0 = time.time()
    log("=" * 60)
    log("AMAZON ML CHALLENGE 2026 — ENTITY RESOLUTION v4")
    log("=" * 60)

    clf, threshold = train_and_optimize()
    run_test_inference(clf, threshold)

    log(f"\nTotal time: {(time.time()-t0)/60:.1f} minutes")

if __name__ == "__main__":
    main()
