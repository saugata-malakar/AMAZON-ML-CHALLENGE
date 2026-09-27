#!/usr/bin/env python3
"""
Point 3: Check cross-country matches DIRECTLY from train_ground_truth.tsv.
No blocking. For each S1 entity in GT, look up each matched target's country
from train_source2/3.tsv and compare to the S1 entity's country.
"""
import sys, os
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train"

print("Loading train_source1 country labels...", flush=True)
s1_country = {}
with open(os.path.join(BASE, "train_source1.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4:
            s1_country[p[0]] = p[3]  # entity_id -> country
print(f"  S1 records: {len(s1_country):,}", flush=True)

print("Loading train_source2 + train_source3 country labels...", flush=True)
tgt_country = {}
for src_file in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(BASE, src_file), encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4:
                tgt_country[p[0]] = p[3]
print(f"  Target records: {len(tgt_country):,}", flush=True)

print("Scanning train_ground_truth.tsv for cross-country pairs (NO blocking)...", flush=True)
total_pairs = 0
cross_country_pairs = 0
s1_missing = 0
tgt_missing = 0
cross_examples = []

with open(os.path.join(BASE, "train_ground_truth.tsv"), encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 2: continue
        s1_id = p[0]
        matches = [x.strip() for x in p[1].split(',') if x.strip()]
        if not matches: continue

        s1_c = s1_country.get(s1_id)
        if s1_c is None:
            s1_missing += 1
            continue

        for tgt_id in matches:
            total_pairs += 1
            tgt_c = tgt_country.get(tgt_id)
            if tgt_c is None:
                tgt_missing += 1
                continue
            if s1_c != tgt_c:
                cross_country_pairs += 1
                if len(cross_examples) < 10:
                    cross_examples.append((s1_id, s1_c, tgt_id, tgt_c))

print(f"\nResult (pure ground-truth check, zero blocking):", flush=True)
print(f"  Total GT pairs scanned: {total_pairs:,}", flush=True)
print(f"  Cross-country pairs: {cross_country_pairs}", flush=True)
print(f"  S1 IDs missing from source1: {s1_missing}", flush=True)
print(f"  Target IDs missing from source2/3: {tgt_missing}", flush=True)

if cross_country_pairs == 0:
    print(f"\n✓ CONFIRMED: Zero cross-country matches in ground truth.", flush=True)
    print("  The same-country blocking restriction loses ZERO recall by construction.", flush=True)
    print("  This result is based on direct GT inspection, not on blocked pairs.", flush=True)
else:
    print(f"\n⚠ WARNING: {cross_country_pairs} cross-country ground-truth matches found!", flush=True)
    print("  Same-country blocking will miss these. Examples:", flush=True)
    for s1_id, s1_c, tgt_id, tgt_c in cross_examples:
        print(f"    S1={s1_id} ({s1_c}) -> target={tgt_id} ({tgt_c})", flush=True)
