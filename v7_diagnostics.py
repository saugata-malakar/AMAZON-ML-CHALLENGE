#!/usr/bin/env python3
"""
v7 Diagnostics — fast, streaming diagnostic checks as requested.
"""
import os, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TEST_DIR = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE, "output")

def check_country_labels():
    print("=" * 70, flush=True)
    print("CHECK 1: literal country values in test files", flush=True)
    print("=" * 70, flush=True)
    for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        counts = Counter()
        path = os.path.join(TEST_DIR, fname)
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4:
                    counts[repr(p[3])] += 1
        print(f"\n{fname}:", flush=True)
        for val, n in counts.most_common():
            print(f"  {val}: {n:,}", flush=True)
    print(
        "\n>> If any country value here is not literally 'France' / 'US' / "
        "'India' (check for extra whitespace, different casing, or an "
        "abbreviation), every `== \"France\"` filter in v7's Phase 6 loop "
        "silently returns zero rows for that country. Fix the string "
        "comparison before doing anything else.",
        flush=True
    )

def check_empty_rate_by_country():
    print("\n" + "=" * 70, flush=True)
    print("CHECK 2: empty-prediction rate by country", flush=True)
    print("=" * 70, flush=True)

    country_of = {}
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                country_of[p[0]] = p[3]

    empty_by_country = Counter()
    total_by_country = Counter()
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    with open(match_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            sid = p[0]
            matches = p[1].strip() if len(p) > 1 else ""
            c = country_of.get(sid, "UNKNOWN")
            total_by_country[c] += 1
            if not matches:
                empty_by_country[c] += 1

    for c, total in total_by_country.items():
        empty = empty_by_country[c]
        pct = 100 * empty / total if total else 0.0
        print(f"  {c}: {empty:,}/{total:,} empty ({pct:.1f}%)", flush=True)

    print(
        "\n>> Training ground truth showed ~5.6% singletons. A country "
        "sitting far above that (especially near 100%) points to a bug "
        "specific to that country's pipeline path, not a general capacity "
        "limit.",
        flush=True
    )

def check_sample_matches(n=15):
    print("\n" + "=" * 70, flush=True)
    print(f"CHECK 3: {n} sample predicted matches (read these by eye)", flush=True)
    print("=" * 70, flush=True)

    # 1. Grab first n matches from matching_results.tsv
    samples = []
    needed_s1 = set()
    needed_tgt = set()
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    with open(match_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            sid = p[0]
            matches = [m for m in p[1].split(",") if m.strip()] if len(p) > 1 else []
            if matches:
                samples.append((sid, matches))
                needed_s1.add(sid)
                needed_tgt.update(matches)
                if len(samples) >= n:
                    break

    # 2. Look up only the needed S1 records
    s1 = {}
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in needed_s1 and len(p) >= 4:
                s1[p[0]] = (p[1], p[2], p[3])
                if len(s1) == len(needed_s1):
                    break

    # 3. Look up only the needed target records
    tgt = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        with open(os.path.join(TEST_DIR, fname), encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[0] in needed_tgt and len(p) >= 4:
                    tgt[p[0]] = (p[1], p[2], p[3])
                    if len(tgt) == len(needed_tgt):
                        break
        if len(tgt) == len(needed_tgt):
            break

    for sid, matches in samples:
        if sid in s1:
            print(f"\n--- {sid} ({s1[sid][2]}) ---", flush=True)
            print(f"  S1 : {s1[sid][0]}  |  {s1[sid][1]}", flush=True)
            for tid in matches:
                if tid in tgt:
                    print(f"  {tid}: {tgt[tid][0]}  |  {tgt[tid][1]}", flush=True)
                else:
                    print(f"  {tid}: (Text record found in target stream)", flush=True)

    print(
        "\n>> If a meaningful fraction of these look like clearly different "
        "businesses, that's a classifier/threshold problem worth "
        "calibrating. If they mostly look reasonable, the score is being "
        "dragged down by missed recall, not bad precision.",
        flush=True
    )

if __name__ == "__main__":
    check_country_labels()
    check_empty_rate_by_country()
    check_sample_matches()
