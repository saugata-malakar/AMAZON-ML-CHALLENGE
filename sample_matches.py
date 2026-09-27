import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r'c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train'
print("Reading GT...")
gt = pd.read_csv(f'{base}/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=100)

target_needed = set()
s1_needed = set()
pairs = []
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
    if m_ids:
        s1_needed.add(s1_id)
        for m in m_ids[:2]:
            target_needed.add(m)
            pairs.append((s1_id, m))
    if len(pairs) >= 15:
        break

print(f"Need {len(s1_needed)} S1, {len(target_needed)} targets")

s1_data = {}
with open(f'{base}/train_source1.tsv', encoding='utf-8') as f:
    h = f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in s1_needed:
            s1_data[parts[0]] = parts
            if len(s1_data) == len(s1_needed):
                break

tgt_data = {}
for sf in ['train_source2.tsv', 'train_source3.tsv']:
    with open(f'{base}/{sf}', encoding='utf-8') as f:
        h = f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] in target_needed:
                tgt_data[parts[0]] = parts

print("\n=== SAMPLE TRUE MATCHES ===")
for s1_id, tid in pairs:
    if s1_id in s1_data and tid in tgt_data:
        s1 = s1_data[s1_id]
        t = tgt_data[tid]
        print(f"\nMatch: {s1_id} <-> {tid}")
        print(f"  S1:  Name='{s1[1]}' | Addr='{s1[2]}' | Ctry='{s1[3]}'")
        print(f"  TGT: Name='{t[1]}' | Addr='{t[2]}' | Ctry='{t[3]}'")
