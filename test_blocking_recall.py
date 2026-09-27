import pandas as pd
import re, time
from collections import defaultdict

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci'
}

def get_distinctive_tokens(name, addr):
    text = (name + " " + addr).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = set()
    for w in text.split():
        if len(w) >= 3 and w not in STOPWORDS:
            tokens.add(w)
    return tokens

# Let's test on 1,000 GT pairs
base = r'c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train'
gt = pd.read_csv(f'{base}/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=1000)

s1_set = set()
tgt_set = set()
true_pairs = []
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
    if m_ids:
        s1_set.add(s1_id)
        for m in m_ids:
            tgt_set.add(m)
            true_pairs.append((s1_id, m))

print(f"Testing on {len(true_pairs)} true pairs across {len(s1_set)} S1 entities")

# Load S1 records
s1_records = {}
with open(f'{base}/train_source1.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in s1_set:
            s1_records[p[0]] = (p[1], p[2], p[3])
            if len(s1_records) == len(s1_set): break

# Load Target records
tgt_records = {}
for sf in ['train_source2.tsv', 'train_source3.tsv']:
    with open(f'{base}/{sf}', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in tgt_set:
                tgt_records[p[0]] = (p[1], p[2], p[3])

shared_token_count = 0
char_3g_overlap = 0

for s1_id, tid in true_pairs:
    if s1_id in s1_records and tid in tgt_records:
        s1_n, s1_a, s1_c = s1_records[s1_id]
        t_n, t_a, t_c = tgt_records[tid]
        
        t1 = get_distinctive_tokens(s1_n, s1_a)
        t2 = get_distinctive_tokens(t_n, t_a)
        
        if len(t1 & t2) > 0:
            shared_token_count += 1
        else:
            # Check 3-gram overlap on name
            s1_3g = set([s1_n.lower()[i:i+3] for i in range(len(s1_n)-2)])
            t_3g = set([t_n.lower()[i:i+3] for i in range(len(t_n)-2)])
            if len(s1_3g & t_3g) >= 2:
                char_3g_overlap += 1
            else:
                print(f"MISSED: S1='{s1_n}' | '{s1_a}' vs TGT='{t_n}' | '{t_a}'")

total_eval = len([1 for s1_id, tid in true_pairs if s1_id in s1_records and tid in tgt_records])
print(f"\nTotal evaluated: {total_eval}")
print(f"Distinctive token overlap: {shared_token_count}/{total_eval} ({shared_token_count/total_eval*100:.2f}%)")
print(f"Char 3g overlap on remaining: {char_3g_overlap}")
print(f"Total recall: {(shared_token_count + char_3g_overlap)/total_eval*100:.2f}%")
