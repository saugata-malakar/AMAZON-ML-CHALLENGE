import time, re
from collections import defaultdict, Counter
import numpy as np

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci', 'france', 'de', 'la',
    'le', 'du', 'des', 'les', 'en', 'rue', 'bd', 'boulevard', 'av', 'impasse'
}

def get_tokens(name, addr):
    text = (name + " " + addr).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) >= 3 and w not in STOPWORDS]

base = r'c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\test'

# 1. Load 100,000 France targets
print("Loading France targets...")
t0 = time.time()
tar_ids = []
tar_names = []
tar_addrs = []
with open(f'{base}/test_source2.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4 and p[3] == 'France':
            tar_ids.append(p[0])
            tar_names.append(p[1])
            tar_addrs.append(p[2])
            if len(tar_ids) >= 100000: break

print(f"Loaded {len(tar_ids)} targets in {time.time()-t0:.2f}s")

# 2. Build index
t1 = time.time()
inv_index = defaultdict(list)
for i in range(len(tar_ids)):
    for tok in set(get_tokens(tar_names[i], tar_addrs[i])):
        inv_index[tok].append(i)

# Filter tokens with freq > 100
inv_index = {k: v for k, v in inv_index.items() if len(v) <= 100}
print(f"Built index in {time.time()-t1:.2f}s with {len(inv_index)} tokens")

# 3. Load 5,000 S1 France entities
s1_ids = []
s1_names = []
s1_addrs = []
with open(f'{base}/test_source1.tsv', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4 and p[3] == 'France':
            s1_ids.append(p[0])
            s1_names.append(p[1])
            s1_addrs.append(p[2])
            if len(s1_ids) >= 5000: break

print(f"Querying {len(s1_ids)} S1 entities...")
t2 = time.time()
cands_found = 0
for i in range(len(s1_ids)):
    toks = get_tokens(s1_names[i], s1_addrs[i])
    counts = Counter()
    for tok in toks:
        if tok in inv_index:
            for idx in inv_index[tok]:
                counts[idx] += 1
    top = counts.most_common(15)
    cands_found += len(top)

q_time = time.time() - t2
print(f"Queried {len(s1_ids)} S1 in {q_time:.2f}s ({len(s1_ids)/q_time:.1f} queries/sec)")
print(f"Total candidates found: {cands_found} (avg {cands_found/len(s1_ids):.1f} per S1)")
