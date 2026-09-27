"""Quick dataset inspection — headers, sample rows, row counts, country distribution."""
import pandas as pd
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset'

files = {
    'train_s1': f'{BASE}/train/train_source1.tsv',
    'train_s2': f'{BASE}/train/train_source2.tsv',
    'train_s3': f'{BASE}/train/train_source3.tsv',
    'train_gt': f'{BASE}/train/train_ground_truth.tsv',
    'test_s1':  f'{BASE}/test/test_source1.tsv',
    'test_s2':  f'{BASE}/test/test_source2.tsv',
    'test_s3':  f'{BASE}/test/test_source3.tsv',
}

for label, path in files.items():
    print(f'\n{"="*60}')
    print(f'  {label}: {path.split("dataset")[-1]}')
    print(f'{"="*60}')
    
    with open(path, encoding='utf-8') as f:
        line_count = sum(1 for _ in f) - 1
    print(f'  Total rows (excl header): {line_count:,}')
    
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, nrows=3)
    print(f'  Columns: {list(df.columns)}')
    for i, row in df.iterrows():
        d = dict(row)
        for k, v in d.items():
            if len(str(v)) > 80:
                d[k] = str(v)[:80] + '...'
        print(f'  Row {i}: {d}')

# Country distribution
print(f'\n{"="*60}')
print('  COUNTRY DISTRIBUTIONS')
print(f'{"="*60}')
for label, path in files.items():
    if 'gt' in label:
        continue
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, usecols=['country'])
    counts = df['country'].value_counts().to_dict()
    print(f'  {label}: {counts}')

# Ground truth stats
print(f'\n{"="*60}')
print('  GROUND TRUTH ANALYSIS')
print(f'{"="*60}')
gt = pd.read_csv(files['train_gt'], sep='\t', dtype=str, keep_default_na=False)
total = len(gt)
singletons = (gt['matched_entity_ids'].str.strip() == '').sum()
print(f'  Total S1 entities in GT: {total:,}')
print(f'  Singletons (empty matches): {singletons:,} ({singletons/total*100:.2f}%)')
print(f'  Non-singletons: {total - singletons:,} ({(total-singletons)/total*100:.2f}%)')

match_counts = gt['matched_entity_ids'].apply(
    lambda x: len([m for m in x.split(',') if m.strip()]) if x.strip() else 0
)
print(f'  Match count distribution:')
for cnt, freq in sorted(match_counts.value_counts().items()):
    pct = freq/total*100
    print(f'    {cnt} matches: {freq:,} entities ({pct:.2f}%)')

# Uniqueness check
print(f'\n  UNIQUENESS CHECK')
target_to_s1 = {}
multi = 0
for _, row in gt.iterrows():
    s1 = row['source1_entity_id']
    matched = row['matched_entity_ids'].strip()
    if not matched:
        continue
    for m in matched.split(','):
        m = m.strip()
        if m:
            if m in target_to_s1:
                multi += 1
            target_to_s1[m] = s1

print(f'  Unique target records referenced: {len(target_to_s1):,}')
print(f'  Multi-assigned targets (>1 S1): {multi}')
if multi == 0:
    print('  -> SAFE to enforce 1-to-1 assignment constraint!')

# Cross-country check
print(f'\n  CROSS-COUNTRY CHECK')
s1_df = pd.read_csv(files['train_s1'], sep='\t', dtype=str, keep_default_na=False, usecols=['entity_id', 'country'])
s2_df = pd.read_csv(files['train_s2'], sep='\t', dtype=str, keep_default_na=False, usecols=['entity_id', 'country'])
s3_df = pd.read_csv(files['train_s3'], sep='\t', dtype=str, keep_default_na=False, usecols=['entity_id', 'country'])

s1_map = dict(zip(s1_df['entity_id'], s1_df['country']))
tgt_map = dict(zip(s2_df['entity_id'], s2_df['country']))
tgt_map.update(dict(zip(s3_df['entity_id'], s3_df['country'])))

cross = 0
total_pairs = 0
for _, row in gt.iterrows():
    s1 = row['source1_entity_id']
    matched = row['matched_entity_ids'].strip()
    if not matched:
        continue
    s1c = s1_map.get(s1, '')
    for m in matched.split(','):
        m = m.strip()
        if m:
            total_pairs += 1
            tc = tgt_map.get(m, '')
            if s1c and tc and s1c != tc:
                cross += 1

print(f'  Total matched pairs: {total_pairs:,}')
print(f'  Cross-country pairs: {cross}')
if cross == 0:
    print('  -> SAFE to block strictly by country!')

print('\nDONE.')
