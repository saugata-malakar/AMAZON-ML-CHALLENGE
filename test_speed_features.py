import time, os, sys, re, math
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource"
TRAIN_DIR = os.path.join(BASE, "dataset", "train")

STOPWORDS = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'and', 'the', 'of', 'in', 'at', 'road', 'rd', 'street',
    'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar', 'colony',
    'floor', 'near', 'opp', 'opposite', 'block', 'sector', 'phase', 'house',
    'plot', 'door', 'no', 'null', 'sarl', 'sas', 'sci', 'france', 'de', 'la',
    'le', 'du', 'des', 'les', 'en', 'rue', 'bd', 'boulevard', 'av', 'impasse',
    'new', 'old', 'east', 'west', 'north', 'south', 'main', 'cross'
}

def clean_toks_set(text):
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    return set(w for w in clean.split() if len(w) >= 3 and w not in STOPWORDS)

def get_words(text):
    return set(re.findall(r'\w+', text.lower()))

def get_2grams(text):
    t = text.lower()
    if len(t) < 2: return set()
    return set(t[i:i+2] for i in range(len(t)-1))

def get_3grams(text):
    t = text.lower()
    if len(t) < 3: return set()
    return set(t[i:i+3] for i in range(len(t)-2))

def fast_jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def extract_features_fast(s1_n, s1_a, t_n, t_a, idf_score):
    """14 pure-C-backed features — NO slow SequenceMatcher."""
    s1_nc = re.sub(r'[^\w\s]', ' ', s1_n.lower()).strip()
    t_nc  = re.sub(r'[^\w\s]', ' ', t_n.lower()).strip()
    s1_ac = re.sub(r'[^\w\s]', ' ', s1_a.lower()).strip()
    t_ac  = re.sub(r'[^\w\s]', ' ', t_a.lower()).strip()

    n_tok1, n_tok2 = get_words(s1_n), get_words(t_n)
    n_2g1,  n_2g2  = get_2grams(s1_n), get_2grams(t_n)
    n_3g1,  n_3g2  = get_3grams(s1_n), get_3grams(t_n)
    a_tok1, a_tok2 = get_words(s1_a), get_words(t_a)
    a_2g1,  a_2g2  = get_2grams(s1_a), get_2grams(t_a)
    a_3g1,  a_3g2  = get_3grams(s1_a), get_3grams(t_a)

    num1 = set(re.findall(r'\d+', s1_a))
    num2 = set(re.findall(r'\d+', t_a))

    # Prefix match length ratio
    min_len = min(len(s1_nc), len(t_nc))
    pfx_len = 0
    if min_len > 0:
        for i in range(min_len):
            if s1_nc[i] == t_nc[i]: pfx_len += 1
            else: break
        pfx_ratio = pfx_len / min_len
    else:
        pfx_ratio = 0.0

    return [
        # Name features (6)
        fast_jaccard(n_tok1, n_tok2),
        fast_jaccard(n_2g1, n_2g2),
        fast_jaccard(n_3g1, n_3g2),
        pfx_ratio,
        (min(len(s1_nc), len(t_nc)) + 1) / (max(len(s1_nc), len(t_nc)) + 1),
        len(n_tok1 & n_tok2) / (max(len(n_tok1), 1)),

        # Address features (5)
        fast_jaccard(a_tok1, a_tok2),
        fast_jaccard(a_2g1, a_2g2),
        fast_jaccard(a_3g1, a_3g2),
        fast_jaccard(num1, num2),
        1.0 if (not s1_ac or not t_ac) else 0.0,

        # Combined (3)
        fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2),
        fast_jaccard(n_3g1 | a_3g1, n_3g2 | a_3g2),
        idf_score,
    ]

# Benchmark extraction speed
t0 = time.time()
n_trials = 50000
for _ in range(n_trials):
    f = extract_features_fast("Apple Inc California", "1 Infinite Loop Cupertino CA", "Apple Computer Inc", "One Infinite Loop Cupertino", 4.5)
elapsed = time.time() - t0
print(f"50,000 feature extractions took: {elapsed:.2f}s ({n_trials/elapsed:,.0f} pairs/sec)")
