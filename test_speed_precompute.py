import time, re

def get_words(text): return set(re.findall(r'\w+', text.lower()))
def get_2grams(text):
    t = text.lower()
    return set(t[i:i+2] for i in range(len(t)-1)) if len(t) >= 2 else set()
def get_3grams(text):
    t = text.lower()
    return set(t[i:i+3] for i in range(len(t)-2)) if len(t) >= 3 else set()
def fast_jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2) if (s1 and s2) else 0.0

def precompute_s1(n, a):
    nc = re.sub(r'[^\w\s]', ' ', n.lower()).strip()
    ac = re.sub(r'[^\w\s]', ' ', a.lower()).strip()
    return (
        get_words(n), get_2grams(n), get_3grams(n),
        get_words(a), get_2grams(a), get_3grams(a),
        set(re.findall(r'\d+', a)),
        nc, ac
    )

def extract_features_precomputed(s1_pre, t_n, t_a, idf_score):
    (n_tok1, n_2g1, n_3g1, a_tok1, a_2g1, a_3g1, num1, s1_nc, s1_ac) = s1_pre
    t_nc  = re.sub(r'[^\w\s]', ' ', t_n.lower()).strip()
    t_ac  = re.sub(r'[^\w\s]', ' ', t_a.lower()).strip()
    n_tok2 = get_words(t_n)
    n_2g2  = get_2grams(t_n)
    n_3g2  = get_3grams(t_n)
    a_tok2 = get_words(t_a)
    a_2g2  = get_2grams(t_a)
    a_3g2  = get_3grams(t_a)
    num2   = set(re.findall(r'\d+', t_a))

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
        fast_jaccard(n_tok1, n_tok2),
        fast_jaccard(n_2g1, n_2g2),
        fast_jaccard(n_3g1, n_3g2),
        pfx_ratio,
        (min(len(s1_nc), len(t_nc)) + 1) / (max(len(s1_nc), len(t_nc)) + 1),
        len(n_tok1 & n_tok2) / (max(len(n_tok1), 1)),

        fast_jaccard(a_tok1, a_tok2),
        fast_jaccard(a_2g1, a_2g2),
        fast_jaccard(a_3g1, a_3g2),
        fast_jaccard(num1, num2),
        1.0 if (not s1_ac or not t_ac) else 0.0,

        fast_jaccard(n_tok1 | a_tok1, n_tok2 | a_tok2),
        fast_jaccard(n_3g1 | a_3g1, n_3g2 | a_3g2),
        idf_score,
    ]

# Benchmark 50,000 queries with 15 candidates each = 750,000 pairs!
t0 = time.time()
n_entities = 50000
for _ in range(n_entities):
    s1_pre = precompute_s1("Apple Inc California", "1 Infinite Loop Cupertino CA")
    for _ in range(15):
        f = extract_features_precomputed(s1_pre, "Apple Computer Inc", "One Infinite Loop Cupertino", 4.5)
elapsed = time.time() - t0
print(f"50,000 S1 x 15 candidates (750,000 pairs) took: {elapsed:.2f}s ({750000/elapsed:,.0f} pairs/sec)")
