#!/usr/bin/env python3
"""
Extend the v6 tau sweep to 0.90-0.99 using the saved clf.
No retraining. Rebuilds index + val predictions only (~3-4 min).
"""
import sys, os, re, math, time, gc, pickle
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')
BASE      = r"c:\Users\Administrator\Downloads\AMAZON\DATASET\student_resource\dataset\train"
TEMP_DIR  = r"c:\Users\Administrator\Downloads\AMAZON\temp_v6"
STOPWORDS = {
    'inc','corp','corporation','llc','ltd','limited','pvt','private','co','company',
    'and','the','of','in','at','road','rd','street','st','avenue','ave','lane','ln',
    'drive','dr','nagar','colony','floor','near','opp','opposite','block','sector',
    'phase','house','plot','door','no','null','sarl','sas','sci','france','de','la',
    'le','du','des','les','en','rue','bd','boulevard','av','impasse','new','old',
    'east','west','north','south','main','cross'
}
def clean_toks(n, a):
    text = (n+" "+a).lower()
    return [w for w in re.sub(r'[^\w\s]',' ',text).split() if len(w)>=3 and w not in STOPWORDS]
def get_3grams_str(text):
    t = re.sub(r'[^\w]','',text.lower())
    return list(t[i:i+3] for i in range(len(t)-2)) if len(t)>=3 else []
def get_words(t): return set(re.findall(r'\w+',t.lower()))
def get_2grams(t): return set(t.lower()[i:i+2] for i in range(len(t.lower())-1)) if len(t)>=2 else set()
def get_3grams(t): return set(t.lower()[i:i+3] for i in range(len(t.lower())-2)) if len(t)>=3 else set()
def jac(a,b): return len(a&b)/len(a|b) if (a and b) else 0.0
def precompute_s1(n,a):
    nc=re.sub(r'[^\w\s]',' ',n.lower()).strip(); ac=re.sub(r'[^\w\s]',' ',a.lower()).strip()
    return (get_words(n),get_2grams(n),get_3grams(n),get_words(a),get_2grams(a),get_3grams(a),set(re.findall(r'\d+',a)),nc,ac)
def extract_features(s1p, tn, ta, idf_w):
    (nt1,n2g1,n3g1,at1,a2g1,a3g1,num1,s1nc,s1ac)=s1p
    tnc=re.sub(r'[^\w\s]',' ',tn.lower()).strip(); tac=re.sub(r'[^\w\s]',' ',ta.lower()).strip()
    nt2=get_words(tn);n2g2=get_2grams(tn);n3g2=get_3grams(tn)
    at2=get_words(ta);a2g2=get_2grams(ta);a3g2=get_3grams(ta);num2=set(re.findall(r'\d+',ta))
    ml=min(len(s1nc),len(tnc)); pfx=sum(1 for i in range(ml) if s1nc[i]==tnc[i] and (i==0 or s1nc[i-1]==tnc[i-1]))/ml if ml>0 else 0.0
    return [jac(nt1,nt2),jac(n2g1,n2g2),jac(n3g1,n3g2),pfx,
            (min(len(s1nc),len(tnc))+1)/(max(len(s1nc),len(tnc))+1),
            len(nt1&nt2)/max(len(nt1),1),jac(at1,at2),jac(a2g1,a2g2),jac(a3g1,a3g2),
            jac(num1,num2),1.0 if (not s1ac or not tac) else 0.0,
            jac(nt1|at1,nt2|at2),jac(n3g1|a3g1,n3g2|a3g2),idf_w]

def macro_f05(gt_dict, pred_dict):
    scores=[]
    for sid,tt in gt_dict.items():
        pp=set(pred_dict.get(sid,[]))
        if not tt: scores.append(1.0 if not pp else 0.0)
        else:
            if not pp: scores.append(0.0)
            else:
                tp=len(pp&tt); p=tp/len(pp); r=tp/len(tt); d=0.25*p+r
                scores.append((1.25*p*r)/d if d>0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

# ── Load model ──────────────────────────────────────────────
print("Loading saved v6 model...", flush=True)
with open(os.path.join(TEMP_DIR,"clf_v6.pkl"),"rb") as f:
    clf, old_best_tau = pickle.load(f)
print(f"  Model loaded. Previous best tau={old_best_tau}", flush=True)

# ── Rebuild training target pool (same 60k GT, same sources) ─
TOTAL_N = 60000; VAL_START = 40000
print("Reloading GT + records...", flush=True)
t0=time.time()
gt_map={}
with open(os.path.join(BASE,"train_ground_truth.tsv"),encoding='utf-8') as f:
    f.readline()
    for line in f:
        p=line.rstrip('\n').split('\t')
        m=[x.strip() for x in p[1].split(',') if x.strip()] if len(p)>1 else []
        gt_map[p[0]]=set(m)
        if len(gt_map)>=TOTAL_N: break

s1_recs={}
s1_needed=set(gt_map.keys())
with open(os.path.join(BASE,"train_source1.tsv"),encoding='utf-8') as f:
    f.readline()
    for line in f:
        p=line.rstrip('\n').split('\t')
        if p[0] in s1_needed:
            s1_recs[p[0]]=(p[1],p[2],p[3])
            if len(s1_recs)==len(s1_needed): break

all_needed=set(); [all_needed.update(v) for v in gt_map.values()]
tgt_recs={}
for sf in ["train_source2.tsv","train_source3.tsv"]:
    with open(os.path.join(BASE,sf),encoding='utf-8') as f:
        f.readline()
        for i,line in enumerate(f):
            p=line.rstrip('\n').split('\t')
            if p[0] in all_needed or i<500000:
                tgt_recs[p[0]]=(p[1],p[2],p[3])

tgt_id_list=list(tgt_recs.keys()); N=len(tgt_id_list)
tgt_ctry={t:tgt_recs[t][2] for t in tgt_id_list}
print(f"  Target pool: {N:,} | {time.time()-t0:.1f}s", flush=True)

# ── Rebuild dual index ───────────────────────────────────────
print("Rebuilding dual index...", flush=True)
t0=time.time()
raw_inv=defaultdict(list)
cg_inv=defaultdict(list)
for i,tid in enumerate(tgt_id_list):
    tn,ta,_=tgt_recs[tid]
    for tok in set(clean_toks(tn,ta)): raw_inv[tok].append(i)
    for g in set(get_3grams_str(tn+" "+ta)): cg_inv[g].append(i)
inv_index={tok:v for tok,v in raw_inv.items() if len(v)<=300}
idf={tok:math.log(N/(len(v)+1)) for tok,v in inv_index.items()}
cg_inv_f={g:v for g,v in cg_inv.items() if 2<=len(v)<=500}
cg_idf={g:math.log(N/(len(v)+1)) for g,v in cg_inv_f.items()}
del raw_inv,cg_inv; gc.collect()
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

def query_dual(name,addr,country,k=15):
    s1=Counter()
    for tok in clean_toks(name,addr):
        if tok in inv_index:
            w=idf[tok]
            for idx in inv_index[tok]: s1[idx]+=w
    c1={(tgt_id_list[i],w) for i,w in s1.most_common(k) if tgt_ctry.get(tgt_id_list[i])==country}
    s2=Counter()
    for g in set(get_3grams_str(name+" "+addr)):
        if g in cg_inv_f:
            w=cg_idf[g]
            for idx in cg_inv_f[g]: s2[idx]+=w
    c2={(tgt_id_list[i],w) for i,w in s2.most_common(k) if tgt_ctry.get(tgt_id_list[i])==country}
    merged={}
    for tid,w in c2: merged[tid]=w
    for tid,w in c1: merged[tid]=w  # blocker1 score wins
    return list(merged.items())

# ── Compute val predictions ──────────────────────────────────
s1_list=list(s1_recs.keys())
val_ids=s1_list[VAL_START:TOTAL_N]
print(f"Computing val features for {len(val_ids):,} entities...", flush=True)
t0=time.time()
val_feats,val_labels,val_meta=[],[],[]
for sid in val_ids:
    s1n,s1a,s1c=s1_recs[sid]
    s1p=precompute_s1(s1n,s1a)
    for tid,idf_w in query_dual(s1n,s1a,s1c,k=15):
        val_feats.append(extract_features(s1p,tgt_recs[tid][0],tgt_recs[tid][1],idf_w))
        val_labels.append(1 if tid in gt_map[sid] else 0)
        val_meta.append((sid,tid))
val_feats=np.array(val_feats,dtype=np.float32)
probs=clf.predict_proba(val_feats)[:,1]
print(f"  {len(val_feats):,} pairs scored in {time.time()-t0:.1f}s", flush=True)

# ── Extended sweep 0.40 → 0.99 ──────────────────────────────
print("\nFull tau sweep (0.40 → 0.99):", flush=True)
gt_val={sid:gt_map[sid] for sid in val_ids}

best_tau,best_f05=0.0,0.0
# Pre-sort once
sorted_scored=sorted(zip(probs,[m[0] for m in val_meta],[m[1] for m in val_meta]),reverse=True)

prev_f05=None
for tau in list(np.arange(0.40,0.90,0.04)) + list(np.arange(0.90,1.00,0.02)):
    tau=round(tau,2)
    assigned=set(); preds=defaultdict(list)
    for p,sid,tid in sorted_scored:
        if p<tau: break
        if tid not in assigned: preds[sid].append(tid); assigned.add(tid)
    f05=macro_f05(gt_val,preds)
    delta=f"(+{f05-prev_f05:.4f})" if prev_f05 is not None else ""
    marker="  ◄ BEST" if f05>best_f05 else ""
    print(f"  tau={tau:.2f}: F0.5={f05:.4f} {delta}{marker}", flush=True)
    if f05>best_f05: best_f05=f05; best_tau=tau
    prev_f05=f05

print(f"\n>>> True optimum: tau={best_tau:.2f}, F0.5={best_f05:.4f} <<<", flush=True)
print(f"    v5 baseline:   tau=0.70,  F0.5=0.7910", flush=True)
print(f"    v6 net gain:  +{best_f05-0.7910:.4f} F0.5", flush=True)

# Save updated best tau
with open(os.path.join(TEMP_DIR,"clf_v6.pkl"),"wb") as f:
    pickle.dump((clf, best_tau), f)
print(f"\nUpdated model saved with tau={best_tau}", flush=True)
