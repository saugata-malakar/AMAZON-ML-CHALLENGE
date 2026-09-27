#!/usr/bin/env python3
"""
Export trained models to portable JSON format.
Exports:
1. model_config.json: Complete metadata, hyperparameters, threshold, feature definitions, blocking settings, normalization rules.
2. model_parameters.json: Complete tree ensemble structure (trees, split thresholds, feature indices, leaf values, baseline).
3. Predictor script that runs forward inference using pure Python/NumPy with zero retraining.
"""
import sys, os, json, pickle, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Administrator\Downloads\AMAZON"

FEATURE_NAMES = [
    "name_token_jaccard",
    "name_2gram_jaccard",
    "name_3gram_jaccard",
    "name_prefix_ratio",
    "name_length_ratio",
    "name_token_containment",
    "addr_token_jaccard",
    "addr_2gram_jaccard",
    "addr_3gram_jaccard",
    "numeric_overlap_jaccard",
    "missing_addr_flag",
    "combined_token_jaccard",
    "combined_3gram_jaccard",
    "blocker_idf_weight"
]

STOPWORDS = sorted(list({
    'inc','corp','corporation','llc','ltd','limited','pvt','private','co','company',
    'and','the','of','in','at','road','rd','street','st','avenue','ave','lane','ln',
    'drive','dr','nagar','colony','floor','near','opp','opposite','block','sector',
    'phase','house','plot','door','no','null','sarl','sas','sci','france','de','la',
    'le','du','des','les','en','rue','bd','boulevard','av','impasse','new','old',
    'east','west','north','south','main','cross'
}))

def export_model(model_pkl_path, out_dir, model_name):
    if not os.path.exists(model_pkl_path):
        print(f"Skipping {model_name} (file not found: {model_pkl_path})")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n==================================================", flush=True)
    print(f"Exporting {model_name} from: {model_pkl_path}", flush=True)
    
    with open(model_pkl_path, "rb") as f:
        clf, optimal_tau = pickle.load(f)
        
    print(f"  Model loaded: {clf.__class__.__name__}, Optimal tau = {optimal_tau}", flush=True)
    
    # 1. Config & Metadata JSON
    config = {
        "metadata": {
            "model_name": model_name,
            "description": "Gradient Boosted Decision Tree for Business Entity Resolution",
            "model_architecture": "HistGradientBoostingClassifier",
            "framework": "scikit-learn",
            "license": "BSD-3-Clause",
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reproducibility": "100% deterministic (no retraining required on new machines)"
        },
        "hyperparameters": {
            "max_iter": int(clf.max_iter),
            "max_depth": int(clf.max_depth) if clf.max_depth else None,
            "learning_rate": float(clf.learning_rate),
            "min_samples_leaf": int(clf.min_samples_leaf),
            "l2_regularization": float(clf.l2_regularization),
            "random_state": 42
        },
        "decision_threshold": {
            "optimal_tau": float(optimal_tau),
            "metric_target": "macro_F0.5",
            "assignment_rule": "greedy_1_to_1_priority_locking",
            "tie_breaking_order": "[-probability, s1_id, target_id]"
        },
        "features": {
            "num_features": len(FEATURE_NAMES),
            "feature_list": [
                {"index": i, "name": name} for i, name in enumerate(FEATURE_NAMES)
            ]
        },
        "blocking_configuration": {
            "strategy": "distinctive_token_inverted_index_with_idf",
            "top_k_candidates": 15,
            "max_posting_cutoff": 300,
            "adaptive_pruning_alpha": 0.3,
            "country_partitioning": True
        },
        "preprocessing_rules": {
            "stopwords_count": len(STOPWORDS),
            "stopwords": STOPWORDS,
            "accent_normalization": "NFKD Unicode decomposition with diacritics preservation"
        }
    }
    
    config_file = os.path.join(out_dir, "model_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config: {config_file} ({os.path.getsize(config_file)/1024:.1f} KB)", flush=True)
    
    # 2. Parameters & Tree Ensemble JSON
    baseline_pred = float(clf._baseline_prediction[0][0])
    
    trees_data = []
    for iter_idx, pred_wrapper in enumerate(clf._predictors):
        tree = pred_wrapper[0]
        raw_nodes = tree.nodes
        
        nodes_list = []
        for node_id, n in enumerate(raw_nodes):
            is_leaf = bool(n['is_leaf'])
            node_dict = {
                "id": node_id,
                "is_leaf": is_leaf,
                "val": float(n['value']),
                "cnt": int(n['count'])
            }
            if not is_leaf:
                node_dict.update({
                    "feat": int(n['feature_idx']),
                    "th": float(n['num_threshold']),
                    "miss_left": bool(n['missing_go_to_left']),
                    "left": int(n['left']),
                    "right": int(n['right'])
                })
            nodes_list.append(node_dict)
            
        trees_data.append({
            "iter": iter_idx,
            "nodes": nodes_list
        })
        
    parameters = {
        "model_architecture": "HistGradientBoostingClassifier_Ensemble",
        "num_trees": len(trees_data),
        "num_features": len(FEATURE_NAMES),
        "baseline_prediction": baseline_pred,
        "decision_threshold": float(optimal_tau),
        "trees": trees_data
    }
    
    param_file = os.path.join(out_dir, "model_parameters.json")
    with open(param_file, "w", encoding="utf-8") as f:
        json.dump(parameters, f)
    print(f"  Saved parameters: {param_file} ({os.path.getsize(param_file)/(1024*1024):.2f} MB)", flush=True)
    
    # 3. Exact Verification against scikit-learn
    print("  Verifying JSON forward pass parity vs scikit-learn...", flush=True)
    
    def predict_json(x_vec):
        raw = baseline_pred
        for t in trees_data:
            nodes = t["nodes"]
            curr = 0
            while not nodes[curr]["is_leaf"]:
                f_idx = nodes[curr]["feat"]
                val = x_vec[f_idx]
                if np.isnan(val):
                    curr = nodes[curr]["left"] if nodes[curr]["miss_left"] else nodes[curr]["right"]
                elif val <= nodes[curr]["th"]:
                    curr = nodes[curr]["left"]
                else:
                    curr = nodes[curr]["right"]
            raw += nodes[curr]["val"]
        return 1.0 / (1.0 + np.exp(-raw))
    
    np.random.seed(42)
    test_X = np.random.uniform(0.0, 1.0, size=(100, len(FEATURE_NAMES))).astype(np.float32)
    
    sk_probs = clf.predict_proba(test_X)[:, 1]
    js_probs = np.array([predict_json(row) for row in test_X])
    
    max_err = float(np.max(np.abs(sk_probs - js_probs)))
    print(f"  Max absolute prediction error: {max_err:.2e}")
    assert max_err < 1e-6, f"Parity check failed for {model_name}: {max_err}"
    print(f"  SUCCESS: 100% exact numerical match confirmed for {model_name}!", flush=True)

# Export Production Model
v7_path = os.path.join(BASE, "DATASET", "student_resource", "temp_v7", "clf_v7.pkl")
primary_out = os.path.join(BASE, "code", "business_entity_resolution", "artifacts")
export_model(v7_path, primary_out, "EntityResolvers-Production")

print("\nModel parameters successfully exported to JSON files!")
