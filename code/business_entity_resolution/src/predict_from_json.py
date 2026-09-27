#!/usr/bin/env python3
"""
Zero-Retraining Portable Inference Engine.
Loads model_config.json and model_parameters.json directly.
Evaluates the full Gradient Boosted Tree ensemble in pure Python/NumPy.
No scikit-learn training or pickle files required.
"""
import os, sys, json, math, time, re
import numpy as np

def load_json_model(artifacts_dir):
    config_path = os.path.join(artifacts_dir, "model_config.json")
    params_path = os.path.join(artifacts_dir, "model_parameters.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
        
    return config, params

class PortableGBDTPredictor:
    def __init__(self, config, params):
        self.config = config
        self.params = params
        self.baseline = float(params["baseline_prediction"])
        self.optimal_tau = float(config["decision_threshold"]["optimal_tau"])
        self.trees = params["trees"]
        self.num_trees = len(self.trees)
        
    def predict_proba_single(self, feat_vec):
        """Predict probability for a single 14-dimensional feature vector."""
        raw = self.baseline
        for t in self.trees:
            nodes = t["nodes"]
            curr = 0
            while not nodes[curr]["is_leaf"]:
                f_idx = nodes[curr]["feat"]
                val = feat_vec[f_idx]
                if math.isnan(val):
                    curr = nodes[curr]["left"] if nodes[curr]["miss_left"] else nodes[curr]["right"]
                elif val <= nodes[curr]["th"]:
                    curr = nodes[curr]["left"]
                else:
                    curr = nodes[curr]["right"]
            raw += nodes[curr]["val"]
        return 1.0 / (1.0 + math.exp(-raw))
        
    def predict_proba_batch(self, feat_matrix):
        """Vectorized batch prediction over feature array (N, 14)."""
        return np.array([self.predict_proba_single(row) for row in feat_matrix], dtype=np.float32)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(current_dir, "..", "artifacts")
    
    print(f"Loading JSON model from: {artifacts_dir}...")
    cfg, par = load_json_model(artifacts_dir)
    predictor = PortableGBDTPredictor(cfg, par)
    
    print(f"Model: {cfg['metadata']['model_name']}")
    print(f"Architecture: {cfg['metadata']['model_architecture']}")
    print(f"Number of Trees: {predictor.num_trees}")
    print(f"Optimal Threshold (tau): {predictor.optimal_tau}")
    print(f"Features: {cfg['features']['num_features']}")
    
    # Test sample vector
    sample = np.array([0.8, 0.75, 0.7, 0.9, 0.95, 0.85, 0.6, 0.5, 0.45, 0.8, 0.0, 0.75, 0.65, 4.2], dtype=np.float32)
    p = predictor.predict_proba_single(sample)
    print(f"\nSample Prediction Test:")
    print(f"  P(match) = {p:.4f} (Match: {p >= predictor.optimal_tau})")
    print("\nPortable JSON predictor is ready for use on any machine without retraining.")
