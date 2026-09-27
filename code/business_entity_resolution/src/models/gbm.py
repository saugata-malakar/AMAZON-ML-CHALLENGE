"""
GBM Pairwise Matching Classifier:
Uses Gradient Boosted Decision Trees (HistGradientBoostingClassifier / LightGBM)
to output calibrated match probabilities P(Match | S1, Candidate).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class EntityMatcherModel:
    def __init__(self, random_state: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state
        )
        self.feature_cols: List[str] = []

    def fit(self, df_train_features: pd.DataFrame):
        """
        Fits the GBDT model on candidate pairs with binary 'label' column.
        """
        exclude_cols = {"s1_id", "cand_id", "label"}
        self.feature_cols = [c for c in df_train_features.columns if c not in exclude_cols]

        X = df_train_features[self.feature_cols].values
        y = df_train_features["label"].values

        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        print(f"Fitting EntityMatcherModel on {len(y):,} pairs ({pos_count:,} positive, {neg_count:,} negative)...")

        self.model.fit(X, y)
        print("Model training complete.")

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Predicts match probability for each candidate pair.
        """
        if not self.feature_cols:
            raise RuntimeError("Model has not been fitted yet.")

        X = df_features[self.feature_cols].values
        # Return probability of class 1 (match)
        return self.model.predict_proba(X)[:, 1]
