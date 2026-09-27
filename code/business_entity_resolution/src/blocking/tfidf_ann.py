"""
TF-IDF Candidate Retrieval for Blocking.
Uses sparse matrix multiplication for ultra-fast, dependency-free kNN retrieval:
1. Subword / Character n-grams (3-4) for typo and transliteration resilience
2. Word-level n-grams with IDF weighting so rare entity names dominate
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfBlocker:
    def __init__(
        self,
        char_ngram_range: Tuple[int, int] = (3, 4),
        word_ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95
    ):
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )
        self.target_char_mat = None
        self.target_word_mat = None
        self.target_ids: List[str] = []

    def fit_targets(self, target_ids: List[str], target_names: List[str], target_texts: List[str]):
        """
        Fits vectorizers and indexes target records (Source 2 and Source 3).
        """
        self.target_ids = target_ids
        # Fit on all target names and texts
        self.target_char_mat = self.char_vectorizer.fit_transform(target_names)
        self.target_word_mat = self.word_vectorizer.fit_transform(target_texts)

    def query(
        self,
        s1_ids: List[str],
        s1_names: List[str],
        s1_texts: List[str],
        top_k: int = 30,
        char_weight: float = 0.6,
        word_weight: float = 0.4
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Queries target index for each S1 entity using sparse matrix multiplication.
        Returns mapping: s1_id -> list of (target_id, similarity_score).
        """
        if self.target_char_mat is None:
            raise RuntimeError("Blocker has not been fitted on targets.")

        s1_char_mat = self.char_vectorizer.transform(s1_names)
        s1_word_mat = self.word_vectorizer.transform(s1_texts)

        # Sparse dot product = cosine similarity because TfidfVectorizer L2-normalizes
        sim_char = s1_char_mat.dot(self.target_char_mat.T)
        sim_word = s1_word_mat.dot(self.target_word_mat.T)

        combined_sim = char_weight * sim_char + word_weight * sim_word

        results: Dict[str, List[Tuple[str, float]]] = {}
        target_ids_arr = np.array(self.target_ids)

        for i, s1_id in enumerate(s1_ids):
            row = combined_sim.getrow(i)
            if row.nnz == 0:
                results[s1_id] = []
                continue

            indices = row.indices
            scores = row.data

            if len(scores) > top_k:
                top_part = np.argpartition(scores, -top_k)[-top_k:]
                sorted_idx = top_part[np.argsort(-scores[top_part])]
            else:
                sorted_idx = np.argsort(-scores)

            best_target_ids = target_ids_arr[indices[sorted_idx]]
            best_scores = scores[sorted_idx]

            results[s1_id] = list(zip(best_target_ids, [float(s) for s in best_scores]))

        return results
