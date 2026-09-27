"""
retrieval.py - Sparse Character & Word Retrieval Engine with Reciprocal Retrieval
Supports:
  - Character n-gram TF-IDF (char_wb, configurable n-gram range and top-K)
  - Word-level TF-IDF (word n-grams)
  - Reciprocal Retrieval (unsupervised forward + reverse top-K nearest neighbors)
  - Dynamic Country Partitioning with universal fallback
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time

from src.business_entity_resolution.blocking.base import BaseBlocker
from src.business_entity_resolution.blocking.config import RetrievalConfig

class CharNgramRetrievalBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "CharNgramRetrievalBlocker",
        config: Optional[RetrievalConfig] = None,
        top_k: int = 20,
        ngram_range: Tuple[int, int] = (2, 4),
        sim_threshold: float = 0.25,
        reciprocal: bool = False,
        reciprocal_top_k: int = 1,
        partition_by_country: bool = True
    ):
        super().__init__(name=name, config=config)
        self.cfg = config or RetrievalConfig()
        self.partition_by_country = partition_by_country
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.sim_threshold = sim_threshold
        self.reciprocal = reciprocal
        self.reciprocal_top_k = reciprocal_top_k
        self.batch_size = self.cfg.batch_size

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.country_matrices: Dict[str, sp.csr_matrix] = {}
        self.country_eids: Dict[str, np.ndarray] = {}
        self.global_matrix: Optional[sp.csr_matrix] = None
        self.global_eids: Optional[np.ndarray] = None
        self.target_countries: Set[str] = set()

    def fit(self, target_df: pd.DataFrame) -> 'CharNgramRetrievalBlocker':
        t0 = time.time()
        self.country_matrices.clear()
        self.country_eids.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        names = target_df['name_clean'].fillna('').astype(str).values
        eids = target_df['entity_id'].values
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        # Fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=self.ngram_range,
            min_df=self.cfg.min_df,
            max_df=self.cfg.max_df,
            sublinear_tf=True,
            dtype=np.float32
        )
        X_all = self.vectorizer.fit_transform(names)
        self.global_matrix = X_all
        self.global_eids = eids

        # Partition target matrices by country
        if has_country:
            unique_c = target_df['country_norm'].unique()
            for c in unique_c:
                mask = (countries == c)
                self.country_matrices[c] = X_all[mask]
                self.country_eids[c] = eids[mask]

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        if self.vectorizer is None or self.global_matrix is None:
            return {eid: set() for eid in query_df['entity_id']}

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        query_names = query_df['name_clean'].fillna('').astype(str).values
        query_eids = query_df['entity_id'].values
        query_countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)

        unique_q_countries = query_df['country_norm'].unique() if has_country else ['__GLOBAL__']
        for c_val in unique_q_countries:
            mask = (query_countries == c_val) if has_country else np.ones(len(query_df), dtype=bool)
            if not np.any(mask):
                continue

            c_key = c_val if (has_country and c_val in self.country_matrices) else '__GLOBAL__'
            if has_country and c_key in self.country_matrices:
                target_mat = self.country_matrices[c_key]
                target_eids = self.country_eids[c_key]
            else:
                target_mat = self.global_matrix
                target_eids = self.global_eids

            if target_mat.shape[0] == 0:
                for q_eid in query_eids[mask]:
                    candidates[q_eid] = set()
                continue

            sub_names = query_names[mask]
            sub_eids = query_eids[mask]

            for chunk_start in range(0, len(sub_names), self.batch_size):
                chunk_end = min(chunk_start + self.batch_size, len(sub_names))
                chunk_names = sub_names[chunk_start:chunk_end]
                chunk_eids = sub_eids[chunk_start:chunk_end]

                X_chunk = self.vectorizer.transform(chunk_names)
                scores_mat = X_chunk.dot(target_mat.T).tocsr()

                indptr = scores_mat.indptr
                data = scores_mat.data
                indices = scores_mat.indices

                # 1. Forward Retrieval: query -> top-K targets
                for i in range(len(chunk_eids)):
                    q_eid = chunk_eids[i]
                    start_ptr = indptr[i]
                    end_ptr = indptr[i + 1]

                    if start_ptr == end_ptr:
                        continue

                    row_data = data[start_ptr:end_ptr]
                    row_indices = indices[start_ptr:end_ptr]

                    valid_mask = row_data >= self.sim_threshold
                    if not np.any(valid_mask):
                        continue

                    f_data = row_data[valid_mask]
                    f_indices = row_indices[valid_mask]

                    if len(f_data) > self.top_k:
                        top_k_idx = np.argpartition(-f_data, self.top_k)[:self.top_k]
                        best_indices = f_indices[top_k_idx]
                    else:
                        best_indices = f_indices

                    cands = set(target_eids[best_indices])
                    candidates[q_eid].update(cands)

                # 2. Reciprocal Retrieval: target -> top-K_rev queries
                if self.reciprocal:
                    rev_mat = scores_mat.T.tocsr()
                    r_indptr = rev_mat.indptr
                    r_data = rev_mat.data
                    r_indices = rev_mat.indices

                    for j in range(rev_mat.shape[0]):
                        r_start = r_indptr[j]
                        r_end = r_indptr[j + 1]
                        if r_start == r_end:
                            continue

                        col_data = r_data[r_start:r_end]
                        col_indices = r_indices[r_start:r_end]

                        r_valid = col_data >= self.sim_threshold
                        if not np.any(r_valid):
                            continue

                        rf_data = col_data[r_valid]
                        rf_indices = col_indices[r_valid]

                        if len(rf_data) > self.reciprocal_top_k:
                            top_r_idx = np.argpartition(-rf_data, self.reciprocal_top_k)[:self.reciprocal_top_k]
                            best_q_indices = rf_indices[top_r_idx]
                        else:
                            best_q_indices = rf_indices

                        t_eid = target_eids[j]
                        for q_idx in best_q_indices:
                            q_eid = chunk_eids[q_idx]
                            candidates[q_eid].add(t_eid)

        # Count total
        for q_eid in query_df['entity_id']:
            if q_eid not in candidates:
                candidates[q_eid] = set()
            self.total_candidates_generated += len(candidates[q_eid])

        self.query_time = time.time() - t0
        return candidates


class WordTfidfRetrievalBlocker(BaseBlocker):
    """
    Sparse Word-Level TF-IDF Retrieval Blocker.
    Retrieves candidates based on whole-word cosine similarity.
    """
    def __init__(
        self,
        name: str = "WordTfidfRetrievalBlocker",
        top_k: int = 15,
        ngram_range: Tuple[int, int] = (1, 2),
        sim_threshold: float = 0.35,
        batch_size: int = 1000,
        partition_by_country: bool = True
    ):
        super().__init__(name=name)
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.sim_threshold = sim_threshold
        self.batch_size = batch_size
        self.partition_by_country = partition_by_country

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.country_matrices: Dict[str, sp.csr_matrix] = {}
        self.country_eids: Dict[str, np.ndarray] = {}
        self.global_matrix: Optional[sp.csr_matrix] = None
        self.global_eids: Optional[np.ndarray] = None
        self.target_countries: Set[str] = set()

    def fit(self, target_df: pd.DataFrame) -> 'WordTfidfRetrievalBlocker':
        t0 = time.time()
        self.country_matrices.clear()
        self.country_eids.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        names = target_df['name_clean'].fillna('').astype(str).values
        eids = target_df['entity_id'].values
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=self.ngram_range,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            dtype=np.float32
        )
        X_all = self.vectorizer.fit_transform(names)
        self.global_matrix = X_all
        self.global_eids = eids

        if has_country:
            unique_c = target_df['country_norm'].unique()
            for c in unique_c:
                mask = (countries == c)
                self.country_matrices[c] = X_all[mask]
                self.country_eids[c] = eids[mask]

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        if self.vectorizer is None or self.global_matrix is None:
            return {eid: set() for eid in query_df['entity_id']}

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        query_names = query_df['name_clean'].fillna('').astype(str).values
        query_eids = query_df['entity_id'].values
        query_countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)

        unique_q_countries = query_df['country_norm'].unique() if has_country else ['__GLOBAL__']
        for c_val in unique_q_countries:
            mask = (query_countries == c_val) if has_country else np.ones(len(query_df), dtype=bool)
            if not np.any(mask):
                continue

            c_key = c_val if (has_country and c_val in self.country_matrices) else '__GLOBAL__'
            if has_country and c_key in self.country_matrices:
                target_mat = self.country_matrices[c_key]
                target_eids = self.country_eids[c_key]
            else:
                target_mat = self.global_matrix
                target_eids = self.global_eids

            if target_mat.shape[0] == 0:
                continue

            sub_names = query_names[mask]
            sub_eids = query_eids[mask]

            for chunk_start in range(0, len(sub_names), self.batch_size):
                chunk_end = min(chunk_start + self.batch_size, len(sub_names))
                chunk_names = sub_names[chunk_start:chunk_end]
                chunk_eids = sub_eids[chunk_start:chunk_end]

                X_chunk = self.vectorizer.transform(chunk_names)
                scores_mat = X_chunk.dot(target_mat.T).tocsr()

                indptr = scores_mat.indptr
                data = scores_mat.data
                indices = scores_mat.indices

                for i in range(len(chunk_eids)):
                    q_eid = chunk_eids[i]
                    start_ptr = indptr[i]
                    end_ptr = indptr[i + 1]

                    if start_ptr == end_ptr:
                        continue

                    row_data = data[start_ptr:end_ptr]
                    row_indices = indices[start_ptr:end_ptr]

                    valid_mask = row_data >= self.sim_threshold
                    if not np.any(valid_mask):
                        continue

                    f_data = row_data[valid_mask]
                    f_indices = row_indices[valid_mask]

                    if len(f_data) > self.top_k:
                        top_k_idx = np.argpartition(-f_data, self.top_k)[:self.top_k]
                        best_indices = f_indices[top_k_idx]
                    else:
                        best_indices = f_indices

                    cands = set(target_eids[best_indices])
                    candidates[q_eid].update(cands)

        for q_eid in query_df['entity_id']:
            if q_eid not in candidates:
                candidates[q_eid] = set()
            self.total_candidates_generated += len(candidates[q_eid])

        self.query_time = time.time() - t0
        return candidates
