"""
rare_token.py - Informative Rare-Token Inverted Index Blocker

Indexes distinct, high-information business name tokens.
Rarity is parameterized empirically by document frequency (max_doc_freq)
and minimum token length (min_token_len), selecting the top-N rarest tokens
per entity to prevent combinatorial candidate explosion.

Avoids generic stop words via GENERIC_BUSINESS_WORDS from normalization.
Completely country-neutral with optional country partitioning and global fallback.
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import math
import time

from src.business_entity_resolution.blocking.base import BaseBlocker
from src.business_entity_resolution.blocking.config import RareTokenConfig
from src.business_entity_resolution.normalization import GENERIC_BUSINESS_WORDS

class RareTokenBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "RareTokenBlocker",
        config: Optional[RareTokenConfig] = None,
        partition_by_country: bool = True
    ):
        super().__init__(name=name, config=config)
        self.cfg = config or RareTokenConfig()
        self.partition_by_country = partition_by_country
        self.max_doc_freq = self.cfg.max_doc_freq
        self.min_token_len = self.cfg.min_token_len
        self.top_n_rare = self.cfg.top_n_rare
        self.max_bucket_size = self.cfg.max_bucket_size

        # Inverted Index: Dict[country, Dict[token, List[entity_id]]]
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.target_countries: Set[str] = set()

        # Token statistics
        self.token_doc_freq: Counter = Counter()
        self.qualified_rare_tokens: Set[str] = set()
        self.bucket_sizes: List[int] = []
        self.oversized_tokens: Set[str] = set()

    def _extract_tokens(self, val: Any) -> List[str]:
        """Safely extracts valid token list from string or list object."""
        if isinstance(val, list):
            tokens = val
        elif isinstance(val, str) and val.strip():
            tokens = val.split()
        else:
            tokens = []

        return [
            t.lower() for t in tokens 
            if len(t) >= self.min_token_len and t.lower() not in GENERIC_BUSINESS_WORDS
        ]

    def fit(self, target_df: pd.DataFrame) -> 'RareTokenBlocker':
        t0 = time.time()
        self.token_doc_freq.clear()
        self.index.clear()
        self.bucket_sizes.clear()
        self.oversized_tokens.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        token_col = 'name_informative_tokens' if 'name_informative_tokens' in target_df.columns else 'name_clean'
        token_series = target_df[token_col].values

        # Pass 1: Compute global document frequencies across target records
        for tokens_val in token_series:
            toks = set(self._extract_tokens(tokens_val))
            for t in toks:
                self.token_doc_freq[t] += 1

        # Identify qualified rare tokens (1 < DF <= max_doc_freq)
        self.qualified_rare_tokens = {
            t for t, df in self.token_doc_freq.items()
            if 1 <= df <= self.max_doc_freq
        }

        # Pass 2: Index top-N rarest tokens per target record
        for idx, eid in enumerate(eids):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            toks = list(set(self._extract_tokens(token_series[idx])))
            if not toks:
                continue

            # Filter to qualified rare tokens and sort by frequency ascending (rarest first)
            rare_toks = [t for t in toks if t in self.qualified_rare_tokens]
            rare_toks.sort(key=lambda t: self.token_doc_freq[t])

            selected = rare_toks[:self.top_n_rare]
            for t in selected:
                self.index[c][t].append(eid)
                self.index['__GLOBAL__'][t].append(eid)

        # Record bucket sizes and enforce max_bucket_size safeguard
        for c, t_map in self.index.items():
            for t, postings in list(t_map.items()):
                b_size = len(postings)
                self.bucket_sizes.append(b_size)
                if b_size > self.max_bucket_size:
                    self.oversized_buckets_encountered += 1
                    self.oversized_tokens.add(t)

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)
        eids = query_df['entity_id'].values

        token_col = 'name_informative_tokens' if 'name_informative_tokens' in query_df.columns else 'name_clean'
        token_series = query_df[token_col].values

        for idx, q_eid in enumerate(eids):
            c_val = countries[idx] if countries[idx] else '__GLOBAL__'
            c_key = c_val if c_val in self.target_countries else '__GLOBAL__'

            toks = list(set(self._extract_tokens(token_series[idx])))
            if not toks:
                candidates[q_eid] = set()
                continue

            # Rank by target document frequency ascending (rarest first)
            rare_toks = [t for t in toks if t in self.token_doc_freq and self.token_doc_freq[t] <= self.max_doc_freq]
            rare_toks.sort(key=lambda t: self.token_doc_freq[t])

            selected = rare_toks[:self.top_n_rare]
            cand_set = set()

            t_index = self.index.get(c_key, {})
            for t in selected:
                postings = t_index.get(t, [])
                if 0 < len(postings) <= self.max_bucket_size:
                    cand_set.update(postings)

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates

    def get_token_diagnostics(self) -> Dict[str, Any]:
        """Returns statistics about token frequency distribution and bucket sizing."""
        df_vals = list(self.token_doc_freq.values()) if self.token_doc_freq else [0]
        return {
            "total_unique_tokens": len(self.token_doc_freq),
            "qualified_rare_tokens": len(self.qualified_rare_tokens),
            "df_mean": float(np.mean(df_vals)),
            "df_median": float(np.median(df_vals)),
            "df_p95": float(np.percentile(df_vals, 95)),
            "df_p99": float(np.percentile(df_vals, 99)),
            "df_max": int(np.max(df_vals)),
            "total_indexed_buckets": len(self.bucket_sizes),
            "bucket_mean": float(np.mean(self.bucket_sizes)) if self.bucket_sizes else 0.0,
            "bucket_p95": float(np.percentile(self.bucket_sizes, 95)) if self.bucket_sizes else 0.0,
            "bucket_max": int(np.max(self.bucket_sizes)) if self.bucket_sizes else 0,
            "oversized_token_count": len(self.oversized_tokens)
        }
