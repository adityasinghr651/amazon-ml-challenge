"""
soft_token.py - Guarded Approximate Token Blocker
Provides soft-token matching for informative/rare tokens (DF <= 100)
using prefix-bucketed candidate filtering (no all-pairs token comparisons).
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import rapidfuzz
from rapidfuzz import distance
import pandas as pd
import time

from src.business_entity_resolution.blocking.base import BaseBlocker

class SoftTokenBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "SoftTokenBlocker",
        max_df: int = 100,
        min_token_len: int = 5,
        max_bucket_size: int = 30,
        jw_threshold: float = 0.90,
        lev_threshold: float = 0.82,
        partition_by_country: bool = True
    ):
        super().__init__(name=name)
        self.max_df = max_df
        self.min_token_len = min_token_len
        self.max_bucket_size = max_bucket_size
        self.jw_threshold = jw_threshold
        self.lev_threshold = lev_threshold
        self.partition_by_country = partition_by_country

        # country -> token -> list of entity_ids
        self.token_postings: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> 3-char prefix -> list of unique tokens
        self.prefix_to_tokens: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.target_countries: Set[str] = set()

    def fit(self, target_df: pd.DataFrame) -> 'SoftTokenBlocker':
        t0 = time.time()
        self.token_postings.clear()
        self.prefix_to_tokens.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values
        names = target_df['name_clean'].fillna('').astype(str).values

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        # 1. Compute Document Frequencies of tokens per country
        country_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for idx, name in enumerate(names):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= self.min_token_len and t.isalpha():
                    country_df[c][t] += 1
                    country_df['__GLOBAL__'][t] += 1

        # 2. Build inverted index for rare/informative tokens
        for idx, name in enumerate(names):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            eid = eids[idx]
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= self.min_token_len and t.isalpha():
                    if country_df[c][t] <= self.max_df:
                        self.token_postings[c][t].append(eid)
                    if country_df['__GLOBAL__'][t] <= self.max_df:
                        self.token_postings['__GLOBAL__'][t].append(eid)

        # 3. Build prefix index for fast candidate token lookup
        for c, postings in self.token_postings.items():
            for t in postings.keys():
                prefix = t[:3]
                self.prefix_to_tokens[c][prefix].append(t)

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)
        eids = query_df['entity_id'].values
        names = query_df['name_clean'].fillna('').astype(str).values

        for idx, q_eid in enumerate(eids):
            c_val = countries[idx] if countries[idx] else '__GLOBAL__'
            c_key = c_val if c_val in self.target_countries else '__GLOBAL__'

            q_tokens = set(names[idx].split())
            cand_set = set()

            for q_tok in q_tokens:
                if len(q_tok) < self.min_token_len or not q_tok.isalpha():
                    continue

                prefix = q_tok[:3]
                candidate_tokens = self.prefix_to_tokens.get(c_key, {}).get(prefix, [])

                for t_tok in candidate_tokens:
                    # Skip exact match (already handled by exact/rare blockers)
                    if q_tok == t_tok:
                        continue

                    # Fast length pre-filter
                    if abs(len(q_tok) - len(t_tok)) > 3:
                        continue

                    # Fast Jaro-Winkler check
                    jw = distance.JaroWinkler.similarity(q_tok, t_tok)
                    if jw >= self.jw_threshold:
                        postings = self.token_postings.get(c_key, {}).get(t_tok, [])
                        if 0 < len(postings) <= self.max_bucket_size:
                            cand_set.update(postings)

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates
