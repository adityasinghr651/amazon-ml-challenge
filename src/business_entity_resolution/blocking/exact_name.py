"""
exact_name.py - High-Performance Multi-Representation Exact Name Blocker

Supports independent and modular activation of:
  A1. raw-normalized name (name_clean)
  A2. suffix-stripped core name (name_core)
  A3. token-sorted name (name_sorted)

Applies dynamic, open-set country partitioning with universal fallback.
Completely vectorized with fast dictionary/hash lookups (no iterrows).
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import pandas as pd
import time

from src.business_entity_resolution.blocking.base import BaseBlocker
from src.business_entity_resolution.blocking.config import NameBlockConfig

class ExactNameBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "ExactNameBlocker",
        config: Optional[NameBlockConfig] = None,
        partition_by_country: bool = True
    ):
        super().__init__(name=name, config=config)
        self.cfg = config or NameBlockConfig()
        self.partition_by_country = partition_by_country

        # Indexes: Dict[country, Dict[key_repr, List[entity_id]]]
        # Country '__GLOBAL__' acts as fallback
        self.index_clean: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_core: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_sorted: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        self.target_countries: Set[str] = set()

    def fit(self, target_df: pd.DataFrame) -> 'ExactNameBlocker':
        t0 = time.time()
        
        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        # 1. Index name_clean (A1)
        if self.cfg.use_raw_clean and 'name_clean' in target_df.columns:
            names = target_df['name_clean'].values
            for c, name, eid in zip(countries, names, eids):
                if name:
                    c_key = c if c else '__GLOBAL__'
                    self.index_clean[c_key][name].append(eid)
                    self.index_clean['__GLOBAL__'][name].append(eid)

        # 2. Index name_core (A2)
        if self.cfg.use_core_name and 'name_core' in target_df.columns:
            cores = target_df['name_core'].values
            for c, core, eid in zip(countries, cores, eids):
                if core:
                    c_key = c if c else '__GLOBAL__'
                    self.index_core[c_key][core].append(eid)
                    self.index_core['__GLOBAL__'][core].append(eid)

        # 3. Index name_sorted (A3)
        if self.cfg.use_sorted_tokens and 'name_sorted' in target_df.columns:
            sorted_names = target_df['name_sorted'].values
            for c, s_name, eid in zip(countries, sorted_names, eids):
                if s_name:
                    c_key = c if c else '__GLOBAL__'
                    self.index_sorted[c_key][s_name].append(eid)
                    self.index_sorted['__GLOBAL__'][s_name].append(eid)

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)
        eids = query_df['entity_id'].values

        names_clean = query_df['name_clean'].values if self.cfg.use_raw_clean and 'name_clean' in query_df.columns else None
        names_core = query_df['name_core'].values if self.cfg.use_core_name and 'name_core' in query_df.columns else None
        names_sorted = query_df['name_sorted'].values if self.cfg.use_sorted_tokens and 'name_sorted' in query_df.columns else None

        for idx, q_eid in enumerate(eids):
            c_val = countries[idx] if countries[idx] else '__GLOBAL__'
            c_key = c_val if c_val in self.target_countries else '__GLOBAL__'
            cand_set = set()

            # A1: Match name_clean
            if names_clean is not None:
                nc = names_clean[idx]
                if nc:
                    idx_clean_c = self.index_clean.get(c_key, {})
                    if nc in idx_clean_c:
                        cand_set.update(idx_clean_c[nc])

            # A2: Match name_core
            if names_core is not None:
                n_core = names_core[idx]
                if n_core:
                    idx_core_c = self.index_core.get(c_key, {})
                    if n_core in idx_core_c:
                        cand_set.update(idx_core_c[n_core])

            # A3: Match name_sorted
            if names_sorted is not None:
                ns = names_sorted[idx]
                if ns:
                    idx_sorted_c = self.index_sorted.get(c_key, {})
                    if ns in idx_sorted_c:
                        cand_set.update(idx_sorted_c[ns])

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates
