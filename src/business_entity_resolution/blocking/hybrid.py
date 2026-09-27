"""
hybrid.py - Name + Numeric Hybrid Blocker
Creates discriminative composite anchors between rare name tokens and address numbers/postal codes.
Enforces strict bucket size caps to eliminate generic address collision risk.
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import re
import pandas as pd
import time

from src.business_entity_resolution.blocking.base import BaseBlocker

NUM_RE = re.compile(r'\b\d+\b')

class NameNumericHybridBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "NameNumericHybridBlocker",
        max_bucket_size: int = 15,
        max_df: int = 100,
        partition_by_country: bool = True
    ):
        super().__init__(name=name)
        self.max_bucket_size = max_bucket_size
        self.max_df = max_df
        self.partition_by_country = partition_by_country

        # country -> composite_key -> list of entity_ids
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.target_countries: Set[str] = set()

    def _extract_keys(
        self,
        name: str,
        addr: str,
        pin: str,
        primary_num: str,
        country_df: Dict[str, int]
    ) -> List[str]:
        keys = []
        tokens = [t for t in name.split() if len(t) >= 4 and t.isalpha()]
        rare_tokens = [t for t in tokens if country_df.get(t, 999999) <= self.max_df]
        
        # Primary number or extracted numbers
        nums = set()
        if primary_num and primary_num.isdigit():
            nums.add(primary_num)
        for n in NUM_RE.findall(addr)[:2]:
            nums.add(n)

        # 1. Composite: num + rare_token
        for r_tok in rare_tokens[:2]:
            for n in nums:
                keys.append(f"NUM_{n}#{r_tok}")

        # 2. Composite: pin + rare_token
        if pin and len(pin) >= 4:
            for r_tok in rare_tokens[:2]:
                keys.append(f"PIN_{pin}#{r_tok}")

        return keys

    def fit(self, target_df: pd.DataFrame) -> 'NameNumericHybridBlocker':
        t0 = time.time()
        self.index.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values
        names = target_df['name_clean'].fillna('').astype(str).values
        addrs = target_df['addr_clean'].fillna('').astype(str).values if 'addr_clean' in target_df.columns else [''] * len(target_df)
        pins = target_df['addr_pin'].fillna('').astype(str).values if 'addr_pin' in target_df.columns else [''] * len(target_df)
        nums = target_df['addr_primary_num'].fillna('').astype(str).values if 'addr_primary_num' in target_df.columns else [''] * len(target_df)

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        # Compute token DF
        country_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for idx, name in enumerate(names):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            for t in set(name.split()):
                if len(t) >= 4 and t.isalpha():
                    country_df[c][t] += 1
                    country_df['__GLOBAL__'][t] += 1

        # Build index
        for idx, eid in enumerate(eids):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            keys = self._extract_keys(names[idx], addrs[idx], pins[idx], nums[idx], country_df[c])
            for k in keys:
                self.index[c][k].append(eid)
                self.index['__GLOBAL__'][k].append(eid)

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)
        eids = query_df['entity_id'].values
        names = query_df['name_clean'].fillna('').astype(str).values
        addrs = query_df['addr_clean'].fillna('').astype(str).values if 'addr_clean' in query_df.columns else [''] * len(query_df)
        pins = query_df['addr_pin'].fillna('').astype(str).values if 'addr_pin' in query_df.columns else [''] * len(query_df)
        nums = query_df['addr_primary_num'].fillna('').astype(str).values if 'addr_primary_num' in query_df.columns else [''] * len(query_df)

        # Build dummy DF for query token checking (assume query tokens are informative if len >= 4)
        dummy_df: Dict[str, int] = defaultdict(int)

        for idx, q_eid in enumerate(eids):
            c_val = countries[idx] if countries[idx] else '__GLOBAL__'
            c_key = c_val if c_val in self.target_countries else '__GLOBAL__'

            keys = self._extract_keys(names[idx], addrs[idx], pins[idx], nums[idx], dummy_df)
            cand_set = set()

            for k in keys:
                postings = self.index.get(c_key, {}).get(k, [])
                if 0 < len(postings) <= self.max_bucket_size:
                    cand_set.update(postings)

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates
