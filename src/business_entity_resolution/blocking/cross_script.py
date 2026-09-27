"""
cross_script.py - Country-Neutral Transliteration & Cross-Script Blocker
Bridges Unicode script divides (e.g. Latin English vs Devanagari, Bengali, Telugu, Tamil, Gujarati)
using offline phonetic transliteration (anyascii). Country-neutral and open-set.
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import pandas as pd
import time
from anyascii import anyascii

from src.business_entity_resolution.blocking.base import BaseBlocker

def is_non_ascii(text: str) -> bool:
    """Checks if string contains non-ASCII characters (e.g. Indic, Cyrillic, accented)."""
    return any(ord(c) >= 128 for c in str(text))

class TransliterationBlocker(BaseBlocker):
    """
    Indexes phonetic transliterations of names to enable cross-script matching.
    Country-neutral: operates on Unicode text regardless of geographical country.
    """
    def __init__(
        self,
        name: str = "TransliterationBlocker",
        max_df: int = 150,
        min_token_len: int = 4,
        max_bucket_size: int = 30,
        partition_by_country: bool = True
    ):
        super().__init__(name=name)
        self.max_df = max_df
        self.min_token_len = min_token_len
        self.max_bucket_size = max_bucket_size
        self.partition_by_country = partition_by_country

        # country -> transliterated_token -> list of entity_ids
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> transliterated_core_name -> list of entity_ids
        self.exact_translit_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.target_countries: Set[str] = set()

    def fit(self, target_df: pd.DataFrame) -> 'TransliterationBlocker':
        t0 = time.time()
        self.index.clear()
        self.exact_translit_index.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values
        names = target_df['name_clean'].fillna('').astype(str).values

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        # Track token frequencies for transliterated tokens
        country_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        target_translits: List[str] = []

        for idx, name in enumerate(names):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            if is_non_ascii(name):
                t_trans = anyascii(name).lower().strip()
            else:
                t_trans = ""
            target_translits.append(t_trans)

            if t_trans:
                toks = set(t_trans.split())
                for t in toks:
                    if len(t) >= self.min_token_len and t.isalpha():
                        country_df[c][t] += 1
                        country_df['__GLOBAL__'][t] += 1

        # Build indexes
        for idx, eid in enumerate(eids):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            t_trans = target_translits[idx]
            if not t_trans:
                continue

            # Exact transliterated full name
            self.exact_translit_index[c][t_trans].append(eid)
            self.exact_translit_index['__GLOBAL__'][t_trans].append(eid)

            # Rare transliterated tokens
            for t in set(t_trans.split()):
                if len(t) >= self.min_token_len and t.isalpha():
                    if country_df[c][t] <= self.max_df:
                        self.index[c][t].append(eid)
                    if country_df['__GLOBAL__'][t] <= self.max_df:
                        self.index['__GLOBAL__'][t].append(eid)

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

            q_name = names[idx].lower().strip()
            # If query itself has non-ascii, transliterate it too
            q_trans = anyascii(q_name).lower().strip() if is_non_ascii(q_name) else q_name

            cand_set = set()

            # 1. Exact transliterated match
            if q_trans:
                exact_hits = self.exact_translit_index.get(c_key, {}).get(q_trans, [])
                if 0 < len(exact_hits) <= self.max_bucket_size:
                    cand_set.update(exact_hits)

            # 2. Transliterated token match
            q_toks = set(q_trans.split())
            for t in q_toks:
                if len(t) >= self.min_token_len and t.isalpha():
                    hits = self.index.get(c_key, {}).get(t, [])
                    if 0 < len(hits) <= self.max_bucket_size:
                        cand_set.update(hits)

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates
