"""
acronym.py - Acronym & Initial-Character Candidate Blocker

Forensic analysis in Phase G identified abbreviation & acronym failures
(e.g., "State Bank of India" vs "SBI", "Tata Consultancy Services" vs "TCS").

This blocker:
  1. Computes acronym initials from informative business tokens
  2. Pairs acronyms with PIN/postal code anchors (acronym + pin) to eliminate
     global 3-4 letter acronym collisions
  3. Matches single-word acronym names against expanded multi-token organizations
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import pandas as pd
import time

from src.business_entity_resolution.blocking.base import BaseBlocker
from src.business_entity_resolution.normalization import GENERIC_BUSINESS_WORDS

class AcronymBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "AcronymBlocker",
        min_acronym_len: int = 3,
        max_acronym_len: int = 6,
        partition_by_country: bool = True
    ):
        super().__init__(name=name)
        self.min_acronym_len = min_acronym_len
        self.max_acronym_len = max_acronym_len
        self.partition_by_country = partition_by_country

        # Inverted index: country -> (acronym_key) -> list of entity_ids
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.target_countries: Set[str] = set()

    def _extract_acronym(self, name: str, tokens_val: Any) -> Optional[str]:
        """Extracts acronym initials if record has >= 2 non-generic tokens."""
        if isinstance(tokens_val, list) and len(tokens_val) >= 2:
            toks = [t for t in tokens_val if t.lower() not in GENERIC_BUSINESS_WORDS]
        elif isinstance(tokens_val, str) and tokens_val.strip():
            toks = [t for t in tokens_val.split() if t.lower() not in GENERIC_BUSINESS_WORDS]
        else:
            toks = [t for t in str(name).split() if t.lower() not in GENERIC_BUSINESS_WORDS]

        if self.min_acronym_len <= len(toks) <= self.max_acronym_len:
            acronym = "".join([t[0].lower() for t in toks if t and t[0].isalnum()])
            if len(acronym) >= self.min_acronym_len:
                return acronym
        return None

    def fit(self, target_df: pd.DataFrame) -> 'AcronymBlocker':
        t0 = time.time()
        self.index.clear()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values
        names = target_df['name_clean'].fillna('').astype(str).values
        pins = target_df['addr_pin'].values if 'addr_pin' in target_df.columns else [''] * len(target_df)
        tok_col = 'name_informative_tokens' if 'name_informative_tokens' in target_df.columns else 'name_clean'
        token_series = target_df[tok_col].values

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        for idx, eid in enumerate(eids):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            name = names[idx]
            pin = pins[idx]

            # Case A: Multi-token name that produces an acronym
            acronym = self._extract_acronym(name, token_series[idx])
            if acronym and pin:
                key = f"{acronym}#{pin}"
                self.index[c][key].append(eid)
                self.index['__GLOBAL__'][key].append(eid)

            # Case B: Record name itself IS a short single-word acronym
            single_word = name.strip()
            if self.min_acronym_len <= len(single_word) <= self.max_acronym_len and ' ' not in single_word and pin:
                key = f"{single_word}#{pin}"
                self.index[c][key].append(eid)
                self.index['__GLOBAL__'][key].append(eid)

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)
        eids = query_df['entity_id'].values
        names = query_df['name_clean'].fillna('').astype(str).values
        pins = query_df['addr_pin'].values if 'addr_pin' in query_df.columns else [''] * len(query_df)
        tok_col = 'name_informative_tokens' if 'name_informative_tokens' in query_df.columns else 'name_clean'
        token_series = query_df[tok_col].values

        for idx, q_eid in enumerate(eids):
            c_val = countries[idx] if countries[idx] else '__GLOBAL__'
            c_key = c_val if c_val in self.target_countries else '__GLOBAL__'
            name = names[idx]
            pin = pins[idx]

            if not pin:
                candidates[q_eid] = set()
                continue

            cand_set = set()
            acronym = self._extract_acronym(name, token_series[idx])
            if acronym:
                key = f"{acronym}#{pin}"
                postings = self.index.get(c_key, {}).get(key, [])
                cand_set.update(postings)

            # Also check if query name is single-word acronym
            single_word = name.strip()
            if self.min_acronym_len <= len(single_word) <= self.max_acronym_len and ' ' not in single_word:
                key = f"{single_word}#{pin}"
                postings = self.index.get(c_key, {}).get(key, [])
                cand_set.update(postings)

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates
