"""
address.py - Address Anchor Candidate Generation Engine

Supports modular address-based blocking keys:
  B1. postal_code (addr_pin)
  B2. postal_code + primary_number (addr_pin + addr_primary_num)
  B3. postal_code + first informative name token (addr_pin + name_prefix)

Includes configurable bucket-size safeguards (max_bucket_size), bucket distribution profiling,
and tracking of oversized buckets to prevent candidate explosions.
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import time

from src.business_entity_resolution.blocking.base import BaseBlocker
from src.business_entity_resolution.blocking.config import AddressBlockConfig

class AddressAnchorBlocker(BaseBlocker):
    def __init__(
        self,
        name: str = "AddressAnchorBlocker",
        config: Optional[AddressBlockConfig] = None,
        partition_by_country: bool = True
    ):
        super().__init__(name=name, config=config)
        self.cfg = config or AddressBlockConfig()
        self.partition_by_country = partition_by_country
        self.max_bucket_size = self.cfg.max_bucket_size

        # Inverted index: Dict[str, Dict[str, List[str]]]
        # key: country -> blocking_key -> [entity_ids]
        self.index_pin: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_pin_num: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_pin_token: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Fallback indexes
        self.index_loc_num_with_pin: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.index_loc_num_no_pin: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        self.target_countries: Set[str] = set()

        # Profiling stats
        self.bucket_sizes: Dict[str, List[int]] = defaultdict(list)
        self.oversized_keys: Dict[str, Set[str]] = defaultdict(set)

    def fit(self, target_df: pd.DataFrame) -> 'AddressAnchorBlocker':
        t0 = time.time()

        has_country = 'country_norm' in target_df.columns and self.partition_by_country
        countries = target_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(target_df)
        eids = target_df['entity_id'].values

        if has_country:
            self.target_countries = set(target_df['country_norm'].unique())

        pins = target_df['addr_pin'].values if 'addr_pin' in target_df.columns else None
        nums = target_df['addr_primary_num'].values if 'addr_primary_num' in target_df.columns else None
        inf_tokens = target_df['name_informative_tokens'].values if 'name_informative_tokens' in target_df.columns else None

        def get_locality(tokens):
            if not isinstance(tokens, (list, np.ndarray)): return ""
            for t in reversed(tokens):
                if isinstance(t, str) and len(t) >= 3 and not t.isdigit():
                    return t
            return ""

        for idx, eid in enumerate(eids):
            c = countries[idx] if countries[idx] else '__GLOBAL__'
            pin = pins[idx] if pins is not None else ""
            num = nums[idx] if nums is not None else ""
            toks = inf_tokens[idx] if inf_tokens is not None else []
            addr_toks = target_df['addr_tokens'].values[idx] if 'addr_tokens' in target_df.columns else []
            
            loc = get_locality(addr_toks)
            
            # Fallback B4: Locality + Primary Number (Only used when PIN is missing on at least one side)
            if self.cfg.use_pin_primary_num and num and loc:
                key_loc_num = f"{loc}_{num}"
                if pin:
                    self.index_loc_num_with_pin[c][key_loc_num].append(eid)
                    self.index_loc_num_with_pin['__GLOBAL__'][key_loc_num].append(eid)
                else:
                    self.index_loc_num_no_pin[c][key_loc_num].append(eid)
                    self.index_loc_num_no_pin['__GLOBAL__'][key_loc_num].append(eid)

            if not pin:
                continue

            # B1: Postal code alone
            if self.cfg.use_pin_only:
                self.index_pin[c][pin].append(eid)
                self.index_pin['__GLOBAL__'][pin].append(eid)

            # B2: Postal code + Primary number
            if self.cfg.use_pin_primary_num and num:
                key_pin_num = f"{pin}_{num}"
                self.index_pin_num[c][key_pin_num].append(eid)
                self.index_pin_num['__GLOBAL__'][key_pin_num].append(eid)

            # B3: Postal code + First informative name token
            if self.cfg.use_pin_name_token and inf_tokens is not None:
                toks = inf_tokens[idx]
                first_tok = toks[0] if isinstance(toks, list) and len(toks) > 0 else ""
                if first_tok:
                    key_pin_tok = f"{pin}_{first_tok}"
                    self.index_pin_token[c][key_pin_tok].append(eid)
                    self.index_pin_token['__GLOBAL__'][key_pin_tok].append(eid)

        # Enforce max_bucket_size safeguard & profile distributions
        indexes_to_profile = [
            ("B1_PIN", self.index_pin), 
            ("B2_PIN_NUM", self.index_pin_num), 
            ("B3_PIN_TOKEN", self.index_pin_token),
            ("B4_LOC_NUM_WITH_PIN", self.index_loc_num_with_pin),
            ("B4_LOC_NUM_NO_PIN", self.index_loc_num_no_pin)
        ]
        for strat_name, idx_dict in indexes_to_profile:
            for c, key_map in idx_dict.items():
                for k, postings in list(key_map.items()):
                    b_size = len(postings)
                    self.bucket_sizes[strat_name].append(b_size)
                    if b_size > self.max_bucket_size:
                        self.oversized_buckets_encountered += 1
                        self.oversized_keys[strat_name].add(k)

        self.index_time = time.time() - t0
        return self

    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        t0 = time.time()
        candidates: Dict[str, Set[str]] = defaultdict(set)

        has_country = 'country_norm' in query_df.columns and self.partition_by_country
        countries = query_df['country_norm'].values if has_country else ['__GLOBAL__'] * len(query_df)
        eids = query_df['entity_id'].values

        pins = query_df['addr_pin'].values if 'addr_pin' in query_df.columns else None
        nums = query_df['addr_primary_num'].values if 'addr_primary_num' in query_df.columns else None
        inf_tokens = query_df['name_informative_tokens'].values if 'name_informative_tokens' in query_df.columns else None

        def get_locality(tokens):
            if not isinstance(tokens, (list, np.ndarray)): return ""
            for t in reversed(tokens):
                if isinstance(t, str) and len(t) >= 3 and not t.isdigit():
                    return t
            return ""

        for idx, q_eid in enumerate(eids):
            c_val = countries[idx] if countries[idx] else '__GLOBAL__'
            c_key = c_val if c_val in self.target_countries else '__GLOBAL__'
            pin = pins[idx] if pins is not None else ""
            num = nums[idx] if nums is not None else ""
            toks = inf_tokens[idx] if inf_tokens is not None else []
            addr_toks = query_df['addr_tokens'].values[idx] if 'addr_tokens' in query_df.columns else []
            loc = get_locality(addr_toks)

            cand_set = set()
            
            if not pin:
                # Query has no PIN. Use fallback if possible.
                if self.cfg.use_pin_primary_num and num and loc:
                    key = f"{loc}_{num}"
                    # Match targets that also have NO PIN
                    postings_no_pin = self.index_loc_num_no_pin.get(c_key, {}).get(key, [])
                    if 0 < len(postings_no_pin) <= self.max_bucket_size:
                        cand_set.update(postings_no_pin)
                    # Match targets that HAVE A PIN (query is missing it)
                    postings_with_pin = self.index_loc_num_with_pin.get(c_key, {}).get(key, [])
                    if 0 < len(postings_with_pin) <= self.max_bucket_size:
                        cand_set.update(postings_with_pin)
                candidates[q_eid] = cand_set
                self.total_candidates_generated += len(cand_set)
                continue

            cand_set = set()

            # B1: Match PIN alone (if enabled and within bucket cap)
            if self.cfg.use_pin_only:
                pin_postings = self.index_pin.get(c_key, {}).get(pin, [])
                if 0 < len(pin_postings) <= self.max_bucket_size:
                    cand_set.update(pin_postings)

            # B2: Match PIN + Primary Number (Primary Path)
            if self.cfg.use_pin_primary_num and num:
                key = f"{pin}_{num}"
                pin_num_postings = self.index_pin_num.get(c_key, {}).get(key, [])
                if 0 < len(pin_num_postings) <= self.max_bucket_size:
                    cand_set.update(pin_num_postings)
                
                # Fallback: Query HAS pin, but Target might be missing it.
                if loc:
                    key_loc = f"{loc}_{num}"
                    postings_no_pin = self.index_loc_num_no_pin.get(c_key, {}).get(key_loc, [])
                    if 0 < len(postings_no_pin) <= self.max_bucket_size:
                        cand_set.update(postings_no_pin)

            # B3: Match PIN + First Informative Name Token
            if self.cfg.use_pin_name_token and toks:
                first_tok = toks[0] if isinstance(toks, list) and len(toks) > 0 else ""
                if first_tok:
                    key = f"{pin}_{first_tok}"
                    pin_tok_postings = self.index_pin_token.get(c_key, {}).get(key, [])
                    if 0 < len(pin_tok_postings) <= self.max_bucket_size:
                        cand_set.update(pin_tok_postings)

            candidates[q_eid] = cand_set
            self.total_candidates_generated += len(cand_set)

        self.query_time = time.time() - t0
        return candidates

    def get_bucket_diagnostics(self) -> Dict[str, Any]:
        """Provides statistics on bucket sizes and oversized buckets."""
        diag = {}
        for strat, sizes in self.bucket_sizes.items():
            if sizes:
                diag[strat] = {
                    "total_buckets": len(sizes),
                    "mean_size": float(np.mean(sizes)),
                    "median_size": float(np.median(sizes)),
                    "p95_size": float(np.percentile(sizes, 95)),
                    "max_size": int(np.max(sizes)),
                    "oversized_count": len(self.oversized_keys[strat])
                }
        return diag
