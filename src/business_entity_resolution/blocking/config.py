"""
config.py - Unified Configuration Engine for Modular Blocking Tournament
Provides structured dataclasses and configuration loaders for all blocking experiments.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import json
try:
    import yaml
except ImportError:
    yaml = None

@dataclass
class NameBlockConfig:
    enabled: bool = True
    use_raw_clean: bool = True       # A1: name_clean
    use_core_name: bool = True       # A2: name_core (legal suffix stripped)
    use_sorted_tokens: bool = True   # A3: name_sorted (alphabetically sorted informative tokens)

@dataclass
class AddressBlockConfig:
    enabled: bool = True
    use_pin_only: bool = False       # B1: addr_pin
    use_pin_primary_num: bool = True # B2: addr_pin + addr_primary_num
    use_pin_name_token: bool = False # B3: addr_pin + first informative name token
    max_bucket_size: int = 30        # Safeguard against common postal codes/numbers

ExactNameConfig = NameBlockConfig
AddressConfig = AddressBlockConfig

@dataclass
class RareTokenConfig:
    enabled: bool = True
    max_doc_freq: int = 200          # Absolute maximum document frequency across corpus
    min_token_len: int = 3           # Minimum token length to consider
    top_n_rare: int = 2              # Top N rarest tokens per entity to index/query
    max_bucket_size: int = 30        # Maximum records per token bucket

@dataclass
class RetrievalConfig:
    enabled: bool = True
    top_k: int = 20                  # Number of nearest neighbors to retrieve
    ngram_range: tuple = (2, 4)      # Character n-gram range
    max_df: float = 0.9              # Max document frequency for TF-IDF
    min_df: int = 2                  # Min document frequency for TF-IDF
    batch_size: int = 1000           # Query batch size for sparse matrix multiplications
    sim_threshold: float = 0.3       # Minimum cosine similarity threshold

@dataclass
class CountryConfig:
    partition_by_country: bool = True # Partition blocking by country
    enable_global_fallback: bool = True # Fallback for records with missing/unknown country

@dataclass
class PruningConfig:
    enabled: bool = True             # Enforce top-K pruning to prevent candidate explosion
    strategy: str = "jaccard_top_k"  # 'none', 'top_k', 'jaccard_top_k'
    max_candidates: int = 50
    prune_top_k: int = 20

@dataclass
class EvaluationConfig:
    compute_pair_recall: bool = True
    compute_entity_recall: bool = True
    compute_percentiles: bool = True
    track_provenance: bool = True

@dataclass
class BlockingConfig:
    name_block: NameBlockConfig = field(default_factory=NameBlockConfig)
    address_block: AddressBlockConfig = field(default_factory=AddressBlockConfig)
    rare_token_block: RareTokenConfig = field(default_factory=RareTokenConfig)
    retrieval_block: RetrievalConfig = field(default_factory=RetrievalConfig)
    country: CountryConfig = field(default_factory=CountryConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a plain dictionary."""
        def _as_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _as_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj
        return _as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BlockingConfig':
        """Constructs configuration from dictionary."""
        cfg = cls()
        if 'name_block' in d:
            cfg.name_block = NameBlockConfig(**d['name_block'])
        if 'address_block' in d:
            cfg.address_block = AddressBlockConfig(**d['address_block'])
        if 'rare_token_block' in d:
            cfg.rare_token_block = RareTokenConfig(**d['rare_token_block'])
        if 'retrieval_block' in d:
            retrieval_dict = dict(d['retrieval_block'])
            if 'ngram_range' in retrieval_dict and isinstance(retrieval_dict['ngram_range'], list):
                retrieval_dict['ngram_range'] = tuple(retrieval_dict['ngram_range'])
            cfg.retrieval_block = RetrievalConfig(**retrieval_dict)
        if 'country' in d:
            cfg.country = CountryConfig(**d['country'])
        if 'pruning' in d:
            cfg.pruning = PruningConfig(**d['pruning'])
        if 'evaluation' in d:
            cfg.evaluation = EvaluationConfig(**d['evaluation'])
        return cfg
