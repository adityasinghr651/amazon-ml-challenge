"""
base.py - Base Blocker Interface for Modular Blocking Tournament
All blockers implement this interface for consistent benchmarking, provenance tracking,
and modular composition.
"""
from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Any, Optional
import pandas as pd
import time

class BaseBlocker(ABC):
    """
    Abstract Base Class for all candidate generation blockers.
    """
    def __init__(self, name: str, config: Optional[Any] = None):
        self.name = name
        self.config = config
        self.index_time = 0.0
        self.query_time = 0.0
        self.total_candidates_generated = 0
        self.oversized_buckets_encountered = 0

    @abstractmethod
    def fit(self, target_df: pd.DataFrame) -> 'BaseBlocker':
        """
        Builds internal index structures over target records (S2 and S3).
        """
        pass

    @abstractmethod
    def block(self, query_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Generates candidate entity IDs for query records (S1).

        Returns:
            Dict[s1_entity_id, Set[candidate_entity_ids]]
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime and indexing diagnostics."""
        return {
            "name": self.name,
            "index_time_s": self.index_time,
            "query_time_s": self.query_time,
            "total_candidates": self.total_candidates_generated,
            "oversized_buckets": self.oversized_buckets_encountered
        }
