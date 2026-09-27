"""
union.py - Candidate Union, Deduplication & Provenance Engine

Aggregates candidates from multiple blockers:
  - Preserves provenance: which strategy/blocker proposed each pair
  - Merges duplicate candidate pairs
  - Computes script mismatch diagnostics (Latin, Devanagari, Tamil)
  - Exports dual outputs:
      1. Official candidate_pairs.tsv (source1_entity_id \t candidate_entity_ids)
      2. Internal candidate_pairs_diagnostics.tsv (source1_entity_id \t candidate_entity_id \t strategies \t script_mismatch)
"""
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict
import unicodedata
import pandas as pd
import os

def detect_primary_script(text: str) -> str:
    """Detects Unicode script (Latin, Devanagari, Tamil, etc.)."""
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, '')
            if 'DEVANAGARI' in name:
                return 'Devanagari'
            if 'TAMIL' in name:
                return 'Tamil'
            if 'LATIN' in name:
                return 'Latin'
    return 'Latin'

class CandidateUnionEngine:
    def __init__(self):
        # Provenance: Dict[s1_id, Dict[cand_id, Set[strategy_name]]]
        self.provenance: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def add_blocker_results(self, blocker_name: str, candidates: Dict[str, Set[str]]):
        """Adds candidate set from a specific blocker, tagging provenance."""
        for s1_id, cand_set in candidates.items():
            for c_id in cand_set:
                self.provenance[s1_id][c_id].add(blocker_name)

    def get_candidate_dict(self) -> Dict[str, Set[str]]:
        """Returns unioned candidates as {s1_id: set(candidate_ids)}."""
        return {s1_id: set(cand_map.keys()) for s1_id, cand_map in self.provenance.items()}

    def to_official_dataframe(self, all_s1_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Builds official DataFrame adhering strictly to competition schema:
        Columns: source1_entity_id \t candidate_entity_ids
        Every S1 entity appears exactly once.
        """
        rows = []
        target_ids = all_s1_ids if all_s1_ids is not None else list(self.provenance.keys())
        for s1_id in target_ids:
            cands = self.provenance.get(s1_id, {})
            c_str = ",".join(sorted(list(cands.keys()))) if cands else ""
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": c_str
            })
        return pd.DataFrame(rows)

    def to_diagnostics_dataframe(
        self,
        s1_names: Optional[Dict[str, str]] = None,
        cand_names: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Builds detailed internal diagnostics DataFrame:
        Columns: source1_entity_id, candidate_entity_id, strategies, script_mismatch
        """
        rows = []
        for s1_id, cand_map in self.provenance.items():
            s1_script = detect_primary_script(s1_names.get(s1_id, '')) if s1_names else 'Latin'
            for c_id, strats in cand_map.items():
                c_script = detect_primary_script(cand_names.get(c_id, '')) if cand_names else 'Latin'
                script_mismatch = (s1_script != c_script)

                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": c_id,
                    "strategies": ",".join(sorted(list(strats))),
                    "script_mismatch": script_mismatch
                })
        return pd.DataFrame(rows)

    def export_outputs(
        self,
        official_path: str = "output/candidate_pairs.tsv",
        diagnostics_path: str = "output/candidate_pairs_diagnostics.tsv",
        all_s1_ids: Optional[List[str]] = None,
        s1_names: Optional[Dict[str, str]] = None,
        cand_names: Optional[Dict[str, str]] = None
    ):
        """Exports both official competition TSV and internal diagnostic TSV."""
        os.makedirs(os.path.dirname(official_path), exist_ok=True)
        os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)

        df_official = self.to_official_dataframe(all_s1_ids=all_s1_ids)
        df_official.to_csv(official_path, sep="\t", index=False)

        df_diag = self.to_diagnostics_dataframe(s1_names=s1_names, cand_names=cand_names)
        df_diag.to_csv(diagnostics_path, sep="\t", index=False)
