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

    def prune(self, s1_df: pd.DataFrame, target_df: pd.DataFrame, max_cands: int = 50, top_k: int = 20, quota_per_strategy: int = 4):
        """
        Quota-based pruning: 
        1. Keep top-N candidates from EACH strategy based on coarse score.
        2. Fill remaining slots up to top_k with the highest overall coarse-score candidates.
        """
        print(f"Applying quota-based pruning (max_cands={max_cands}, top_k={top_k}, quota={quota_per_strategy})...")
        s1_data = {row['entity_id']: row for _, row in s1_df.iterrows()}
        target_data = {row['entity_id']: row for _, row in target_df.iterrows()}

        for s1_id, cand_map in self.provenance.items():
            if len(cand_map) <= max_cands:
                continue
            
            s1_row = s1_data.get(s1_id)
            if s1_row is None:
                continue
                
            s1_nt = set(s1_row.get('name_informative_tokens', [])) if isinstance(s1_row.get('name_informative_tokens'), list) else set()
            s1_at = set(s1_row.get('addr_tokens', [])) if isinstance(s1_row.get('addr_tokens'), list) else set()

            # 1. Score all candidates
            c_scores = {}
            for c_id in cand_map.keys():
                c_row = target_data.get(c_id)
                if c_row is None:
                    c_scores[c_id] = 0.0
                    continue
                
                c_nt = set(c_row.get('name_informative_tokens', [])) if isinstance(c_row.get('name_informative_tokens'), list) else set()
                c_at = set(c_row.get('addr_tokens', [])) if isinstance(c_row.get('addr_tokens'), list) else set()
                
                nu = len(s1_nt | c_nt)
                ns = len(s1_nt & c_nt) / nu if nu > 0 else 0.0
                
                au = len(s1_at | c_at)
                ast = len(s1_at & c_at) / au if au > 0 else 0.0
                
                c_scores[c_id] = ns + (ast * 0.2)
                
            # 2. Group by strategy
            strategy_groups = defaultdict(list)
            for c_id, strats in cand_map.items():
                score = c_scores[c_id]
                for strat in strats:
                    strategy_groups[strat].append((score, c_id))
                    
            # 3. Select top candidates per strategy
            selected_c_ids = set()
            for strat, scored_list in strategy_groups.items():
                scored_list.sort(key=lambda x: x[0], reverse=True)
                for _, c_id in scored_list[:quota_per_strategy]:
                    selected_c_ids.add(c_id)
                    
            # 4. Fill remaining slots if we haven't reached top_k
            if len(selected_c_ids) < top_k:
                remaining_cands = [(score, c_id) for c_id, score in c_scores.items() if c_id not in selected_c_ids]
                remaining_cands.sort(key=lambda x: x[0], reverse=True)
                needed = top_k - len(selected_c_ids)
                for _, c_id in remaining_cands[:needed]:
                    selected_c_ids.add(c_id)
                    
            # 5. Remove dropped candidates from provenance
            dropped = set(cand_map.keys()) - selected_c_ids
            for c_id in dropped:
                del cand_map[c_id]

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
