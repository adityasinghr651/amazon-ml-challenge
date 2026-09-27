"""
evaluation.py - Rigorous Blocking Evaluation & Diagnostics Harness

Calculates both:
  A. Pair-level blocking recall (True positive pairs / Total true positive pairs)
  B. Full entity recovery (S1 entities with 100% true matches recovered / Total non-singleton S1 entities)
  C. Distributional statistics (Mean, Median, P90, P95, P99, P99.9, Max)
  D. Candidate reduction ratio against theoretical Cartesian product
"""
from typing import Dict, List, Set, Any, Union, Optional
import numpy as np
import pandas as pd
import time

def evaluate_candidate_pairs(
    candidates: Union[Dict[str, Set[str]], pd.DataFrame],
    ground_truth: Union[Dict[str, Set[str]], pd.DataFrame],
    total_target_records: int = 0
) -> Dict[str, Any]:
    """
    Evaluates candidate generation performance against ground truth.

    Args:
        candidates: Either a dict {s1_id: set(candidate_ids)} or a DataFrame
                    with ['source1_entity_id', 'candidate_entity_ids']
        ground_truth: Either a dict {s1_id: set(true_match_ids)} or a DataFrame
                      with ['source1_entity_id', 'matched_entity_ids']
        total_target_records: Total count of (S2 + S3) records in the search space.

    Returns:
        dict containing comprehensive blocking metrics.
    """
    # 1. Parse candidates into dict
    cand_dict: Dict[str, Set[str]] = {}
    if isinstance(candidates, pd.DataFrame):
        for _, row in candidates.iterrows():
            c_str = str(row.get('candidate_entity_ids', ''))
            c_list = [c.strip() for c in c_str.split(',') if c.strip()]
            cand_dict[str(row['source1_entity_id'])] = set(c_list)
    elif isinstance(candidates, dict):
        for k, v in candidates.items():
            cand_dict[str(k)] = set(v) if isinstance(v, (set, list)) else set()
    else:
        raise TypeError(f"Unsupported type for candidates: {type(candidates)}")

    # 2. Parse ground truth into dict
    gt_dict: Dict[str, Set[str]] = {}
    if isinstance(ground_truth, pd.DataFrame):
        for _, row in ground_truth.iterrows():
            m_str = str(row.get('matched_entity_ids', ''))
            m_list = [m.strip() for m in m_str.split(',') if m.strip()]
            gt_dict[str(row['source1_entity_id'])] = set(m_list)
    elif isinstance(ground_truth, dict):
        for k, v in ground_truth.items():
            gt_dict[str(k)] = set(v) if isinstance(v, (set, list)) else set()
    else:
        raise TypeError(f"Unsupported type for ground_truth: {type(ground_truth)}")

    # 3. Compute Metrics
    total_gt_pairs = 0
    recovered_gt_pairs = 0
    non_singleton_entities = 0
    fully_recovered_entities = 0
    singleton_entities = 0
    zero_cand_singletons = 0
    zero_cand_non_singletons = 0

    all_s1_ids = set(cand_dict.keys()) | set(gt_dict.keys())
    cand_counts: List[int] = []

    for s1_id in all_s1_ids:
        cands = cand_dict.get(s1_id, set())
        true_matches = gt_dict.get(s1_id, set())
        n_cands = len(cands)
        cand_counts.append(n_cands)

        is_singleton = len(true_matches) == 0

        if is_singleton:
            singleton_entities += 1
            if n_cands == 0:
                zero_cand_singletons += 1
        else:
            non_singleton_entities += 1
            n_true = len(true_matches)
            total_gt_pairs += n_true

            n_recovered = len(true_matches & cands)
            recovered_gt_pairs += n_recovered

            if n_recovered == n_true:
                fully_recovered_entities += 1
            if n_cands == 0:
                zero_cand_non_singletons += 1

    # Pair-level recall
    pair_recall = (recovered_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 1.0

    # Full entity recovery
    entity_recovery = (fully_recovered_entities / non_singleton_entities) if non_singleton_entities > 0 else 1.0

    # Candidate volume statistics
    total_candidates = sum(cand_counts)
    n_evaluated_entities = len(cand_counts)
    avg_candidates = float(np.mean(cand_counts)) if cand_counts else 0.0
    median_candidates = float(np.median(cand_counts)) if cand_counts else 0.0
    p90_candidates = float(np.percentile(cand_counts, 90)) if cand_counts else 0.0
    p95_candidates = float(np.percentile(cand_counts, 95)) if cand_counts else 0.0
    p99_candidates = float(np.percentile(cand_counts, 99)) if cand_counts else 0.0
    p99_9_candidates = float(np.percentile(cand_counts, 99.9)) if cand_counts else 0.0
    max_candidates = int(np.max(cand_counts)) if cand_counts else 0

    # Reduction ratio
    reduction_ratio = 0.0
    if total_target_records > 0 and n_evaluated_entities > 0:
        theoretical_pairs = n_evaluated_entities * total_target_records
        reduction_ratio = 1.0 - (total_candidates / theoretical_pairs)
    elif total_target_records > 0:
        reduction_ratio = 1.0 - (avg_candidates / total_target_records)

    return {
        "pair_recall": pair_recall,
        "recovered_pairs": recovered_gt_pairs,
        "total_true_pairs": total_gt_pairs,
        "entity_recovery": entity_recovery,
        "fully_recovered_entities": fully_recovered_entities,
        "total_non_singletons": non_singleton_entities,
        "total_singletons": singleton_entities,
        "zero_cand_singletons": zero_cand_singletons,
        "zero_cand_non_singletons": zero_cand_non_singletons,
        "total_candidates": total_candidates,
        "evaluated_entities": n_evaluated_entities,
        "avg_candidates": avg_candidates,
        "median_candidates": median_candidates,
        "p90_candidates": p90_candidates,
        "p95_candidates": p95_candidates,
        "p99_candidates": p99_candidates,
        "p99_9_candidates": p99_9_candidates,
        "max_candidates": max_candidates,
        "reduction_ratio": reduction_ratio
    }

def format_evaluation_summary(metrics: Dict[str, Any], title: str = "Blocking Evaluation Summary") -> str:
    """Formats metrics dictionary into a readable report."""
    lines = [
        f"============================================================",
        f" {title}",
        f"============================================================",
        f" Pair-Level Blocking Recall   : {metrics['pair_recall']*100:.2f}% ({metrics['recovered_pairs']:,} / {metrics['total_true_pairs']:,} pairs)",
        f" Full Entity Recovery Rate    : {metrics['entity_recovery']*100:.2f}% ({metrics['fully_recovered_entities']:,} / {metrics['total_non_singletons']:,} entities)",
        f" Total Candidate Pairs        : {metrics['total_candidates']:,}",
        f" Avg Candidates per S1 Entity : {metrics['avg_candidates']:.2f}",
        f" Median Candidates            : {metrics['median_candidates']:.0f}",
        f" P90 Candidates               : {metrics['p90_candidates']:.0f}",
        f" P95 Candidates               : {metrics['p95_candidates']:.0f}",
        f" P99 Candidates               : {metrics['p99_candidates']:.0f}",
        f" P99.9 Candidates             : {metrics['p99_9_candidates']:.0f}",
        f" Max Candidates               : {metrics['max_candidates']:,}",
        f" Search Space Reduction Ratio : {metrics['reduction_ratio']*100:.4f}%",
        f" Non-Singletons with 0 Cands  : {metrics['zero_cand_non_singletons']:,}",
        f" Singletons with 0 Cands      : {metrics['zero_cand_singletons']:,} / {metrics['total_singletons']:,}",
        f"============================================================"
    ]
    return "\n".join(lines)
