"""
preprocessing.py - Data preparation: ground-truth parsing, labeling, entity-grouped splits.

All labeling and split logic lives here so it can be reused by any training
or evaluation script without duplication.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from sklearn.model_selection import train_test_split


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Parse ground truth DataFrame into a dictionary.
    Returns: {source1_entity_id: [matched_s2s3_id, ...]}
    Singletons map to an empty list.
    """
    gt_dict: Dict[str, List[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched = str(row.get("matched_entity_ids", "")).strip()
        if matched:
            gt_dict[s1_id] = [m.strip() for m in matched.split(",") if m.strip()]
        else:
            gt_dict[s1_id] = []
    return gt_dict


def build_labeled_pairs(
    candidates_df: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Anti-join blocking candidates against ground-truth to produce labeled pairs.

    Parameters
    ----------
    candidates_df : DataFrame with columns [source1_entity_id, candidate_entity_ids]
        Output of blocking.generate_candidates().
    ground_truth : dict mapping S1 ID -> list of true S2/S3 match IDs.

    Returns
    -------
    DataFrame with columns [source1_entity_id, candidate_entity_id, label]
        label=1 if candidate is a true match, label=0 otherwise.
    """
    rows = []
    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        cand_str = str(row.get("candidate_entity_ids", "")).strip()
        if not cand_str:
            continue
        cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
        true_matches = set(ground_truth.get(s1_id, []))
        for c_id in cand_ids:
            label = 1 if c_id in true_matches else 0
            rows.append({"source1_entity_id": s1_id, "candidate_entity_id": c_id, "label": label})

    return pd.DataFrame(rows)


def entity_grouped_split(
    s1_ids: np.ndarray,
    ground_truth: Dict[str, List[str]],
    val_frac: float = 0.15,
    holdout_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split S1 entity IDs into train / val / holdout groups.

    The split is at the S1-entity level (not pair level) so that all
    associated ground-truth matches stay together — preventing label leakage.

    Parameters
    ----------
    s1_ids : array of S1 entity IDs.
    ground_truth : dict mapping S1 ID -> list of true match IDs.
    val_frac : fraction of S1 entities for validation.
    holdout_frac : fraction of S1 entities for holdout.
    seed : random state for reproducibility.

    Returns
    -------
    (train_ids, val_ids, holdout_ids) — each a list of S1 entity IDs.
    """
    all_ids = list(set(s1_ids))

    # First split: separate holdout
    remaining_ids, holdout_ids = train_test_split(
        all_ids,
        test_size=holdout_frac,
        random_state=seed,
    )

    # Second split: separate val from train
    val_relative_frac = val_frac / (1.0 - holdout_frac)
    train_ids, val_ids = train_test_split(
        remaining_ids,
        test_size=val_relative_frac,
        random_state=seed,
    )

    # Report statistics
    def _pct_singleton(ids):
        singletons = sum(1 for x in ids if not ground_truth.get(str(x), []))
        return (singletons / len(ids) * 100) if len(ids) > 0 else 0.0

    print(f"Entity-grouped split (seed={seed}):")
    print(f"  Train:   {len(train_ids):>8,} S1 entities ({_pct_singleton(train_ids):.1f}% singletons)")
    print(f"  Val:     {len(val_ids):>8,} S1 entities ({_pct_singleton(val_ids):.1f}% singletons)")
    print(f"  Holdout: {len(holdout_ids):>8,} S1 entities ({_pct_singleton(holdout_ids):.1f}% singletons)")

    return (
        [str(x) for x in train_ids],
        [str(x) for x in val_ids],
        [str(x) for x in holdout_ids],
    )


def sample_s1_grouped(
    s1_df: pd.DataFrame,
    frac: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample a fraction of S1 entities (by entity ID, not by row).
    Returns the subset of s1_df containing the sampled entity IDs.
    """
    all_ids = s1_df["entity_id"].unique()
    n_sample = max(1, int(len(all_ids) * frac))
    rng = np.random.RandomState(seed)
    sampled_ids = set(rng.choice(all_ids, size=n_sample, replace=False))
    return s1_df[s1_df["entity_id"].isin(sampled_ids)]
