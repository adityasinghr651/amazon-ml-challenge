"""
Phase 6: decision.py
Threshold optimization, singleton handling, and multi-match decision logic.

Two-band decision:
  - Scores >= t_high → confident match
  - Scores < t_low  → confident non-match / singleton
  - t_low <= scores < t_high → borderline → default to non-match
    (since false merges cost 2× under F0.5)

Thresholds are tuned jointly on the val split, never on holdout.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def predict_matches(
    scores_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """
    Convert model probabilities to final matches using a single threshold.

    Parameters
    ----------
    scores_df : DataFrame with columns [source1_entity_id, candidate_entity_id, score].
    threshold : probability cutoff for match decision.

    Returns
    -------
    Dict mapping S1 ID -> list of matched S2/S3 IDs.
    """
    predictions: Dict[str, List[str]] = {}

    for s1_id in scores_df["source1_entity_id"].unique():
        predictions[s1_id] = []

    matches = scores_df[scores_df["score"] > threshold]
    for _, row in matches.iterrows():
        predictions[row["source1_entity_id"]].append(row["candidate_entity_id"])

    return predictions


def predict_matches_two_band(
    scores_df: pd.DataFrame,
    t_high: float = 0.70,
    t_low: float = 0.40,
) -> Dict[str, List[str]]:
    """
    Two-band decision logic:
      - score >= t_high → match
      - score < t_low   → non-match
      - t_low <= score < t_high → borderline → default NON-MATCH
        (precision-heavy F0.5: false merges cost 2× more than missed matches)

    Parameters
    ----------
    scores_df : DataFrame with [source1_entity_id, candidate_entity_id, score].
    t_high : upper threshold for confident matches.
    t_low : lower threshold for confident non-matches.

    Returns
    -------
    Dict mapping S1 ID -> list of matched S2/S3 IDs.
    """
    predictions: Dict[str, List[str]] = {}

    for s1_id in scores_df["source1_entity_id"].unique():
        predictions[s1_id] = []

    # Only scores at or above t_high qualify as matches
    confident_matches = scores_df[scores_df["score"] >= t_high]
    for _, row in confident_matches.iterrows():
        predictions[row["source1_entity_id"]].append(row["candidate_entity_id"])

    return predictions


def optimize_threshold(
    scores_df: pd.DataFrame,
    val_gt: Dict[str, List[str]],
    t_min: float = 0.30,
    t_max: float = 0.95,
    t_step: float = 0.025,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Grid search over single threshold, optimizing macro F0.5 on val split.

    Returns
    -------
    (best_threshold, best_f05, best_metrics_dict)
    """
    from src.business_entity_resolution.evaluation import calculate_metrics

    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    best_t = 0.5
    best_f05 = 0.0
    best_metrics = {}

    for t in thresholds:
        preds = predict_matches(scores_df, threshold=t)
        metrics = calculate_metrics(preds, val_gt)
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_t = float(t)
            best_metrics = metrics

    return best_t, best_f05, best_metrics


def optimize_two_band_threshold(
    scores_df: pd.DataFrame,
    val_gt: Dict[str, List[str]],
    t_high_range: Tuple[float, float, float] = (0.50, 0.95, 0.025),
    t_low_range: Tuple[float, float, float] = (0.20, 0.60, 0.025),
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Joint grid search over (t_high, t_low) optimizing macro F0.5 on val split.

    The two-band logic:
      score >= t_high → match, score < t_low → non-match,
      borderline → non-match (default to precision under F0.5).

    Constraint: t_low <= t_high enforced.

    Returns
    -------
    (best_t_high, best_t_low, best_f05, best_metrics)
    """
    from src.business_entity_resolution.evaluation import calculate_metrics

    t_highs = np.arange(t_high_range[0], t_high_range[1] + t_high_range[2] / 2, t_high_range[2])
    t_lows = np.arange(t_low_range[0], t_low_range[1] + t_low_range[2] / 2, t_low_range[2])

    best_th = 0.70
    best_tl = 0.40
    best_f05 = 0.0
    best_metrics = {}

    for th in t_highs:
        for tl in t_lows:
            if tl > th:
                continue  # constraint: t_low <= t_high

            preds = predict_matches_two_band(scores_df, t_high=th, t_low=tl)
            metrics = calculate_metrics(preds, val_gt)
            if metrics["macro_f05"] > best_f05:
                best_f05 = metrics["macro_f05"]
                best_th = float(th)
                best_tl = float(tl)
                best_metrics = metrics

    return best_th, best_tl, best_f05, best_metrics
