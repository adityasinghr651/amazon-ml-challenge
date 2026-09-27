"""
Phase 5 & 6: training.py
Model training: LogReg, RandomForest, LightGBM, and hard negative mining.

LightGBM hyperparameters are read from config.yaml.
Hard negative mining operates within the train split only.
"""

import numpy as np
import joblib
import rapidfuzz
from typing import Dict, Any, Optional, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logreg(X_train, y_train, balanced=False):
    """Train a Logistic Regression classifier."""
    class_weight = "balanced" if balanced else None
    model = LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_lgbm(X_train, y_train, cfg: Dict[str, Any]):
    """
    Train a LightGBM classifier using config.yaml hyperparameters.

    scale_pos_weight / is_unbalance is treated as tunable —
    validate against val macro F0.5, not hardcoded.
    """
    import lightgbm as lgb

    lgb_cfg = cfg.get("lightgbm", {})
    seed = cfg.get("random_state", 42)

    model = lgb.LGBMClassifier(
        n_estimators=lgb_cfg.get("n_estimators", 500),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        num_leaves=lgb_cfg.get("num_leaves", 63),
        max_depth=lgb_cfg.get("max_depth", -1),
        min_child_samples=lgb_cfg.get("min_child_samples", 20),
        subsample=lgb_cfg.get("subsample", 0.8),
        colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
        reg_alpha=lgb_cfg.get("reg_alpha", 0.1),
        reg_lambda=lgb_cfg.get("reg_lambda", 1.0),
        is_unbalance=lgb_cfg.get("is_unbalance", True),
        n_jobs=lgb_cfg.get("n_jobs", -1),
        verbose=lgb_cfg.get("verbose", -1),
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def mine_hard_negatives(
    candidates_df,
    records_db: Dict[str, Dict],
    ground_truth: Dict[str, List[str]],
    train_s1_ids: List[str],
    top_k_per_s1: int = 5,
    min_name_sim: float = 0.3,
) -> List[Tuple[str, str, int]]:
    """
    Mine hard negative examples from the candidate set (train split only).

    Hard negatives = candidates that are NOT true matches but have high
    name similarity to the S1 entity. These force the classifier to learn
    subtle distinctions rather than trivially easy ones.

    Parameters
    ----------
    candidates_df : DataFrame with [source1_entity_id, candidate_entity_ids].
    records_db : dict mapping entity_id -> record dict (normalized).
    ground_truth : dict mapping S1 ID -> list of true match IDs.
    train_s1_ids : list of S1 IDs in the train split ONLY.
    top_k_per_s1 : max hard negatives per S1 entity.
    min_name_sim : minimum name similarity to qualify as a "hard" negative.

    Returns
    -------
    List of (s1_id, candidate_id, label=0) triples.
    """
    train_set = set(train_s1_ids)
    hard_negs = []

    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        if s1_id not in train_set:
            continue

        cand_str = str(row.get("candidate_entity_ids", "")).strip()
        if not cand_str:
            continue

        true_matches = set(ground_truth.get(s1_id, []))
        cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]

        # Get S1 name for similarity comparison
        s1_rec = records_db.get(s1_id)
        if s1_rec is None:
            continue
        s1_name = str(s1_rec.get("name_clean", ""))
        if not s1_name:
            continue

        # Score non-matching candidates by name similarity
        scored_negs = []
        for c_id in cand_ids:
            if c_id in true_matches:
                continue  # skip true matches
            c_rec = records_db.get(c_id)
            if c_rec is None:
                continue
            c_name = str(c_rec.get("name_clean", ""))
            if not c_name:
                continue

            sim = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_name, c_name)
            if sim >= min_name_sim:
                scored_negs.append((sim, c_id))

        # Take top-k most similar non-matches
        scored_negs.sort(key=lambda x: x[0], reverse=True)
        for _, c_id in scored_negs[:top_k_per_s1]:
            hard_negs.append((s1_id, c_id, 0))

    return hard_negs


def save_model(model, filepath: str):
    """Serialize model to disk."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Load serialized model from disk."""
    return joblib.load(filepath)
