"""
run_phase5.py — End-to-end training pipeline.

1. Load data (10% S1 sample for speed, full S2/S3 for blocking)
2. Normalize → Block → Label (using preprocessing.build_labeled_pairs)
3. Entity-grouped train/val/holdout split
4. Extract features → Train models (LR, RF, LightGBM)
5. Threshold search on val split
6. Report metrics
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.io import load_train_data
from src.business_entity_resolution.normalization import process_dataframe
from src.business_entity_resolution.blocking import generate_candidates
from src.business_entity_resolution.preprocessing import (
    parse_ground_truth,
    build_labeled_pairs,
    entity_grouped_split,
    sample_s1_grouped,
)
from src.business_entity_resolution.features import extract_features, extract_features_batch
from src.business_entity_resolution.training import train_logreg, train_rf, train_lgbm, save_model
from src.business_entity_resolution.evaluation import calculate_metrics
from src.business_entity_resolution.decision import predict_matches, optimize_threshold
from src.business_entity_resolution.utils import load_config, get_blocking_config, log_experiment

os.makedirs("output/train", exist_ok=True)
os.makedirs("experiments/models", exist_ok=True)


def main():
    cfg = load_config()
    seed = cfg["random_state"]
    blk_config = get_blocking_config(cfg)

    # ===================================================================
    # 1. LOAD DATA
    # ===================================================================
    print("=" * 70)
    print("PHASE 5 — Training Pipeline (config-driven, no true-match injection)")
    print("=" * 70)

    print("\nLoading data...")
    s1_full, s2, s3, gt_df = load_train_data()
    ground_truth = parse_ground_truth(gt_df)

    # 10% entity-grouped sample of S1 for tractable runtime
    print("Sampling 10% of S1 (entity-grouped)...")
    s1 = sample_s1_grouped(s1_full, frac=0.10, seed=seed)
    print(f"  S1 sample: {len(s1):,} rows")
    del s1_full

    # ===================================================================
    # 2. NORMALIZE → BLOCK → LABEL
    # ===================================================================
    print("\nNormalizing...")
    t0 = time.time()
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    s3_norm = process_dataframe(s3)
    print(f"  Normalization: {time.time()-t0:.1f}s")

    print("\nGenerating candidates (blocking)...")
    t0 = time.time()
    candidates_df = generate_candidates(s1_norm, s2_norm, s3_norm, blk_config)
    print(f"  Blocking: {time.time()-t0:.1f}s  |  {len(candidates_df)} S1 entities")

    print("\nLabeling candidate pairs against ground truth...")
    labeled_df = build_labeled_pairs(candidates_df, ground_truth)
    n_pos = labeled_df["label"].sum()
    n_neg = len(labeled_df) - n_pos
    print(f"  Total pairs: {len(labeled_df):,}  |  Positives: {n_pos:,}  |  Negatives: {n_neg:,}")
    if n_pos == 0:
        print("  ⚠️  WARNING: Zero positive labels — blocking recall may be too low!")
        print("  Run run_blocking_exp.py first to diagnose.")
        return

    # ===================================================================
    # 3. ENTITY-GROUPED SPLIT
    # ===================================================================
    s1_ids = s1["entity_id"].unique()
    train_ids, val_ids, holdout_ids = entity_grouped_split(
        s1_ids, ground_truth,
        val_frac=cfg["splits"]["val_frac"],
        holdout_frac=cfg["splits"]["holdout_frac"],
        seed=seed,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    # holdout is NOT touched during Steps 2-4

    train_pairs = labeled_df[labeled_df["source1_entity_id"].isin(train_set)]
    val_pairs = labeled_df[labeled_df["source1_entity_id"].isin(val_set)]
    print(f"\n  Train pairs: {len(train_pairs):,} (pos: {train_pairs['label'].sum():,})")
    print(f"  Val pairs:   {len(val_pairs):,} (pos: {val_pairs['label'].sum():,})")

    # ===================================================================
    # 4. EXTRACT FEATURES
    # ===================================================================
    # Build record lookup from normalized dataframes
    print("\nBuilding record lookup...")
    records_db = {}
    for df in [s1_norm, s2_norm, s3_norm]:
        for _, r in df.iterrows():
            records_db[r["entity_id"]] = r.to_dict()

    print("Extracting features (train)...")
    t0 = time.time()
    X_train, y_train, meta_train, feature_names = _build_feature_matrix(train_pairs, records_db)
    print(f"  {len(X_train)} pairs, {len(feature_names)} features — {time.time()-t0:.1f}s")
    print(f"  Features: {feature_names}")

    print("Extracting features (val)...")
    t0 = time.time()
    X_val, y_val, meta_val, _ = _build_feature_matrix(val_pairs, records_db)
    print(f"  {len(X_val)} pairs — {time.time()-t0:.1f}s")

    # ===================================================================
    # 5. TRAIN MODELS
    # ===================================================================
    models_to_test = {
        "LogReg_Plain": lambda: train_logreg(X_train, y_train, balanced=False),
        "LogReg_Balanced": lambda: train_logreg(X_train, y_train, balanced=True),
        "RandomForest": lambda: train_rf(X_train, y_train),
        "LightGBM": lambda: train_lgbm(X_train, y_train, cfg),
    }

    # Build val ground truth and candidates for metric computation
    val_gt = {k: ground_truth.get(k, []) for k in val_ids}
    val_cands_dict = {}
    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        if s1_id in val_set:
            cand_str = str(row.get("candidate_entity_ids", "")).strip()
            val_cands_dict[s1_id] = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str else []

    avg_cands = np.mean([len(v) for v in val_cands_dict.values()]) if val_cands_dict else 0.0

    best_model_name = None
    best_f05 = 0.0
    best_threshold = 0.5

    for model_name, train_fn in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        t0 = time.time()
        model = train_fn()
        train_time = time.time() - t0

        save_model(model, f"experiments/models/{model_name}.pkl")

        if len(X_val) == 0:
            print("  No val data — skipping evaluation.")
            continue

        probs = model.predict_proba(X_val)[:, 1]

        # Build scores DataFrame for threshold search
        scores_df = pd.DataFrame(meta_val, columns=["source1_entity_id", "candidate_entity_id"])
        scores_df["score"] = probs

        # Threshold search on val split
        t_cfg = cfg["threshold"]
        best_t, best_val_f05, best_metrics = optimize_threshold(
            scores_df, val_gt,
            t_min=t_cfg["grid_min"],
            t_max=t_cfg["grid_max"],
            t_step=t_cfg["grid_step"],
        )

        runtime = time.time() - t0

        print(f"  Best threshold:    {best_t:.3f}")
        print(f"  Macro F0.5:        {best_metrics['macro_f05']:.4f}")
        print(f"  Macro Precision:   {best_metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:      {best_metrics['macro_recall']:.4f}")
        print(f"  Train time:        {train_time:.2f}s")
        print(f"  Total runtime:     {runtime:.2f}s")

        log_experiment(
            cfg, experiment_name=f"Phase5_{model_name}",
            blocking_desc="ABCD_Pruned",
            features_desc=f"{X_train.shape[1]}_features",
            model_name=model_name,
            threshold=best_t,
            metrics={**best_metrics, "candidate_recall": ""},
            avg_candidates=avg_cands,
            runtime=runtime,
        )

        if best_val_f05 > best_f05:
            best_f05 = best_val_f05
            best_model_name = model_name
            best_threshold = best_t

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}  |  F0.5={best_f05:.4f}  |  Threshold={best_threshold:.3f}")
    print(f"{'='*70}")


def _build_feature_matrix(pairs_df, records_db):
    """Delegate to extract_features_batch — no duplicated logic."""
    return extract_features_batch(pairs_df, records_db)


if __name__ == "__main__":
    main()
