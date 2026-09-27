"""
experiments/stage1_baselines_experiment.py
Stage 1 Implementation:
  1. Leakage-safe entity-level split (4,000 Train / 1,000 Val S1s)
  2. Feature Set v1 extraction (32 features) across 474,960 pairs
  3. Feature Quality Audit (reports/feature_audit.tsv)
  4. Logistic Regression Baseline (Plain vs Class-Weighted Balanced)
  5. Random Forest Baseline
  6. Threshold optimization on validation split for Macro F0.5
  7. Detailed singleton accuracy & error diagnostics
  8. Experiment logging (experiments/experiment_log.csv)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from anyascii import anyascii

# Add project root and src to path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_dataframe
from src.business_entity_resolution.features import extract_pairwise_features, FEATURE_NAMES_V1
from src.business_entity_resolution.training import train_logreg, train_rf
from src.business_entity_resolution.decision import predict_matches
from src.business_entity_resolution.evaluation import split_entities, calculate_metrics

def main():
    print("=" * 80)
    print("STAGE 1: LEAKAGE-SAFE SPLIT, FEATURE SET V1 & BASELINE TOURNAMENT")
    print("=" * 80)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    t0 = time.time()
    # 1. Load Data
    print("\n[1/6] Loading benchmark data and frozen candidate pairs...")
    s1_path = "dataset/benchmark/s1_benchmark_5000.tsv"
    s2_path = "dataset/benchmark/s2_benchmark_5000_100000.tsv"
    s3_path = "dataset/benchmark/s3_benchmark_5000_100000.tsv"
    gt_path = "dataset/benchmark/gt_benchmark_5000.tsv"
    cand_diag_path = "output/candidate_pairs_diagnostics.tsv"

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    cand_diag = pd.read_csv(cand_diag_path, sep="\t", dtype=str).fillna("")

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_ids = set(target_df['entity_id'])

    print(f"Loaded: S1={len(s1_df):,}, Target Pool={len(target_df):,}")
    print(f"Loaded Frozen Candidate Pairs: {len(cand_diag):,}")

    # Build Ground Truth Map (in-target matches)
    ground_truth: Dict[str, List[str]] = {}
    total_in_target_gt = 0
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
        in_target_matches = [m for m in matches if m in target_ids]
        ground_truth[s1_id] = in_target_matches
        total_in_target_gt += len(in_target_matches)

    all_s1_ids = list(s1_df['entity_id'].unique())
    singletons = [s1 for s1 in all_s1_ids if not ground_truth.get(s1)]
    print(f"Total S1 entities: {len(all_s1_ids):,}")
    print(f"Total in-target ground truth pairs: {total_in_target_gt:,}")
    print(f"True Singletons in pool: {len(singletons):,} ({len(singletons)/len(all_s1_ids)*100:.2f}%)")

    # 2. Build Fast In-Memory Lookup Database with Pre-computed Transliterations
    print("\n[2/6] Building lookup database & pre-computing transliterations...")
    t_norm = time.time()
    if 'name_clean' not in s1_df.columns:
        s1_df = normalize_dataframe(s1_df)
    if 'name_clean' not in target_df.columns:
        target_df = normalize_dataframe(target_df)

    records_db: Dict[str, dict] = {}
    for _, row in s1_df.iterrows():
        d = row.to_dict()
        n = str(d.get("name_clean", ""))
        d["name_translit"] = anyascii(n).lower().strip()
        records_db[d["entity_id"]] = d

    for _, row in target_df.iterrows():
        d = row.to_dict()
        n = str(d.get("name_clean", ""))
        d["name_translit"] = anyascii(n).lower().strip()
        records_db[d["entity_id"]] = d

    print(f"Database indexed {len(records_db):,} records in {time.time()-t_norm:.2f}s")

    # 3. Extract Feature Set v1 (32 features)
    print(f"\n[3/6] Extracting Feature Set v1 ({len(FEATURE_NAMES_V1)} features) across {len(cand_diag):,} pairs...")
    t_feat = time.time()

    pairs_s1 = cand_diag['source1_entity_id'].values
    pairs_c = cand_diag['candidate_entity_id'].values
    pairs_strat = cand_diag['strategies'].values

    n_pairs = len(cand_diag)
    X = np.zeros((n_pairs, len(FEATURE_NAMES_V1)), dtype=np.float32)
    y = np.zeros(n_pairs, dtype=np.int32)
    meta_s1 = []
    meta_cand = []

    for i in range(n_pairs):
        s1_id = pairs_s1[i]
        c_id = pairs_c[i]
        
        s1_rec = records_db.get(s1_id, {})
        c_rec = records_db.get(c_id, {})
        
        feat_dict = extract_pairwise_features(s1_rec, c_rec, pairs_strat[i])
        for j, fname in enumerate(FEATURE_NAMES_V1):
            X[i, j] = feat_dict[fname]

        is_true = 1 if c_id in ground_truth.get(s1_id, []) else 0
        y[i] = is_true
        meta_s1.append(s1_id)
        meta_cand.append(c_id)

    meta_s1 = np.array(meta_s1)
    meta_cand = np.array(meta_cand)

    n_pos = int(np.sum(y))
    print(f"Extracted features in {time.time()-t_feat:.2f}s! Shape: {X.shape}")
    print(f"True Positives: {n_pos:,} ({n_pos/len(y)*100:.2f}%), Negatives (Distractors): {len(y)-n_pos:,}")

    # 4. Feature Quality Audit
    print("\n[4/6] Conducting Feature Quality Audit...")
    audit_rows = []
    for j, fname in enumerate(FEATURE_NAMES_V1):
        col = X[:, j]
        n_missing = int(np.isnan(col).sum())
        n_inf = int(np.isinf(col).sum())
        c_min = float(np.min(col))
        c_mean = float(np.mean(col))
        c_max = float(np.max(col))
        c_std = float(np.std(col))
        is_const = bool(c_std == 0.0)

        audit_rows.append({
            'feature': fname,
            'missing_count': n_missing,
            'inf_count': n_inf,
            'is_constant': is_const,
            'min': round(c_min, 4),
            'mean': round(c_mean, 4),
            'max': round(c_max, 4),
            'std': round(c_std, 4)
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv("reports/feature_audit.tsv", sep="\t", index=False)
    print("Feature Quality Audit saved to reports/feature_audit.tsv:")
    print(audit_df[['feature', 'missing_count', 'is_constant', 'min', 'mean', 'max']].to_string(index=False))

    # 5. Leakage-Safe Entity-Level Split
    print("\n[5/6] Splitting by Source 1 Entity ID (80% Train / 20% Val, Leak-Free)...")
    train_s1_ids, val_s1_ids = split_entities(all_s1_ids, ground_truth, val_frac=0.2, seed=42)
    train_set = set(train_s1_ids)
    val_set = set(val_s1_ids)

    # Verify zero leakage
    leakage = train_set.intersection(val_set)
    assert len(leakage) == 0, f"CRITICAL LEAKAGE ERROR: {len(leakage)} entities in both train and val!"
    print(f"Entity leakage check: 0 overlapping entities (PASS)")

    train_mask = np.isin(meta_s1, list(train_set))
    val_mask = np.isin(meta_s1, list(val_set))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    val_meta_s1 = meta_s1[val_mask]
    val_meta_cand = meta_cand[val_mask]

    train_pos = int(np.sum(y_train))
    val_pos = int(np.sum(y_val))

    print(f"Train split: {len(train_s1_ids):,} S1 entities -> {len(X_train):,} pairs (Pos: {train_pos:,}, Neg: {len(y_train)-train_pos:,})")
    print(f"Val split:   {len(val_s1_ids):,} S1 entities -> {len(X_val):,} pairs (Pos: {val_pos:,}, Neg: {len(y_val)-val_pos:,})")

    # Build validation ground truth mapping
    val_gt = {s1: ground_truth.get(s1, []) for s1 in val_s1_ids}
    val_singletons = sum(1 for s1, m in val_gt.items() if not m)
    print(f"Validation Singletons: {val_singletons:,} ({val_singletons/len(val_gt)*100:.2f}%)")

    # 6. Train and Evaluate Baseline Models
    print("\n[6/6] Training & Evaluating Baseline Models...")
    models = {}

    # Model 1a: Logistic Regression (Plain)
    print("\n--- Training Model 1a: Logistic Regression (Plain) ---")
    t_start = time.time()
    lr_plain = train_logreg(X_train, y_train, balanced=False)
    lr_plain_time = time.time() - t_start
    models['LogReg_Plain'] = (lr_plain, lr_plain_time)
    print(f"Trained LogReg (Plain) in {lr_plain_time:.2f}s")

    # Model 1b: Logistic Regression (Class-Weighted Balanced)
    print("\n--- Training Model 1b: Logistic Regression (Balanced) ---")
    t_start = time.time()
    lr_balanced = train_logreg(X_train, y_train, balanced=True)
    lr_balanced_time = time.time() - t_start
    models['LogReg_Balanced'] = (lr_balanced, lr_balanced_time)
    print(f"Trained LogReg (Balanced) in {lr_balanced_time:.2f}s")

    # Model 2: Random Forest
    print("\n--- Training Model 2: Random Forest (n=100, max_depth=10) ---")
    t_start = time.time()
    rf_model = train_rf(X_train, y_train)
    rf_time = time.time() - t_start
    models['RandomForest'] = (rf_model, rf_time)
    print(f"Trained Random Forest in {rf_time:.2f}s")

    # Threshold Sweep & Evaluation on Validation Split
    print("\nRunning Threshold Sweep across validation entities (Grid: 0.10 to 0.95)...")
    threshold_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    experiment_log_rows = []
    exp_summary = []

    for model_name, (model, train_time) in models.items():
        val_probs = model.predict_proba(X_val)[:, 1]
        scores_df = pd.DataFrame({
            'source1_entity_id': val_meta_s1,
            'candidate_entity_id': val_meta_cand,
            'score': val_probs
        })

        best_f05 = -1.0
        best_row = None

        for tau in threshold_grid:
            preds = predict_matches(scores_df, threshold=tau)
            metrics = calculate_metrics(preds, val_gt)

            row = {
                'experiment_id': f"EXP_{model_name}_tau_{int(tau*100)}",
                'date': time.strftime("%Y-%m-%d"),
                'candidate_version': 'frozen_v2_474k',
                'feature_set': 'Feature_Set_v1_32',
                'train_entities': len(train_s1_ids),
                'validation_entities': len(val_s1_ids),
                'model': model_name,
                'model_parameters': str(model.get_params()),
                'threshold': round(tau, 3),
                'macro_f0_5': round(metrics['macro_f05'], 4),
                'macro_precision': round(metrics['macro_precision'], 4),
                'macro_recall': round(metrics['macro_recall'], 4),
                'singleton_accuracy': round(metrics['singleton_accuracy'], 4),
                'matched_recall': round(metrics['matched_recall'], 4),
                'runtime_s': round(train_time, 2),
                'notes': 'Baseline evaluation Stage 1'
            }
            experiment_log_rows.append(row)

            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_row = row

        exp_summary.append(best_row)
        print(f"[{model_name}] Optimal Threshold: {best_row['threshold']:.2f} -> Macro F0.5: {best_row['macro_f0_5']*100:.2f}% (P: {best_row['macro_precision']*100:.2f}%, R: {best_row['macro_recall']*100:.2f}%, Singleton Acc: {best_row['singleton_accuracy']*100:.2f}%)")

    # Save to experiments/experiment_log.csv
    exp_log_path = "experiments/experiment_log.csv"
    exp_df = pd.DataFrame(experiment_log_rows)
    exp_df.to_csv(exp_log_path, index=False)
    print(f"\nLogged {len(exp_df)} threshold evaluations to {exp_log_path}")

    # Stage 1 Summary Report
    print("\n" + "=" * 80)
    print("STAGE 1 BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(exp_summary)[['model', 'threshold', 'macro_f0_5', 'macro_precision', 'macro_recall', 'singleton_accuracy', 'matched_recall', 'runtime_s']]
    print(summary_df.to_string(index=False))

    print("\nStage 1 execution successfully completed! Ready for analysis.")

if __name__ == "__main__":
    main()
