"""
experiments/stage3_hard_negative_experiment.py
Stage 3 Implementation:
  1. 5-Fold Entity-Isolated Cross-Validation within Train split to generate
     leakage-free Out-Of-Fold (OOF) prediction probabilities for all 380,333 training pairs.
  2. Hard-Negative Mining (HNM):
     - Identify high-probability false positives (P >= 0.30, P >= 0.50) where y=0.
     - Isolate hard negatives specifically from Singleton S1 entities and confusing multi-candidate entities.
     - Test controlled integration strategies:
         a) Sample weighting of hard negatives (weight 2.0x, 3.0x, 5.0x)
         b) Oversampling / duplication of hard negatives in controlled ratios
         c) Hard negatives from singletons prioritized
  3. Retrain LightGBM (63 leaves, depth=8) on augmented/re-weighted training set.
  4. Evaluate on the EXACT SAME frozen validation split (1,000 S1 entities, seed=42).
  5. Run threshold sweep [0.40 - 0.95] optimizing official Macro F0.5 and protecting singletons.
  6. Perform 5-Fold Entity-Stratified Cross-Validation on the full 5,000 S1 dataset to assess
     generalization stability and variance across entity folds.
  7. Log all experiments to experiments/experiment_log.csv.
  8. Save comparison report to reports/stage3_hnm_comparison.tsv.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
from anyascii import anyascii
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_dataframe
from src.business_entity_resolution.features import extract_pairwise_features, FEATURE_NAMES_V1
from src.business_entity_resolution.decision import predict_matches
from src.business_entity_resolution.evaluation import split_entities, calculate_metrics

def main():
    print("=" * 80)
    print("STAGE 3: HARD-NEGATIVE MINING & ENTITY-STRATIFIED CROSS-VALIDATION")
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

    # Ground Truth Map
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

    # 2. Fast In-Memory Lookup
    print("\n[2/6] Building lookup database with pre-computed transliterations...")
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
    print(f"Features extracted in {time.time()-t_feat:.2f}s!")

    # 4. Leakage-Safe Entity-Level Split (Identical to Stages 1 & 2)
    print("\n[4/6] Isolating Identical Entity-Level Split (4,000 Train / 1,000 Val, seed=42)...")
    train_s1_ids, val_s1_ids = split_entities(all_s1_ids, ground_truth, val_frac=0.2, seed=42)
    train_set = set(train_s1_ids)
    val_set = set(val_s1_ids)

    train_mask = np.isin(meta_s1, list(train_set))
    val_mask = np.isin(meta_s1, list(val_set))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    train_meta_s1 = meta_s1[train_mask]
    train_meta_cand = meta_cand[train_mask]
    val_meta_s1 = meta_s1[val_mask]
    val_meta_cand = meta_cand[val_mask]

    val_gt = {s1: ground_truth.get(s1, []) for s1 in val_s1_ids}
    val_singletons = sum(1 for s1, m in val_gt.items() if not m)

    print(f"Train pairs: {len(X_train):,} (Pos: {np.sum(y_train):,}, Neg: {len(y_train)-np.sum(y_train):,})")
    print(f"Val pairs:   {len(X_val):,} (Pos: {np.sum(y_val):,}, Neg: {len(y_val)-np.sum(y_val):,})")
    print(f"Val Singletons: {val_singletons:,} ({val_singletons/len(val_gt)*100:.2f}%)")

    # Baseline LightGBM hyperparameters (Stage 2 Winner: 63 leaves, depth 8)
    base_lgbm_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Evaluate Baseline Model first
    print("\n--- Verifying Frozen Baseline Model (Stage 2 Winner) ---")
    t_base = time.time()
    baseline_model = lgb.LGBMClassifier(**base_lgbm_params)
    baseline_model.fit(X_train, y_train, feature_name=FEATURE_NAMES_V1)
    t_base_fit = time.time() - t_base

    val_probs_base = baseline_model.predict_proba(X_val)[:, 1]
    scores_df_base = pd.DataFrame({
        'source1_entity_id': val_meta_s1,
        'candidate_entity_id': val_meta_cand,
        'score': val_probs_base
    })
    preds_base_80 = predict_matches(scores_df_base, threshold=0.80)
    m_base_80 = calculate_metrics(preds_base_80, val_gt)
    print(f"Baseline verified at tau=0.80: Macro F0.5={m_base_80['macro_f05']*100:.2f}%, "
          f"P={m_base_80['macro_precision']*100:.2f}%, R={m_base_80['macro_recall']*100:.2f}%, "
          f"Singleton Acc={m_base_80['singleton_accuracy']*100:.2f}%, Matched Recall={m_base_80['matched_recall']*100:.2f}%")

    # 5. Out-Of-Fold Predictions on Train Split via 5-Fold Entity Isolation
    print("\n[5/6] Generating 5-Fold Entity-Isolated OOF Probabilities on Train Set...")
    t_oof = time.time()
    unique_train_s1 = np.array(train_s1_ids)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_train_probs = np.zeros(len(X_train), dtype=np.float32)

    for fold_idx, (f_trn_idx, f_val_idx) in enumerate(kf.split(unique_train_s1)):
        fold_train_s1 = set(unique_train_s1[f_trn_idx])
        fold_val_s1 = set(unique_train_s1[f_val_idx])

        fold_train_mask = np.isin(train_meta_s1, list(fold_train_s1))
        fold_val_mask = np.isin(train_meta_s1, list(fold_val_s1))

        X_f_trn, y_f_trn = X_train[fold_train_mask], y_train[fold_train_mask]
        X_f_val = X_train[fold_val_mask]

        fold_model = lgb.LGBMClassifier(**base_lgbm_params)
        fold_model.fit(X_f_trn, y_f_trn, feature_name=FEATURE_NAMES_V1)
        
        oof_train_probs[fold_val_mask] = fold_model.predict_proba(X_f_val)[:, 1]
        print(f"  Fold {fold_idx+1}/5 complete: {len(fold_train_s1)} train S1s -> {len(fold_val_s1)} val S1s")

    print(f"OOF generation completed in {time.time()-t_oof:.2f}s!")

    # Analyze OOF False Positives
    train_is_singleton = np.array([len(ground_truth.get(s1, [])) == 0 for s1 in train_meta_s1])
    
    # False positives: y == 0 and prob >= cutoff
    fp_30 = (y_train == 0) & (oof_train_probs >= 0.30)
    fp_50 = (y_train == 0) & (oof_train_probs >= 0.50)
    fp_70 = (y_train == 0) & (oof_train_probs >= 0.70)

    fp_singleton_30 = fp_30 & train_is_singleton
    fp_singleton_50 = fp_50 & train_is_singleton

    print(f"\nOOF Training False Positives:")
    print(f"  Total pairs with y=0: {(y_train == 0).sum():,}")
    print(f"  FP with OOF Prob >= 0.30: {fp_30.sum():,} (Singleton FPs: {fp_singleton_30.sum():,})")
    print(f"  FP with OOF Prob >= 0.50: {fp_50.sum():,} (Singleton FPs: {fp_singleton_50.sum():,})")
    print(f"  FP with OOF Prob >= 0.70: {fp_70.sum():,}")

    # 6. Hard Negative Mining Experiment Grid
    print("\n" + "=" * 80)
    print("[6/6] EXPERIMENTING WITH HARD NEGATIVE MINING (HNM) STRATEGIES")
    print("=" * 80)

    threshold_grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    hnm_strategies = {}

    # Strategy 0: Frozen Baseline (no HNM)
    hnm_strategies['Baseline_NoHNM'] = {
        'weights': np.ones(len(X_train), dtype=np.float32),
        'extra_X': None,
        'extra_y': None,
        'num_hard_negatives': 0,
        'description': 'Stage 2 Frozen Baseline (No HNM)'
    }

    # Strategy 1: Singleton Hard Negatives Upweighted (w=3.0 for singleton FP >= 0.30)
    w_sing_3 = np.ones(len(X_train), dtype=np.float32)
    w_sing_3[fp_singleton_30] = 3.0
    hnm_strategies['HNM_Singleton_FP30_w3'] = {
        'weights': w_sing_3,
        'extra_X': None,
        'extra_y': None,
        'num_hard_negatives': int(fp_singleton_30.sum()),
        'description': f'Upweight Singleton FPs (P>=0.30, N={fp_singleton_30.sum()}) with weight=3.0'
    }

    # Strategy 2: Singleton Hard Negatives Upweighted (w=5.0 for singleton FP >= 0.50)
    w_sing_5 = np.ones(len(X_train), dtype=np.float32)
    w_sing_5[fp_singleton_50] = 5.0
    hnm_strategies['HNM_Singleton_FP50_w5'] = {
        'weights': w_sing_5,
        'extra_X': None,
        'extra_y': None,
        'num_hard_negatives': int(fp_singleton_50.sum()),
        'description': f'Upweight Singleton FPs (P>=0.50, N={fp_singleton_50.sum()}) with weight=5.0'
    }

    # Strategy 3: All Confusing Hard Negatives Upweighted (w=2.0 for all FP >= 0.50)
    w_all_2 = np.ones(len(X_train), dtype=np.float32)
    w_all_2[fp_50] = 2.0
    hnm_strategies['HNM_AllHard_FP50_w2'] = {
        'weights': w_all_2,
        'extra_X': None,
        'extra_y': None,
        'num_hard_negatives': int(fp_50.sum()),
        'description': f'Upweight All FPs (P>=0.50, N={fp_50.sum()}) with weight=2.0'
    }

    # Strategy 4: All Confusing Hard Negatives Upweighted (w=3.0 for all FP >= 0.50)
    w_all_3 = np.ones(len(X_train), dtype=np.float32)
    w_all_3[fp_50] = 3.0
    hnm_strategies['HNM_AllHard_FP50_w3'] = {
        'weights': w_all_3,
        'extra_X': None,
        'extra_y': None,
        'num_hard_negatives': int(fp_50.sum()),
        'description': f'Upweight All FPs (P>=0.50, N={fp_50.sum()}) with weight=3.0'
    }

    # Strategy 5: Controlled Duplication / Oversampling of Singleton FPs (P >= 0.30)
    extra_X_sing = X_train[fp_singleton_30]
    extra_y_sing = y_train[fp_singleton_30]
    hnm_strategies['HNM_Duplication_Singleton_FP30'] = {
        'weights': None,
        'extra_X': extra_X_sing,
        'extra_y': extra_y_sing,
        'num_hard_negatives': int(len(extra_y_sing)),
        'description': f'Duplicate Singleton FPs (P>=0.30, N={len(extra_y_sing)}) in training set'
    }

    # Strategy 6: Controlled Duplication of Top Hard Negatives (P >= 0.50)
    extra_X_all = X_train[fp_50]
    extra_y_all = y_train[fp_50]
    hnm_strategies['HNM_Duplication_AllHard_FP50'] = {
        'weights': None,
        'extra_X': extra_X_all,
        'extra_y': extra_y_all,
        'num_hard_negatives': int(len(extra_y_all)),
        'description': f'Duplicate All Hard FPs (P>=0.50, N={len(extra_y_all)}) in training set'
    }

    hnm_results = []
    experiment_log_rows = []

    for strat_name, strat in hnm_strategies.items():
        print(f"\n--- Evaluating Strategy: {strat_name} ---")
        print(f"Description: {strat['description']}")
        
        t_strat_start = time.time()
        
        # Prepare training data
        if strat['extra_X'] is not None:
            cur_X_train = np.vstack([X_train, strat['extra_X']])
            cur_y_train = np.concatenate([y_train, strat['extra_y']])
            cur_weights = None
        else:
            cur_X_train = X_train
            cur_y_train = y_train
            cur_weights = strat['weights']

        # Train model
        model = lgb.LGBMClassifier(**base_lgbm_params)
        if cur_weights is not None:
            model.fit(cur_X_train, cur_y_train, sample_weight=cur_weights, feature_name=FEATURE_NAMES_V1)
        else:
            model.fit(cur_X_train, cur_y_train, feature_name=FEATURE_NAMES_V1)

        strat_runtime = time.time() - t_strat_start

        # Predict on validation set
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

            exp_row = {
                'experiment_id': f"EXP_Stage3_{strat_name}_tau_{int(tau*100)}",
                'date': time.strftime("%Y-%m-%d"),
                'candidate_version': 'frozen_v2_474k',
                'feature_set': 'Feature_Set_v1_32',
                'train_entities': len(train_s1_ids),
                'validation_entities': len(val_s1_ids),
                'model': f"LightGBM_63L_{strat_name}",
                'model_parameters': f"HNM_strat={strat_name};num_hn={strat['num_hard_negatives']}",
                'threshold': round(tau, 3),
                'macro_f0_5': round(metrics['macro_f05'], 4),
                'macro_precision': round(metrics['macro_precision'], 4),
                'macro_recall': round(metrics['macro_recall'], 4),
                'singleton_accuracy': round(metrics['singleton_accuracy'], 4),
                'matched_recall': round(metrics['matched_recall'], 4),
                'runtime_s': round(strat_runtime, 2),
                'notes': f"Stage 3 HNM (added={strat['num_hard_negatives']})"
            }
            experiment_log_rows.append(exp_row)

            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_row = {
                    'Strategy': strat_name,
                    'Num Hard Negatives': strat['num_hard_negatives'],
                    'Optimal Threshold': round(tau, 3),
                    'Macro F0.5': round(metrics['macro_f05'] * 100, 2),
                    'Precision': round(metrics['macro_precision'] * 100, 2),
                    'Recall': round(metrics['macro_recall'] * 100, 2),
                    'Singleton Accuracy': round(metrics['singleton_accuracy'] * 100, 2),
                    'Matched Recall': round(metrics['matched_recall'] * 100, 2),
                    'Runtime (s)': round(strat_runtime, 2),
                    'Delta vs Baseline': round((metrics['macro_f05'] - m_base_80['macro_f05']) * 100, 2)
                }

        hnm_results.append(best_row)
        print(f"-> Optimal tau={best_row['Optimal Threshold']} | Macro F0.5: {best_row['Macro F0.5']}% | "
              f"Singleton Acc: {best_row['Singleton Accuracy']}% | Delta: {best_row['Delta vs Baseline']:+.2f}%")

    # Summary DataFrame
    hnm_summary_df = pd.DataFrame(hnm_results)
    hnm_summary_df.to_csv("reports/stage3_hnm_comparison.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("STAGE 3 HARD-NEGATIVE MINING COMPARISON SUMMARY")
    print("=" * 80)
    print(hnm_summary_df.to_string(index=False))

    # Append to experiment_log.csv
    exp_log_path = "experiments/experiment_log.csv"
    if os.path.exists(exp_log_path):
        existing_log = pd.read_csv(exp_log_path)
        combined_log = pd.concat([existing_log, pd.DataFrame(experiment_log_rows)], ignore_index=True)
    else:
        combined_log = pd.DataFrame(experiment_log_rows)
    combined_log.to_csv(exp_log_path, index=False)
    print(f"\nLogged {len(experiment_log_rows)} Stage 3 evaluations to {exp_log_path}")

    # 7. 5-Fold Entity-Stratified Cross-Validation on Full Benchmark (5,000 S1s)
    print("\n" + "=" * 80)
    print("[7/7] 5-FOLD ENTITY-STRATIFIED CROSS-VALIDATION (FULL 5,000 S1 BENCHMARK)")
    print("=" * 80)
    
    kf_full = KFold(n_splits=5, shuffle=True, random_state=42)
    all_s1_arr = np.array(all_s1_ids)

    cv_fold_metrics = []

    for fold_idx, (trn_idx, val_idx) in enumerate(kf_full.split(all_s1_arr)):
        f_trn_s1 = set(all_s1_arr[trn_idx])
        f_val_s1 = set(all_s1_arr[val_idx])

        f_trn_mask = np.isin(meta_s1, list(f_trn_s1))
        f_val_mask = np.isin(meta_s1, list(f_val_s1))

        X_f_trn, y_f_trn = X[f_trn_mask], y[f_trn_mask]
        X_f_val, y_f_val = X[f_val_mask], y[f_val_mask]

        f_val_s1_meta = meta_s1[f_val_mask]
        f_val_cand_meta = meta_cand[f_val_mask]
        f_val_gt = {s1: ground_truth.get(s1, []) for s1 in f_val_s1}

        cv_model = lgb.LGBMClassifier(**base_lgbm_params)
        cv_model.fit(X_f_trn, y_f_trn, feature_name=FEATURE_NAMES_V1)

        val_probs_cv = cv_model.predict_proba(X_f_val)[:, 1]
        scores_df_cv = pd.DataFrame({
            'source1_entity_id': f_val_s1_meta,
            'candidate_entity_id': f_val_cand_meta,
            'score': val_probs_cv
        })

        # Evaluate at optimal threshold 0.80
        preds_cv = predict_matches(scores_df_cv, threshold=0.80)
        fold_m = calculate_metrics(preds_cv, f_val_gt)

        fold_row = {
            'Fold': fold_idx + 1,
            'Train S1 Entities': len(f_trn_s1),
            'Val S1 Entities': len(f_val_s1),
            'Macro F0.5': round(fold_m['macro_f05'] * 100, 2),
            'Precision': round(fold_m['macro_precision'] * 100, 2),
            'Recall': round(fold_m['macro_recall'] * 100, 2),
            'Singleton Accuracy': round(fold_m['singleton_accuracy'] * 100, 2),
            'Matched Recall': round(fold_m['matched_recall'] * 100, 2)
        }
        cv_fold_metrics.append(fold_row)
        print(f"Fold {fold_idx+1}: Macro F0.5 = {fold_row['Macro F0.5']}% | "
              f"P = {fold_row['Precision']}% | R = {fold_row['Recall']}% | "
              f"Singleton Acc = {fold_row['Singleton Accuracy']}%")

    cv_df = pd.DataFrame(cv_fold_metrics)
    mean_row = {
        'Fold': 'Mean',
        'Train S1 Entities': 4000,
        'Val S1 Entities': 1000,
        'Macro F0.5': round(cv_df['Macro F0.5'].mean(), 2),
        'Precision': round(cv_df['Precision'].mean(), 2),
        'Recall': round(cv_df['Recall'].mean(), 2),
        'Singleton Accuracy': round(cv_df['Singleton Accuracy'].mean(), 2),
        'Matched Recall': round(cv_df['Matched Recall'].mean(), 2)
    }
    std_row = {
        'Fold': 'Std',
        'Train S1 Entities': 0,
        'Val S1 Entities': 0,
        'Macro F0.5': round(cv_df['Macro F0.5'].std(), 2),
        'Precision': round(cv_df['Precision'].std(), 2),
        'Recall': round(cv_df['Recall'].std(), 2),
        'Singleton Accuracy': round(cv_df['Singleton Accuracy'].std(), 2),
        'Matched Recall': round(cv_df['Matched Recall'].std(), 2)
    }
    cv_summary_df = pd.concat([cv_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    cv_summary_df.to_csv("reports/stage3_cv_summary.tsv", sep="\t", index=False)

    print("\n5-Fold Cross-Validation Summary:")
    print(cv_summary_df.to_string(index=False))

    print("\nStage 3 experiment execution completed successfully!")

if __name__ == "__main__":
    main()
