"""
experiments/stage2_lightgbm_experiment.py
Stage 2 Implementation:
  1. LightGBM classifier training on frozen Feature Set v1 (32 features)
  2. Controlled, small hyperparameter grid search (4 principled configurations)
  3. Identical S1 entity-level split (4,000 train / 1,000 val, seed=42)
  4. Decision threshold sweep optimizing Macro F0.5
  5. Within-S1 candidate ranking and score margin logic evaluation:
     - Pure threshold
     - Top-1 only
     - Top-1 + margin-constrained secondary matches
  6. Feature importance analysis (split & gain)
  7. Detailed comparison against Stage 1 Random Forest baseline
  8. Append all experiment results to experiments/experiment_log.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
from anyascii import anyascii
import lightgbm as lgb

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_dataframe
from src.business_entity_resolution.features import extract_pairwise_features, FEATURE_NAMES_V1
from src.business_entity_resolution.decision import predict_matches
from src.business_entity_resolution.evaluation import split_entities, calculate_metrics

def main():
    print("=" * 80)
    print("STAGE 2: LIGHTGBM MATCHING MODEL & ENTITY RANKING DECISION LAYER")
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

    # 2. Build Fast In-Memory Lookup Database
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

    n_pos = int(np.sum(y))
    print(f"Extracted features in {time.time()-t_feat:.2f}s! Shape: {X.shape}")
    print(f"True Positives: {n_pos:,} ({n_pos/len(y)*100:.2f}%), Negatives: {len(y)-n_pos:,}")

    # 4. Leakage-Safe Entity-Level Split (Identical to Stage 1)
    print("\n[4/6] Applying Identical Entity-Level Split (4,000 Train / 1,000 Val, seed=42)...")
    train_s1_ids, val_s1_ids = split_entities(all_s1_ids, ground_truth, val_frac=0.2, seed=42)
    train_set = set(train_s1_ids)
    val_set = set(val_s1_ids)

    train_mask = np.isin(meta_s1, list(train_set))
    val_mask = np.isin(meta_s1, list(val_set))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    val_meta_s1 = meta_s1[val_mask]
    val_meta_cand = meta_cand[val_mask]

    val_gt = {s1: ground_truth.get(s1, []) for s1 in val_s1_ids}
    val_singletons = sum(1 for s1, m in val_gt.items() if not m)
    print(f"Train pairs: {len(X_train):,} (Pos: {np.sum(y_train):,}, Neg: {len(y_train)-np.sum(y_train):,})")
    print(f"Val pairs:   {len(X_val):,} (Pos: {np.sum(y_val):,}, Neg: {len(y_val)-np.sum(y_val):,})")
    print(f"Validation Singletons: {val_singletons:,} ({val_singletons/len(val_gt)*100:.2f}%)")

    # 5. Controlled LightGBM Hyperparameter Grid
    print("\n[5/6] Training & Evaluating Controlled LightGBM Configurations...")
    
    lgbm_configs = {
        'LGBM_Default_31Leaves': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'LGBM_Deeper_63Leaves': {
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
        },
        'LGBM_Conservative_Regularized': {
            'n_estimators': 400,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'LGBM_Precision_MinChild100': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    }

    threshold_grid = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    stage2_log_rows = []
    best_config_name = None
    best_lgbm_model = None
    best_overall_f05 = -1.0
    best_lgbm_summary = []
    best_val_scores_df = None

    for cfg_name, params in lgbm_configs.items():
        print(f"\n--- Training {cfg_name} ---")
        t_start = time.time()
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, feature_name=FEATURE_NAMES_V1)
        train_time = time.time() - t_start
        print(f"Trained in {train_time:.2f}s")

        val_probs = model.predict_proba(X_val)[:, 1]
        scores_df = pd.DataFrame({
            'source1_entity_id': val_meta_s1,
            'candidate_entity_id': val_meta_cand,
            'score': val_probs
        })

        best_m_f05 = -1.0
        best_row = None

        for tau in threshold_grid:
            preds = predict_matches(scores_df, threshold=tau)
            metrics = calculate_metrics(preds, val_gt)

            row = {
                'experiment_id': f"EXP_Stage2_{cfg_name}_tau_{int(tau*100)}",
                'date': time.strftime("%Y-%m-%d"),
                'candidate_version': 'frozen_v2_474k',
                'feature_set': 'Feature_Set_v1_32',
                'train_entities': len(train_s1_ids),
                'validation_entities': len(val_s1_ids),
                'model': cfg_name,
                'model_parameters': str(params),
                'threshold': round(tau, 3),
                'macro_f0_5': round(metrics['macro_f05'], 4),
                'macro_precision': round(metrics['macro_precision'], 4),
                'macro_recall': round(metrics['macro_recall'], 4),
                'singleton_accuracy': round(metrics['singleton_accuracy'], 4),
                'matched_recall': round(metrics['matched_recall'], 4),
                'runtime_s': round(train_time, 2),
                'notes': 'Stage 2 LightGBM controlled grid evaluation'
            }
            stage2_log_rows.append(row)

            if metrics['macro_f05'] > best_m_f05:
                best_m_f05 = metrics['macro_f05']
                best_row = row

        best_lgbm_summary.append(best_row)
        print(f"[{cfg_name}] Optimal Threshold: {best_row['threshold']:.2f} -> Macro F0.5: {best_row['macro_f0_5']*100:.2f}% (P: {best_row['macro_precision']*100:.2f}%, R: {best_row['macro_recall']*100:.2f}%, Singleton Acc: {best_row['singleton_accuracy']*100:.2f}%)")

        if best_m_f05 > best_overall_f05:
            best_overall_f05 = best_m_f05
            best_config_name = cfg_name
            best_lgbm_model = model
            best_val_scores_df = scores_df

    # Feature Importance for Winning LightGBM model
    print(f"\nExtracting Feature Importance for winning model: {best_config_name}...")
    importances_split = best_lgbm_model.feature_importances_
    importances_gain = best_lgbm_model.booster_.feature_importance(importance_type='gain')

    feat_imp_df = pd.DataFrame({
        'feature': FEATURE_NAMES_V1,
        'importance_split': importances_split,
        'importance_gain': np.round(importances_gain, 2)
    }).sort_values(by='importance_gain', ascending=False).reset_index(drop=True)

    feat_imp_df.to_csv("reports/stage2_feature_importances.tsv", sep="\t", index=False)
    print("\nTop 15 Most Discriminative Features (by Gain):")
    print(feat_imp_df.head(15).to_string(index=False))

    # 6. Within-S1 Entity Candidate Ranking & Score Margin Logic
    print("\n" + "=" * 80)
    print("[6/6] EVALUATING WITHIN-S1 CANDIDATE RANKING & SCORE MARGIN LOGIC")
    print("=" * 80)

    # Winning threshold under pure threshold
    best_tau = [r for r in best_lgbm_summary if r['model'] == best_config_name][0]['threshold']
    print(f"Baseline Pure Threshold: tau={best_tau:.2f} -> Macro F0.5 = {best_overall_f05*100:.2f}%")

    def predict_with_entity_ranking_and_margin(
        scores_df: pd.DataFrame,
        tau: float,
        mode: str = 'pure_threshold',
        margin: float = 0.10,
        max_matches: Optional[int] = None
    ) -> Dict[str, List[str]]:
        predictions = {s1: [] for s1 in val_s1_ids}
        
        # Group by S1
        grouped = scores_df.groupby('source1_entity_id')
        for s1_id, group in grouped:
            # Sort candidates by probability descending
            sorted_group = group.sort_values(by='score', ascending=False)
            
            top_cand = sorted_group.iloc[0]
            top_score = top_cand['score']

            # If top score doesn't reach threshold, entity is predicted as singleton []
            if top_score < tau:
                continue

            if mode == 'top1_only':
                predictions[s1_id] = [top_cand['candidate_entity_id']]
            elif mode == 'top1_plus_margin':
                # First candidate qualifies
                selected = [top_cand['candidate_entity_id']]
                # Secondary candidates must exceed tau AND be within (top_score - margin)
                for idx in range(1, len(sorted_group)):
                    row = sorted_group.iloc[idx]
                    if row['score'] >= tau and row['score'] >= (top_score - margin):
                        selected.append(row['candidate_entity_id'])
                        if max_matches and len(selected) >= max_matches:
                            break
                    else:
                        break  # since sorted descending, further candidates have lower scores
                predictions[s1_id] = selected
            else:  # pure threshold
                qual = sorted_group[sorted_group['score'] >= tau]['candidate_entity_id'].tolist()
                predictions[s1_id] = qual

        return predictions

    ranking_experiments = []

    # A. Test Pure Threshold
    p_pure = predict_with_entity_ranking_and_margin(best_val_scores_df, tau=best_tau, mode='pure_threshold')
    m_pure = calculate_metrics(p_pure, val_gt)
    ranking_experiments.append({
        'decision_rule': 'Pure_Threshold',
        'tau': best_tau,
        'margin': 'N/A',
        'macro_f0_5': round(m_pure['macro_f05'], 4),
        'macro_precision': round(m_pure['macro_precision'], 4),
        'macro_recall': round(m_pure['macro_recall'], 4),
        'singleton_accuracy': round(m_pure['singleton_accuracy'], 4),
        'matched_recall': round(m_pure['matched_recall'], 4)
    })

    # B. Test Top-1 Only Requirement across tau
    for t in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        p_top1 = predict_with_entity_ranking_and_margin(best_val_scores_df, tau=t, mode='top1_only')
        m_top1 = calculate_metrics(p_top1, val_gt)
        ranking_experiments.append({
            'decision_rule': 'Top1_Only',
            'tau': t,
            'margin': '0.00',
            'macro_f0_5': round(m_top1['macro_f05'], 4),
            'macro_precision': round(m_top1['macro_precision'], 4),
            'macro_recall': round(m_top1['macro_recall'], 4),
            'singleton_accuracy': round(m_top1['singleton_accuracy'], 4),
            'matched_recall': round(m_top1['matched_recall'], 4)
        })

    # C. Test Top-1 + Score Margin Delta across margins [0.05, 0.10, 0.15, 0.20, 0.25]
    for margin_val in [0.05, 0.10, 0.15, 0.20, 0.25]:
        p_margin = predict_with_entity_ranking_and_margin(best_val_scores_df, tau=best_tau, mode='top1_plus_margin', margin=margin_val)
        m_margin = calculate_metrics(p_margin, val_gt)
        ranking_experiments.append({
            'decision_rule': f'Top1_Plus_Margin',
            'tau': best_tau,
            'margin': str(margin_val),
            'macro_f0_5': round(m_margin['macro_f05'], 4),
            'macro_precision': round(m_margin['macro_precision'], 4),
            'macro_recall': round(m_margin['macro_recall'], 4),
            'singleton_accuracy': round(m_margin['singleton_accuracy'], 4),
            'matched_recall': round(m_margin['matched_recall'], 4)
        })

    ranking_df = pd.DataFrame(ranking_experiments)
    print("\nDecision Layer / Candidate Ranking Comparison:")
    print(ranking_df.to_string(index=False))

    # Append all evaluations to experiments/experiment_log.csv
    exp_log_path = "experiments/experiment_log.csv"
    if os.path.exists(exp_log_path):
        existing_log = pd.read_csv(exp_log_path)
        combined_log = pd.concat([existing_log, pd.DataFrame(stage2_log_rows)], ignore_index=True)
    else:
        combined_log = pd.DataFrame(stage2_log_rows)
    combined_log.to_csv(exp_log_path, index=False)
    print(f"\nAppended {len(stage2_log_rows)} Stage 2 evaluations to {exp_log_path}")

    # Stage 2 Direct Comparison Table against Frozen Stage 1 Baselines
    print("\n" + "=" * 80)
    print("STAGE 2 FINAL HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    # Retrieve Stage 1 best rows
    rf_best = existing_log[existing_log['model'] == 'RandomForest'].sort_values(by='macro_f0_5', ascending=False).iloc[0]
    lr_best = existing_log[existing_log['model'] == 'LogReg_Plain'].sort_values(by='macro_f0_5', ascending=False).iloc[0]
    lgbm_best = pd.DataFrame(best_lgbm_summary).sort_values(by='macro_f0_5', ascending=False).iloc[0]

    head_to_head = pd.DataFrame([
        {
            'Stage': 'Stage 1 Baseline',
            'Model': 'Logistic Regression (Plain)',
            'Threshold': lr_best['threshold'],
            'Macro F0.5': f"{lr_best['macro_f0_5']*100:.2f}%",
            'Precision': f"{lr_best['macro_precision']*100:.2f}%",
            'Recall': f"{lr_best['macro_recall']*100:.2f}%",
            'Singleton Acc': f"{lr_best['singleton_accuracy']*100:.2f}%",
            'Matched Recall': f"{lr_best['matched_recall']*100:.2f}%",
            'Runtime': f"{lr_best['runtime_s']:.2f}s"
        },
        {
            'Stage': 'Stage 1 Baseline',
            'Model': 'Random Forest (100 trees)',
            'Threshold': rf_best['threshold'],
            'Macro F0.5': f"{rf_best['macro_f0_5']*100:.2f}%",
            'Precision': f"{rf_best['macro_precision']*100:.2f}%",
            'Recall': f"{rf_best['macro_recall']*100:.2f}%",
            'Singleton Acc': f"{rf_best['singleton_accuracy']*100:.2f}%",
            'Matched Recall': f"{rf_best['matched_recall']*100:.2f}%",
            'Runtime': f"{rf_best['runtime_s']:.2f}s"
        },
        {
            'Stage': 'Stage 2 Winner',
            'Model': f"LightGBM ({best_config_name})",
            'Threshold': lgbm_best['threshold'],
            'Macro F0.5': f"{lgbm_best['macro_f0_5']*100:.2f}%",
            'Precision': f"{lgbm_best['macro_precision']*100:.2f}%",
            'Recall': f"{lgbm_best['macro_recall']*100:.2f}%",
            'Singleton Acc': f"{lgbm_best['singleton_accuracy']*100:.2f}%",
            'Matched Recall': f"{lgbm_best['matched_recall']*100:.2f}%",
            'Runtime': f"{lgbm_best['runtime_s']:.2f}s"
        }
    ])

    print(head_to_head.to_string(index=False))

    # Save comparison table
    head_to_head.to_csv("reports/stage2_model_comparison.tsv", sep="\t", index=False)
    ranking_df.to_csv("reports/stage2_decision_ranking.tsv", sep="\t", index=False)
    print("\nStage 2 completed and stopped! Ready for reporting.")

if __name__ == "__main__":
    main()
