"""
src/business_entity_resolution/train_and_save_champion_model.py
Trains the FROZEN LightGBM champion matcher on the benchmark dataset
and persists it to models/final_lightgbm_champion.pkl for zero-latency inference.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from anyascii import anyascii
import lightgbm as lgb

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_dataframe
from src.business_entity_resolution.features import extract_pairwise_features, FEATURE_NAMES_V1

FROZEN_LGBM_PARAMS = {
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

def main():
    print("=" * 80)
    print("TRAINING & PERSISTING FROZEN LIGHTGBM CHAMPION MATCHER")
    print("=" * 80)

    os.makedirs("models", exist_ok=True)
    model_path = "models/final_lightgbm_champion.pkl"

    s1_path = "dataset/benchmark/s1_benchmark_5000.tsv"
    s2_path = "dataset/benchmark/s2_benchmark_5000_100000.tsv"
    s3_path = "dataset/benchmark/s3_benchmark_5000_100000.tsv"
    gt_path = "dataset/benchmark/gt_benchmark_5000.tsv"
    cand_diag_path = "output/candidate_pairs_diagnostics.tsv"

    print("Loading benchmark data...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    cand_diag = pd.read_csv(cand_diag_path, sep="\t", dtype=str).fillna("")

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_ids = set(target_df['entity_id'])

    ground_truth = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
        ground_truth[s1_id] = [m for m in matches if m in target_ids]

    print("Normalizing records...")
    s1_df = normalize_dataframe(s1_df)
    target_df = normalize_dataframe(target_df)

    records_db = {}
    for _, row in s1_df.iterrows():
        d = row.to_dict()
        d["name_translit"] = anyascii(str(d.get("name_clean", ""))).lower().strip()
        records_db[d["entity_id"]] = d

    for _, row in target_df.iterrows():
        d = row.to_dict()
        d["name_translit"] = anyascii(str(d.get("name_clean", ""))).lower().strip()
        records_db[d["entity_id"]] = d

    print(f"Extracting features across {len(cand_diag):,} candidate pairs...")
    pairs_s1 = cand_diag['source1_entity_id'].values
    pairs_c = cand_diag['candidate_entity_id'].values
    pairs_strat = cand_diag['strategies'].values

    n_pairs = len(cand_diag)
    X = np.zeros((n_pairs, len(FEATURE_NAMES_V1)), dtype=np.float32)
    y = np.zeros(n_pairs, dtype=np.int32)

    for i in range(n_pairs):
        s1_id = pairs_s1[i]
        c_id = pairs_c[i]
        s1_rec = records_db.get(s1_id, {})
        c_rec = records_db.get(c_id, {})
        feat_dict = extract_pairwise_features(s1_rec, c_rec, pairs_strat[i])
        for j, fname in enumerate(FEATURE_NAMES_V1):
            X[i, j] = feat_dict[fname]
        y[i] = 1 if c_id in ground_truth.get(s1_id, []) else 0

    print("Fitting model...")
    t_fit = time.time()
    model = lgb.LGBMClassifier(**FROZEN_LGBM_PARAMS)
    model.fit(X, y, feature_name=FEATURE_NAMES_V1)
    print(f"Model fitted in {time.time()-t_fit:.2f}s!")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved champion model to {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
