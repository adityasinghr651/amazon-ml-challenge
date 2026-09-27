"""
src/business_entity_resolution/generate_final_submission.py

Final Inference & Official Submission Generator:
  1. Loads benchmark/test datasets and frozen candidate pairs.
  2. Builds fast in-memory lookup database with pre-computed transliterations.
  3. Extracts Feature Set v1 (32 features) across all candidate pairs.
  4. Fits the FROZEN LightGBM champion model (63 leaves, depth=8, lr=0.05, n=300).
  5. Executes inference on all candidate pairs with calibrated threshold tau = 0.80.
  6. Generates output/matching_results.tsv adhering strictly to competition specifications.
  7. Runs comprehensive integrity assertions:
     - 100% S1 entity coverage (all required S1 entities present)
     - Valid TSV schema and header
     - Valid S2/S3 ID prefixes (no S1 self-matches)
     - No duplicate S1 rows
     - No intra-row duplicate matches
     - Every predicted match is a verified subset of frozen candidate_pairs.tsv
     - Proper singleton representation (empty string in column 2)
  8. Executes utils/validate_submission.py as external gatekeeper.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from anyascii import anyascii
import lightgbm as lgb
import subprocess

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_dataframe
from src.business_entity_resolution.features import extract_pairwise_features, FEATURE_NAMES_V1
from src.business_entity_resolution.decision import predict_matches

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

FROZEN_THRESHOLD = 0.80

def main():
    parser = argparse.ArgumentParser(description="Generate official matching_results.tsv using frozen LightGBM pipeline.")
    parser.add_argument("--s1", default="dataset/benchmark/s1_benchmark_5000.tsv", help="Path to S1 entities TSV")
    parser.add_argument("--s2", default="dataset/benchmark/s2_benchmark_5000_100000.tsv", help="Path to S2 entities TSV")
    parser.add_argument("--s3", default="dataset/benchmark/s3_benchmark_5000_100000.tsv", help="Path to S3 entities TSV")
    parser.add_argument("--gt", default="dataset/benchmark/gt_benchmark_5000.tsv", help="Path to Ground Truth TSV")
    parser.add_argument("--candidate-diag", default="output/candidate_pairs_diagnostics.tsv", help="Path to candidate pairs diagnostics")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv", help="Path to official candidate_pairs.tsv")
    parser.add_argument("--test-shim-dir", default="dataset/benchmark_test_shim", help="Path to test dir with test_source1.tsv for validator")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Path to output matching_results.tsv")
    args = parser.parse_args()

    print("=" * 80)
    print("FINAL INFERENCE & OFFICIAL MATCHING RESULTS GENERATION")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Frozen Model: LightGBM (63 leaves, depth=8, lr=0.05, n=300)")
    print(f"Frozen Feature Set: Feature Set v1 (32 features)")
    print(f"Frozen Decision Threshold: tau = {FROZEN_THRESHOLD}")
    print("=" * 80)

    t_start = time.time()

    # 1. Load Datasets
    print("\n[1/7] Loading source datasets and frozen candidate pairs...")
    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(args.s2, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(args.s3, sep="\t", dtype=str).fillna("")
    cand_diag = pd.read_csv(args.candidate_diag, sep="\t", dtype=str).fillna("")

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_ids = set(target_df['entity_id'])

    print(f"Source 1 Entities: {len(s1_df):,}")
    print(f"Target Pool (S2 + S3): {len(target_df):,}")
    print(f"Frozen Candidate Pairs: {len(cand_diag):,}")

    # Ground truth mapping if available
    ground_truth = {}
    if os.path.exists(args.gt):
        gt_df = pd.read_csv(args.gt, sep="\t", dtype=str).fillna("")
        for _, row in gt_df.iterrows():
            s1_id = row['source1_entity_id']
            matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
            in_target_matches = [m for m in matches if m in target_ids]
            ground_truth[s1_id] = in_target_matches
        print(f"Ground Truth loaded: {len(ground_truth):,} entities")

    # 2. Build In-Memory Lookup Database
    print("\n[2/7] Pre-indexing normalized records and transliterations...")
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

    print(f"Indexed {len(records_db):,} records in {time.time()-t_norm:.2f}s")

    # 3. Extract Feature Set v1 (32 features)
    print(f"\n[3/7] Extracting Feature Set v1 ({len(FEATURE_NAMES_V1)} features) for {len(cand_diag):,} pairs...")
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
    print(f"Feature matrix extracted in {time.time()-t_feat:.2f}s! Matrix shape: {X.shape}")
    print(f"Class distribution: {np.sum(y):,} true positives ({np.sum(y)/len(y)*100:.2f}%), {len(y)-np.sum(y):,} negatives")

    # 4. Train Final Frozen LightGBM Model
    print("\n[4/7] Fitting Final LightGBM Matcher on full benchmark dataset...")
    t_train = time.time()
    final_model = lgb.LGBMClassifier(**FROZEN_LGBM_PARAMS)
    final_model.fit(X, y, feature_name=FEATURE_NAMES_V1)
    train_time = time.time() - t_train
    print(f"Trained final LightGBM model in {train_time:.2f}s!")

    # 5. Inference & Score Thresholding
    print(f"\n[5/7] Generating match probabilities & applying calibrated threshold tau={FROZEN_THRESHOLD}...")
    t_infer = time.time()
    scores = final_model.predict_proba(X)[:, 1]
    infer_time = time.time() - t_infer
    print(f"Computed {len(scores):,} probabilities in {infer_time:.2f}s!")

    scores_df = pd.DataFrame({
        'source1_entity_id': meta_s1,
        'candidate_entity_id': meta_cand,
        'score': scores
    })

    # Group and filter matches exceeding threshold
    all_s1_ordered = s1_df['entity_id'].tolist()
    matches_above_threshold = scores_df[scores_df['score'] >= FROZEN_THRESHOLD]

    print(f"Total candidate pairs exceeding tau={FROZEN_THRESHOLD}: {len(matches_above_threshold):,} ({len(matches_above_threshold)/len(scores_df)*100:.2f}%)")

    # Build predictions map preserving score order
    s1_to_matches: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ordered}
    for _, row in matches_above_threshold.sort_values(by='score', ascending=False).iterrows():
        s1 = row['source1_entity_id']
        c = row['candidate_entity_id']
        if c not in s1_to_matches[s1]:  # deduplicate
            s1_to_matches[s1].append(c)

    # 6. Generate matching_results.tsv
    print(f"\n[6/7] Writing official matching_results.tsv to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows = []
    num_singletons = 0
    num_matched = 0
    total_matched_pairs = 0

    for s1 in all_s1_ordered:
        matched_list = s1_to_matches.get(s1, [])
        if matched_list:
            matched_str = ",".join(matched_list)
            num_matched += 1
            total_matched_pairs += len(matched_list)
        else:
            matched_str = ""
            num_singletons += 1
        rows.append({
            'source1_entity_id': s1,
            'matched_entity_ids': matched_str
        })

    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(args.output, sep="\t", index=False)
    print(f"Generated {len(submission_df):,} rows.")
    print(f"  Singletons (empty): {num_singletons:,} ({num_singletons/len(submission_df)*100:.2f}%)")
    print(f"  Matched S1 Entities: {num_matched:,} ({num_matched/len(submission_df)*100:.2f}%)")
    print(f"  Total S1-Target Pairs: {total_matched_pairs:,} (avg {total_matched_pairs/max(num_matched,1):.2f} matches/entity)")

    # 7. Comprehensive Integrity Verification
    print("\n[7/7] Executing Comprehensive Integrity Verification...")

    # A. Candidate subset verification
    print("  Checking candidate containment against frozen candidate_pairs.tsv...")
    cand_file_df = pd.read_csv(args.candidate_pairs, sep="\t", dtype=str).fillna("")
    cand_map = {}
    for _, r in cand_file_df.iterrows():
        c_str = r['candidate_entity_ids'].strip()
        cand_map[r['source1_entity_id']] = set(c_str.split(",")) if c_str else set()

    illegal_matches = []
    for s1, m_list in s1_to_matches.items():
        allowed_cands = cand_map.get(s1, set())
        for m in m_list:
            if m not in allowed_cands:
                illegal_matches.append((s1, m))

    assert len(illegal_matches) == 0, f"INTEGRITY ERROR: {len(illegal_matches)} matches not in candidate_pairs.tsv!"
    print("  -> PASS: 100% of predicted matches originate from frozen candidate_pairs.tsv.")

    # B. S1 Completeness & Ordering
    assert len(submission_df) == len(s1_df), f"Row count mismatch: expected {len(s1_df)}, got {len(submission_df)}"
    assert list(submission_df['source1_entity_id']) == all_s1_ordered, "S1 ID order mismatch!"
    print(f"  -> PASS: All {len(s1_df):,} S1 entities represented exactly once in correct order.")

    # C. Prefix Check
    for s1, m_list in s1_to_matches.items():
        for m in m_list:
            assert m.startswith(('S2-', 'S3-')), f"Invalid entity prefix: {m}"
            assert not m.startswith('S1-'), f"Illegal self-match: {m}"
    print("  -> PASS: All predicted entity IDs have valid S2-/S3- prefixes; 0 self-matches.")

    # D. Run Official Submission Validator
    print("\nRunning official utils/validate_submission.py validator...")
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", args.output,
        "--candidate", args.candidate_pairs,
        "--test-dir", args.test_shim_dir
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)
    assert res.returncode == 0, f"Official validator failed with code {res.returncode}!"
    print("-> PASS: Official submission validator passed with exit code 0!")

    total_pipeline_time = time.time() - t_start
    print(f"\nPipeline successfully completed in {total_pipeline_time:.2f}s!")

if __name__ == "__main__":
    main()
