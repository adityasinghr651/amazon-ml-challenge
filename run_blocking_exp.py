"""
run_blocking_exp.py — Blocking evaluation on a 10% entity-grouped sample.

Reports, split by match-containing S1s vs singleton S1s:
  - Candidate recall (blocking recall) on match-containing entities
  - Avg candidates per entity
  - False-candidate rate on singleton entities
  - Theoretical F0.5 ceiling implied by blocking recall alone
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.io import load_train_data
from src.business_entity_resolution.normalization import process_dataframe
from src.business_entity_resolution.blocking_legacy import generate_candidates, measure_blocking_performance
from src.business_entity_resolution.preprocessing import parse_ground_truth, sample_s1_grouped
from src.business_entity_resolution.utils import load_config, get_blocking_config


def compute_f05_ceiling(blocking_recall: float) -> float:
    """
    Theoretical best F0.5 if classifier were perfect, given blocking recall.
    Perfect classifier => precision=1.0, recall=blocking_recall.
    F0.5 = 1.25 * P * R / (0.25 * P + R) = 1.25 * R / (0.25 + R)
    """
    if blocking_recall <= 0:
        return 0.0
    return (1.25 * blocking_recall) / (0.25 + blocking_recall)


def run_experiment():
    cfg = load_config()
    blk_config = get_blocking_config(cfg)
    seed = cfg["random_state"]

    print("=" * 70)
    print("BLOCKING EXPERIMENT — 10% Entity-Grouped Sample")
    print("=" * 70)

    # 1. Load data
    print("\nLoading datasets...")
    s1_full, s2, s3, gt_df = load_train_data()
    ground_truth = parse_ground_truth(gt_df)

    # 2. Entity-grouped 10% sample of S1
    print("Sampling 10% of S1 entities (entity-grouped, not row-random)...")
    s1 = sample_s1_grouped(s1_full, frac=0.10, seed=seed)
    sampled_ids = set(s1["entity_id"].unique())
    print(f"  Sampled {len(sampled_ids):,} S1 entities out of {len(s1_full):,}")
    del s1_full

    # Categorize sampled S1s
    match_s1_ids = [sid for sid in sampled_ids if ground_truth.get(sid, [])]
    singleton_s1_ids = [sid for sid in sampled_ids if not ground_truth.get(sid, [])]
    print(f"  Match-containing S1s: {len(match_s1_ids):,}")
    print(f"  Singleton S1s:        {len(singleton_s1_ids):,}")

    # 3. Normalize
    print("\nNormalizing S1 ({:,} rows)...".format(len(s1)))
    s1_norm = process_dataframe(s1)
    print("Normalizing S2 ({:,} rows)...".format(len(s2)))
    s2_norm = process_dataframe(s2)
    print("Normalizing S3 ({:,} rows)...".format(len(s3)))
    s3_norm = process_dataframe(s3)

    total_s2s3 = len(s2) + len(s3)

    # 4. Run blocking experiments
    experiments = [
        {"name": "Block A (Exact Name)", "config": {**blk_config, "active_blocks": ["A"], "enable_pruning": False}},
        {"name": "Block A+B (Name+Addr)", "config": {**blk_config, "active_blocks": ["A", "B"], "enable_pruning": False}},
        {"name": "Block A+B+C (Rare Tok)", "config": {**blk_config, "active_blocks": ["A", "B", "C"], "enable_pruning": True}},
        {"name": "Full A+B+C+D", "config": blk_config},
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print(f"  Config: {exp['config']}")
        print(f"{'='*60}")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = time.time()

        candidates_df = generate_candidates(s1_norm, s2_norm, s3_norm, exp["config"])

        runtime = time.time() - t0
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_delta = mem_after - mem_before

        # --- Decompose metrics by match-containing vs singleton ---

        # Parse candidates into dict
        cands_dict = {}
        for _, row in candidates_df.iterrows():
            s1_id = str(row["source1_entity_id"])
            cand_str = str(row.get("candidate_entity_ids", "")).strip()
            cands_dict[s1_id] = set(c.strip() for c in cand_str.split(",") if c.strip()) if cand_str else set()

        # -- Match-containing S1s --
        fully_recovered = 0
        total_match_cands = 0
        for sid in match_s1_ids:
            true_matches = set(ground_truth[sid])
            pred_cands = cands_dict.get(sid, set())
            total_match_cands += len(pred_cands)
            if true_matches.issubset(pred_cands):
                fully_recovered += 1

        match_recall = fully_recovered / len(match_s1_ids) if match_s1_ids else 0.0
        avg_cands_match = total_match_cands / len(match_s1_ids) if match_s1_ids else 0.0

        # -- Singleton S1s --
        total_singleton_cands = 0
        singletons_with_cands = 0
        for sid in singleton_s1_ids:
            pred_cands = cands_dict.get(sid, set())
            total_singleton_cands += len(pred_cands)
            if pred_cands:
                singletons_with_cands += 1

        avg_cands_singleton = total_singleton_cands / len(singleton_s1_ids) if singleton_s1_ids else 0.0
        false_cand_rate = singletons_with_cands / len(singleton_s1_ids) if singleton_s1_ids else 0.0

        # -- Overall --
        all_cand_counts = [len(cands_dict.get(sid, set())) for sid in sampled_ids]
        avg_cands_overall = np.mean(all_cand_counts) if all_cand_counts else 0.0
        median_cands = np.median(all_cand_counts) if all_cand_counts else 0.0
        p95_cands = np.percentile(all_cand_counts, 95) if all_cand_counts else 0.0
        max_cands = np.max(all_cand_counts) if all_cand_counts else 0.0
        reduction_ratio = 1 - (avg_cands_overall / total_s2s3) if total_s2s3 > 0 else 0.0

        f05_ceiling = compute_f05_ceiling(match_recall)

        print(f"\n--- Results: {exp['name']} ---")
        print(f"  MATCH-CONTAINING S1s ({len(match_s1_ids):,}):")
        print(f"    Blocking recall:     {match_recall:.4f} ({fully_recovered}/{len(match_s1_ids)})")
        print(f"    Avg candidates:      {avg_cands_match:.2f}")
        print(f"  SINGLETON S1s ({len(singleton_s1_ids):,}):")
        print(f"    False-candidate rate: {false_cand_rate:.4f} ({singletons_with_cands}/{len(singleton_s1_ids)})")
        print(f"    Avg candidates:      {avg_cands_singleton:.2f}")
        print(f"  OVERALL:")
        print(f"    Avg candidates:      {avg_cands_overall:.2f}")
        print(f"    Median candidates:   {median_cands:.0f}")
        print(f"    P95 candidates:      {p95_cands:.0f}")
        print(f"    Max candidates:      {max_cands:.0f}")
        print(f"    Reduction ratio:     {reduction_ratio:.8f}")
        print(f"  THEORETICAL F0.5 CEILING: {f05_ceiling:.4f}")
        print(f"    (Best possible F0.5 if classifier were perfect)")
        print(f"  Runtime: {runtime:.2f}s  |  Memory delta: {peak_delta:.0f} MB")

        # Check target
        if match_recall < 0.95:
            print(f"\n  ⚠️  WARNING: Blocking recall {match_recall:.4f} < 0.95 target.")
            print(f"     Fix blocking before proceeding to model training.")
        if avg_cands_overall > 50:
            print(f"\n  ⚠️  WARNING: Avg candidates {avg_cands_overall:.2f} > 50 target.")
            print(f"     Consider tighter pruning or reduced ann_max_distance.")


if __name__ == "__main__":
    run_experiment()
