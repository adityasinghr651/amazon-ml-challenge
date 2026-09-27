"""
Phase 5 & 6: inference.py
End-to-end inference pipeline for test data.

Imports and calls the EXACT same functions used in training:
  - normalization.process_dataframe()
  - blocking.generate_candidates()
  - features.extract_features()
  - decision.predict_matches_two_band()

No reimplementation of any logic.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import psutil
from typing import Dict, Any, Optional

sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.io import load_test_data, save_matching_results, save_candidate_pairs
from src.business_entity_resolution.normalization import process_dataframe
from src.business_entity_resolution.blocking import generate_candidates
from src.business_entity_resolution.features import extract_features, extract_features_batch
from src.business_entity_resolution.decision import predict_matches_two_band
from src.business_entity_resolution.training import load_model
from src.business_entity_resolution.utils import load_config, get_blocking_config


def profile_on_sample(
    s1_norm: pd.DataFrame,
    s2_norm: pd.DataFrame,
    s3_norm: pd.DataFrame,
    blk_config: Dict[str, Any],
    sample_frac: float = 0.01,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Profile wall-clock time and memory on a 1% S1 sample,
    then extrapolate to full scale.

    Returns dict with profiling results.
    """
    all_s1_ids = s1_norm["entity_id"].unique()
    n_sample = max(1, int(len(all_s1_ids) * sample_frac))
    rng = np.random.RandomState(seed)
    sample_ids = set(rng.choice(all_s1_ids, size=n_sample, replace=False))
    s1_sample = s1_norm[s1_norm["entity_id"].isin(sample_ids)]

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()

    _ = generate_candidates(s1_sample, s2_norm, s3_norm, blk_config)

    elapsed = time.time() - t0
    mem_after = process.memory_info().rss / (1024 * 1024)

    scale_factor = len(all_s1_ids) / n_sample
    projected_time = elapsed * scale_factor
    projected_mem = (mem_after - mem_before) * scale_factor

    return {
        "sample_size": n_sample,
        "full_size": len(all_s1_ids),
        "sample_time_s": elapsed,
        "projected_time_s": projected_time,
        "projected_time_min": projected_time / 60,
        "mem_delta_mb": mem_after - mem_before,
        "projected_mem_mb": projected_mem,
    }


def run_inference(
    model_path: str,
    config_path: Optional[str] = None,
    t_high: float = 0.70,
    t_low: float = 0.40,
    output_dir: str = "output/test",
    skip_profiling: bool = False,
):
    """
    Full test inference pipeline:
      load test data → normalize → generate candidates → extract features →
      model.predict_proba → two-band threshold → write TSVs.

    Parameters
    ----------
    model_path : path to the serialized model (.pkl).
    config_path : optional override for config.yaml path.
    t_high, t_low : two-band decision thresholds.
    output_dir : directory to write output TSVs.
    skip_profiling : if True, skip the 1% profiling step.
    """
    cfg = load_config(config_path)
    blk_config = get_blocking_config(cfg)
    seed = cfg["random_state"]

    print("=" * 70)
    print("INFERENCE PIPELINE — Test Data")
    print("=" * 70)

    # 1. Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)

    # 2. Load test data
    print("Loading test data...")
    s1, s2, s3 = load_test_data()
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

    # 3. Normalize
    print("Normalizing...")
    t0 = time.time()
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    s3_norm = process_dataframe(s3)
    print(f"  Normalization: {time.time()-t0:.1f}s")

    # 4. Profile on 1% sample first
    if not skip_profiling:
        print("\nProfiling on 1% sample...")
        profile = profile_on_sample(s1_norm, s2_norm, s3_norm, blk_config, sample_frac=0.01, seed=seed)
        print(f"  Sample: {profile['sample_size']:,} S1 entities → {profile['sample_time_s']:.1f}s")
        print(f"  Projected full ({profile['full_size']:,} entities):")
        print(f"    Time: {profile['projected_time_min']:.1f} min")
        print(f"    Memory delta: {profile['projected_mem_mb']:.0f} MB")

        if profile["projected_time_min"] > 120:
            print(f"\n  ⚠️  WARNING: Projected runtime {profile['projected_time_min']:.0f} min > 2 hours.")
            print(f"     Consider reducing candidate volume before running full inference.")
            return
        if profile["projected_mem_mb"] > 16000:
            print(f"\n  ⚠️  WARNING: Projected memory {profile['projected_mem_mb']:.0f} MB > 16 GB.")
            print(f"     Consider reducing candidate volume before running full inference.")
            return

    # 5. Generate candidates (full)
    print("\nGenerating candidates (full test set)...")
    t0 = time.time()
    candidates_df = generate_candidates(s1_norm, s2_norm, s3_norm, blk_config)
    print(f"  Blocking: {time.time()-t0:.1f}s  |  {len(candidates_df)} S1 entities")

    # 6. Build record lookup
    print("Building record lookup...")
    records_db = {}
    for df in [s1_norm, s2_norm, s3_norm]:
        for _, r in df.iterrows():
            records_db[r["entity_id"]] = r.to_dict()

    # 7. Extract features + predict
    print("Extracting features and predicting...")
    t0 = time.time()

    all_s1_ids = set(s1["entity_id"].unique())
    results = []

    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        cand_str = str(row.get("candidate_entity_ids", "")).strip()
        if not cand_str:
            continue

        cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
        for c_id in cand_ids:
            if s1_id in records_db and c_id in records_db:
                feat = extract_features(records_db[s1_id], records_db[c_id])
                feature_names = sorted(feat.keys())
                results.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": c_id,
                    "features": [feat[k] for k in feature_names],
                })

    if results:
        X_test = np.array([r["features"] for r in results])
        probs = model.predict_proba(X_test)[:, 1]

        scores_df = pd.DataFrame([
            {"source1_entity_id": r["source1_entity_id"],
             "candidate_entity_id": r["candidate_entity_id"],
             "score": p}
            for r, p in zip(results, probs)
        ])
    else:
        scores_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    print(f"  Feature extraction + prediction: {time.time()-t0:.1f}s")

    # 8. Apply two-band decision
    print(f"\nApplying two-band decision (t_high={t_high}, t_low={t_low})...")
    predictions = predict_matches_two_band(scores_df, t_high=t_high, t_low=t_low)

    # 9. Write output files
    os.makedirs(output_dir, exist_ok=True)

    # matching_results.tsv — one row per S1 entity
    mr_rows = []
    for s1_id in s1["entity_id"].unique():
        matched = predictions.get(s1_id, [])
        mr_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched) if matched else "",
        })
    mr_df = pd.DataFrame(mr_rows)
    save_matching_results(mr_df, os.path.join(output_dir, "matching_results.tsv"))

    # candidate_pairs.tsv — one row per S1 entity
    save_candidate_pairs(candidates_df, os.path.join(output_dir, "candidate_pairs.tsv"))

    # Stats
    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = len(mr_df) - n_matched
    print(f"\n  Output written to {output_dir}/")
    print(f"  S1 entities: {len(mr_df):,}")
    print(f"  Matched: {n_matched:,}  |  Singletons: {n_singleton:,}")
    print("DONE.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run test inference")
    parser.add_argument("--model", required=True, help="Path to serialized model (.pkl)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--t-high", type=float, default=0.70, help="Upper threshold for matches")
    parser.add_argument("--t-low", type=float, default=0.40, help="Lower threshold for non-matches")
    parser.add_argument("--output-dir", default="output/test", help="Output directory")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip 1%% profiling step")
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        config_path=args.config,
        t_high=args.t_high,
        t_low=args.t_low,
        output_dir=args.output_dir,
        skip_profiling=args.skip_profiling,
    )
