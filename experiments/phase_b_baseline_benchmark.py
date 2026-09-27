"""
phase_b_baseline_benchmark.py - Phase B Baseline Evaluation Benchmark

Measures the existing baseline candidates against ground truth using the new
evaluation framework:
  1. Baseline 1: baseline_pipeline.py (country token Jaccard intersection)
  2. Baseline 2: blocking_legacy.py (A+B+C+D with default top-20 pruning)
"""
import os
import sys
import time
from typing import Tuple, Dict, Set
import tracemalloc
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.blocking.benchmark_data import build_or_load_benchmark
from src.business_entity_resolution.blocking.evaluation import evaluate_candidate_pairs, format_evaluation_summary
import src.business_entity_resolution.blocking_legacy as legacy

def run_baseline_1_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], float]:
    """Runs baseline_pipeline.py token intersection candidate generation."""
    t0 = time.time()
    countries = sorted(list(set(s1_df["country_norm"].unique()) | 
                            set(s2_df["country_norm"].unique()) | 
                            set(s3_df["country_norm"].unique())))
    
    candidate_pairs: Dict[str, Set[str]] = defaultdict(set)
    chunk_size = 200

    for country in countries:
        c_s1 = s1_df[s1_df["country_norm"] == country]
        c_s2 = s2_df[s2_df["country_norm"] == country]
        c_s3 = s3_df[s3_df["country_norm"] == country]

        if len(c_s1) == 0 or (len(c_s2) == 0 and len(c_s3) == 0):
            continue

        vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
        vectorizer.fit(c_s1["name_clean"])

        X_s1 = vectorizer.transform(c_s1["name_clean"])
        X_s2 = vectorizer.transform(c_s2["name_clean"]) if len(c_s2) > 0 else None
        X_s3 = vectorizer.transform(c_s3["name_clean"]) if len(c_s3) > 0 else None

        s1_eids = c_s1["entity_id"].values
        s2_eids = c_s2["entity_id"].values if len(c_s2) > 0 else np.array([])
        s3_eids = c_s3["entity_id"].values if len(c_s3) > 0 else np.array([])

        for start in range(0, X_s1.shape[0], chunk_size):
            end = min(start + chunk_size, X_s1.shape[0])
            X_s1_chunk = X_s1[start:end]
            eids_chunk = s1_eids[start:end]

            # Match against S2
            if X_s2 is not None and X_s2.shape[0] > 0:
                intersections_s2 = (X_s1_chunk.toarray() @ X_s2.T)
                for i in range(X_s1_chunk.shape[0]):
                    cand_idx = np.where(intersections_s2[i] > 0)[0]
                    candidate_pairs[eids_chunk[i]].update(s2_eids[cand_idx])

            # Match against S3
            if X_s3 is not None and X_s3.shape[0] > 0:
                intersections_s3 = (X_s1_chunk.toarray() @ X_s3.T)
                for i in range(X_s1_chunk.shape[0]):
                    cand_idx = np.where(intersections_s3[i] > 0)[0]
                    candidate_pairs[eids_chunk[i]].update(s3_eids[cand_idx])

    runtime = time.time() - t0
    return candidate_pairs, runtime

def run_baseline_2_legacy(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], float]:
    """Runs legacy blocking.py A+B+C+D pipeline with top-20 pruning."""
    t0 = time.time()
    config = {
        "active_blocks": ["A", "B", "C", "D"],
        "rare_token_percentile": 25,
        "ann_neighbors": 10,
        "ann_max_distance": 0.5,
        "enable_pruning": True,
        "max_candidates_before_pruning": 50,
        "prune_top_k": 20
    }
    cands_df = legacy.generate_candidates(s1_df, s2_df, s3_df, config)
    candidate_dict = {}
    for _, row in cands_df.iterrows():
        c_str = str(row['candidate_entity_ids'])
        c_list = [c.strip() for c in c_str.split(',') if c.strip()]
        candidate_dict[row['source1_entity_id']] = set(c_list)
    runtime = time.time() - t0
    return candidate_dict, runtime

def main():
    print("=" * 70)
    print("PHASE B: BASELINE MEASUREMENT & EVALUATION HARNESS AUDIT")
    print("=" * 70)

    # 1. Load / Build Benchmark
    s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
    total_target = len(s2) + len(s3)

    os.makedirs("reports", exist_ok=True)
    report_file = "reports/blocking_benchmark.tsv"
    records = []

    # 2. Evaluate Baseline 1
    print("\n--- Evaluating Baseline 1 (CountVectorizer Jaccard Token Intersections) ---")
    tracemalloc.start()
    b1_cands, b1_time = run_baseline_1_blocking(s1, s2, s3)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb_b1 = peak_mem / (1024 * 1024)

    b1_metrics = evaluate_candidate_pairs(b1_cands, gt, total_target)
    b1_metrics["runtime_s"] = b1_time
    b1_metrics["memory_mb"] = mem_mb_b1

    print(format_evaluation_summary(b1_metrics, title="Baseline 1: Token Intersection"))
    print(f" Runtime: {b1_time:.2f}s | Peak Memory: {mem_mb_b1:.1f} MB")

    records.append({
        "Experiment": "Baseline_1_Token_Intersection",
        "Active_Blockers": "Country_Token_Jaccard",
        "Pair_Recall": f"{b1_metrics['pair_recall']:.4f}",
        "Entity_Recovery": f"{b1_metrics['entity_recovery']:.4f}",
        "Total_Candidates": b1_metrics["total_candidates"],
        "Avg_Candidates": f"{b1_metrics['avg_candidates']:.2f}",
        "Median_Candidates": f"{b1_metrics['median_candidates']:.0f}",
        "P95_Candidates": f"{b1_metrics['p95_candidates']:.0f}",
        "P99_Candidates": f"{b1_metrics['p99_candidates']:.0f}",
        "Max_Candidates": b1_metrics["max_candidates"],
        "Reduction_Ratio": f"{b1_metrics['reduction_ratio']:.6f}",
        "Runtime_s": f"{b1_time:.2f}",
        "Memory_MB": f"{mem_mb_b1:.1f}"
    })

    # 3. Evaluate Baseline 2 (Legacy A+B+C+D with pruning)
    print("\n--- Evaluating Baseline 2 (Legacy A+B+C+D with Top-20 Pruning) ---")
    tracemalloc.start()
    b2_cands, b2_time = run_baseline_2_legacy(s1, s2, s3)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb_b2 = peak_mem / (1024 * 1024)

    b2_metrics = evaluate_candidate_pairs(b2_cands, gt, total_target)
    b2_metrics["runtime_s"] = b2_time
    b2_metrics["memory_mb"] = mem_mb_b2

    print(format_evaluation_summary(b2_metrics, title="Baseline 2: Legacy A+B+C+D (Pruned)"))
    print(f" Runtime: {b2_time:.2f}s | Peak Memory: {mem_mb_b2:.1f} MB")

    records.append({
        "Experiment": "Baseline_2_Legacy_ABCD_Pruned",
        "Active_Blockers": "Name_PINNum_RareTok_CharANN_Prune20",
        "Pair_Recall": f"{b2_metrics['pair_recall']:.4f}",
        "Entity_Recovery": f"{b2_metrics['entity_recovery']:.4f}",
        "Total_Candidates": b2_metrics["total_candidates"],
        "Avg_Candidates": f"{b2_metrics['avg_candidates']:.2f}",
        "Median_Candidates": f"{b2_metrics['median_candidates']:.0f}",
        "P95_Candidates": f"{b2_metrics['p95_candidates']:.0f}",
        "P99_Candidates": f"{b2_metrics['p99_candidates']:.0f}",
        "Max_Candidates": b2_metrics["max_candidates"],
        "Reduction_Ratio": f"{b2_metrics['reduction_ratio']:.6f}",
        "Runtime_s": f"{b2_time:.2f}",
        "Memory_MB": f"{b2_metrics['memory_mb']:.1f}"
    })

    # 4. Save to reports
    df_rep = pd.DataFrame(records)
    df_rep.to_csv(report_file, sep="\t", index=False)
    print(f"\nBaseline benchmark recorded to {report_file}")

if __name__ == "__main__":
    main()
