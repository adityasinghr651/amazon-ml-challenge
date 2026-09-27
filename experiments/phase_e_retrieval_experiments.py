"""
phase_e_retrieval_experiments.py - Phase E Character N-Gram Retrieval Experiments

Evaluates:
  1. Top-K sweep over K in [5, 10, 20, 50]
  2. N-gram range sweep: (2, 4) vs (3, 5)
  3. Similarity threshold sweep: 0.25 vs 0.35
  4. Cumulative Union: Block A + Block B + Block C + Block D
  5. Provenance tracking and persistent benchmark reporting
"""
import os
import sys
import time
import tracemalloc
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.blocking import (
    NameBlockConfig,
    AddressBlockConfig,
    RareTokenConfig,
    RetrievalConfig,
    ExactNameBlocker,
    AddressAnchorBlocker,
    RareTokenBlocker,
    CharNgramRetrievalBlocker,
    CandidateUnionEngine,
    evaluate_candidate_pairs,
    format_evaluation_summary,
    build_or_load_benchmark
)

def run_blocker_eval(exp_name: str, blocker, s1_df: pd.DataFrame, target_pool: pd.DataFrame, gt_df: pd.DataFrame, total_target: int):
    print(f"\n--- Running Experiment: {exp_name} ---")
    tracemalloc.start()
    t0 = time.time()
    
    blocker.fit(target_pool)
    fit_time = time.time() - t0

    t1 = time.time()
    cands = blocker.block(s1_df)
    query_time = time.time() - t1
    total_time = fit_time + query_time

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = peak_mem / (1024 * 1024)

    metrics = evaluate_candidate_pairs(cands, gt_df, total_target)
    metrics["runtime_s"] = total_time
    metrics["fit_time_s"] = fit_time
    metrics["query_time_s"] = query_time
    metrics["memory_mb"] = mem_mb

    print(format_evaluation_summary(metrics, title=exp_name))
    print(f" Latency: Fit={fit_time:.2f}s | Query={query_time:.2f}s | Total={total_time:.2f}s | Peak RAM: {mem_mb:.1f} MB")

    record = {
        "Experiment": exp_name,
        "Active_Blockers": blocker.name,
        "Pair_Recall": f"{metrics['pair_recall']:.4f}",
        "Entity_Recovery": f"{metrics['entity_recovery']:.4f}",
        "Total_Candidates": metrics["total_candidates"],
        "Avg_Candidates": f"{metrics['avg_candidates']:.2f}",
        "Median_Candidates": f"{metrics['median_candidates']:.0f}",
        "P95_Candidates": f"{metrics['p95_candidates']:.0f}",
        "P99_Candidates": f"{metrics['p99_candidates']:.0f}",
        "Max_Candidates": metrics["max_candidates"],
        "Reduction_Ratio": f"{metrics['reduction_ratio']:.6f}",
        "Runtime_s": f"{total_time:.2f}",
        "Memory_MB": f"{mem_mb:.1f}"
    }
    return cands, metrics, record

def main():
    print("=" * 75)
    print("PHASE E: SPARSE CHARACTER N-GRAM RETRIEVAL TOURNAMENT")
    print("=" * 75)

    s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
    target_pool = pd.concat([s2, s3], ignore_index=True)
    total_target = len(target_pool)
    records = []

    # =========================================================================
    # PART 1: TOP-K SENSITIVITY SWEEP (K in [5, 10, 20, 50], ngram=(2,4), sim=0.30)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SWEEP 1: Top-K Sensitivity (ngram=(2,4), sim=0.30)")
    print("=" * 70)

    k_values = [5, 10, 20, 50]
    best_retrieval_cands = None
    best_k = 20

    for k in k_values:
        cfg = RetrievalConfig(top_k=k, ngram_range=(2, 4), sim_threshold=0.30, batch_size=1000)
        blk = CharNgramRetrievalBlocker(name=f"Block_D_TopK_{k}", config=cfg)
        cands, metrics, rec = run_blocker_eval(f"Block_D_Retrieval_TopK_{k}", blk, s1, target_pool, gt, total_target)
        records.append(rec)

        if k == best_k:
            best_retrieval_cands = cands

    # =========================================================================
    # PART 2: N-GRAM RANGE & SIMILARITY THRESHOLD SWEEPS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SWEEP 2: N-Gram Range & Similarity Threshold (K=20)")
    print("=" * 70)

    # N-Gram Range (3, 5)
    cfg_35 = RetrievalConfig(top_k=20, ngram_range=(3, 5), sim_threshold=0.30, batch_size=1000)
    blk_35 = CharNgramRetrievalBlocker(name="Block_D_Ngram_3_5", config=cfg_35)
    cands_35, m_35, r_35 = run_blocker_eval("Block_D_Retrieval_Ngram_3_5", blk_35, s1, target_pool, gt, total_target)
    records.append(r_35)

    # Similarity Threshold 0.25 (looser)
    cfg_sim25 = RetrievalConfig(top_k=20, ngram_range=(2, 4), sim_threshold=0.25, batch_size=1000)
    blk_sim25 = CharNgramRetrievalBlocker(name="Block_D_Sim_025", config=cfg_sim25)
    cands_s25, m_s25, r_s25 = run_blocker_eval("Block_D_Retrieval_Sim_025", blk_sim25, s1, target_pool, gt, total_target)
    records.append(r_s25)

    # =========================================================================
    # PART 3: CUMULATIVE UNION: BLOCK A + BLOCK B + BLOCK C + BLOCK D
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Cumulative Union: Block A + Block B + Block C + Block D")
    print("=" * 70)

    # Block A
    cfg_a = NameBlockConfig(use_raw_clean=True, use_core_name=True, use_sorted_tokens=True)
    blk_a = ExactNameBlocker(name="Block_A", config=cfg_a)
    blk_a.fit(target_pool)
    cands_a = blk_a.block(s1)

    # Block B
    cfg_b = AddressBlockConfig(use_pin_only=False, use_pin_primary_num=True, use_pin_name_token=True, max_bucket_size=500)
    blk_b = AddressAnchorBlocker(name="Block_B", config=cfg_b)
    blk_b.fit(target_pool)
    cands_b = blk_b.block(s1)

    # Block C
    cfg_c = RareTokenConfig(max_doc_freq=100, min_token_len=3, top_n_rare=2, max_bucket_size=500)
    blk_c = RareTokenBlocker(name="Block_C", config=cfg_c)
    blk_c.fit(target_pool)
    cands_c = blk_c.block(s1)

    # Union Engine: A + B + C + D
    tracemalloc.start()
    t0 = time.time()

    union_engine = CandidateUnionEngine()
    union_engine.add_blocker_results("Block_A_ExactName", cands_a)
    union_engine.add_blocker_results("Block_B_AddressAnchors", cands_b)
    union_engine.add_blocker_results("Block_C_RareTokens", cands_c)
    union_engine.add_blocker_results("Block_D_CharRetrieval", best_retrieval_cands)

    cands_abcd = union_engine.get_candidate_dict()
    union_time = time.time() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    union_mem = peak_mem / (1024 * 1024)

    m_abcd = evaluate_candidate_pairs(cands_abcd, gt, total_target)
    m_abcd["runtime_s"] = union_time
    m_abcd["memory_mb"] = union_mem
    print(format_evaluation_summary(m_abcd, title="Cumulative Union: Full Baseline Pipeline (A + B + C + D)"))

    r_abcd = {
        "Experiment": "Cumulative_Block_A_plus_B_plus_C_plus_D",
        "Active_Blockers": "Name(All)+Address(All)+RareToken(DF100)+CharRetrieval(TopK20)",
        "Pair_Recall": f"{m_abcd['pair_recall']:.4f}",
        "Entity_Recovery": f"{m_abcd['entity_recovery']:.4f}",
        "Total_Candidates": m_abcd["total_candidates"],
        "Avg_Candidates": f"{m_abcd['avg_candidates']:.2f}",
        "Median_Candidates": f"{m_abcd['median_candidates']:.0f}",
        "P95_Candidates": f"{m_abcd['p95_candidates']:.0f}",
        "P99_Candidates": f"{m_abcd['p99_candidates']:.0f}",
        "Max_Candidates": m_abcd["max_candidates"],
        "Reduction_Ratio": f"{m_abcd['reduction_ratio']:.6f}",
        "Runtime_s": f"{union_time:.2f}",
        "Memory_MB": f"{union_mem:.1f}"
    }
    records.append(r_abcd)

    # Export Dual Outputs
    s1_names = dict(zip(s1['entity_id'], s1['name_clean']))
    cand_names = dict(zip(target_pool['entity_id'], target_pool['name_clean']))

    official_path = "output/candidate_pairs.tsv"
    diag_path = "output/candidate_pairs_diagnostics.tsv"
    union_engine.export_outputs(
        official_path=official_path,
        diagnostics_path=diag_path,
        all_s1_ids=list(s1['entity_id']),
        s1_names=s1_names,
        cand_names=cand_names
    )
    print(f"\nExported updated official candidate pairs to {official_path}")
    print(f"Exported updated internal diagnostics to {diag_path}")

    # Inspect diagnostics
    diag_df = pd.read_csv(diag_path, sep="\t")
    print(f"\nDiagnostics Total Pairs: {len(diag_df):,}")
    print("Top 10 Provenance Combinations:")
    print(diag_df['strategies'].value_counts().head(10))

    # Append to persistent report
    rep_df = pd.DataFrame(records)
    rep_file = "reports/blocking_benchmark.tsv"
    existing_rep = pd.read_csv(rep_file, sep="\t") if os.path.exists(rep_file) else pd.DataFrame()
    final_rep = pd.concat([existing_rep, rep_df], ignore_index=True).drop_duplicates(subset=["Experiment"], keep="last")
    final_rep.to_csv(rep_file, sep="\t", index=False)
    print(f"\nAll Phase E results appended to {rep_file}")

if __name__ == "__main__":
    main()
