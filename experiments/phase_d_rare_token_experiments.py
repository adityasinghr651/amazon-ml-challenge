"""
phase_d_rare_token_experiments.py - Phase D Rare Token Blocker Experiments

Evaluates:
  1. Token frequency distribution & empirical rarity thresholds
  2. Sensitivity sweep over max_doc_freq in [20, 50, 100, 200, 500]
  3. Sensitivity sweep over top_n_rare in [1, 2, 3]
  4. Cumulative Block A + B + C Union & Provenance tracking
  5. Persistent benchmark reporting
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
    ExactNameBlocker,
    AddressAnchorBlocker,
    RareTokenBlocker,
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
    print("PHASE D: RARE / INFORMATIVE TOKEN BLOCKER TOURNAMENT")
    print("=" * 75)

    s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
    target_pool = pd.concat([s2, s3], ignore_index=True)
    total_target = len(target_pool)
    records = []

    # =========================================================================
    # PART 1: MAX_DOC_FREQ SENSITIVITY SWEEP (DF in [20, 50, 100, 200, 500])
    # =========================================================================
    print("\n" + "=" * 70)
    print("SWEEP 1: Maximum Document Frequency Sensitivity (top_n_rare=2)")
    print("=" * 70)

    df_cutoffs = [20, 50, 100, 200, 500]
    best_df_cands = None
    best_df_blocker = None

    for df_cutoff in df_cutoffs:
        cfg = RareTokenConfig(max_doc_freq=df_cutoff, min_token_len=3, top_n_rare=2, max_bucket_size=500)
        blk = RareTokenBlocker(name=f"Block_C_DF_{df_cutoff}", config=cfg)
        cands, metrics, rec = run_blocker_eval(f"Block_C_RareToken_DF{df_cutoff}", blk, s1, target_pool, gt, total_target)
        records.append(rec)

        diag = blk.get_token_diagnostics()
        print(f"  Diagnostics: Unique tokens={diag['total_unique_tokens']:,} | Qualified rare={diag['qualified_rare_tokens']:,} | Bucket P95={diag['bucket_p95']:.1f} | Oversized={diag['oversized_token_count']}")

        if df_cutoff == 100:
            best_df_cands = cands
            best_df_blocker = blk

    # =========================================================================
    # PART 2: TOP_N_RARE SENSITIVITY SWEEP (top_n in [1, 2, 3], DF=100)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SWEEP 2: Number of Rare Tokens per Entity (DF=100)")
    print("=" * 70)

    for n_tokens in [1, 3]:
        cfg = RareTokenConfig(max_doc_freq=100, min_token_len=3, top_n_rare=n_tokens, max_bucket_size=500)
        blk = RareTokenBlocker(name=f"Block_C_TopN_{n_tokens}", config=cfg)
        cands, metrics, rec = run_blocker_eval(f"Block_C_RareToken_TopN{n_tokens}_DF100", blk, s1, target_pool, gt, total_target)
        records.append(rec)

    # =========================================================================
    # PART 3: CUMULATIVE UNION: BLOCK A + BLOCK B + BLOCK C
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Cumulative Union: Block A + Block B + Block C")
    print("=" * 70)

    # Generate Block A (Exact Name Full)
    cfg_all_a = NameBlockConfig(use_raw_clean=True, use_core_name=True, use_sorted_tokens=True)
    blk_a = ExactNameBlocker(name="Block_A_All_Names", config=cfg_all_a)
    blk_a.fit(target_pool)
    cands_a = blk_a.block(s1)

    # Generate Block B (Address Anchors Full)
    cfg_b = AddressBlockConfig(use_pin_only=False, use_pin_primary_num=True, use_pin_name_token=True, max_bucket_size=500)
    blk_b = AddressAnchorBlocker(name="Block_B_AddressAnchors", config=cfg_b)
    blk_b.fit(target_pool)
    cands_b = blk_b.block(s1)

    # Union Engine: A + B + C
    tracemalloc.start()
    t0 = time.time()

    union_engine = CandidateUnionEngine()
    union_engine.add_blocker_results("Block_A_ExactName", cands_a)
    union_engine.add_blocker_results("Block_B_AddressAnchors", cands_b)
    union_engine.add_blocker_results("Block_C_RareTokens", best_df_cands)

    cands_abc = union_engine.get_candidate_dict()
    union_time = time.time() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    union_mem = peak_mem / (1024 * 1024)

    m_abc = evaluate_candidate_pairs(cands_abc, gt, total_target)
    m_abc["runtime_s"] = union_time
    m_abc["memory_mb"] = union_mem
    print(format_evaluation_summary(m_abc, title="Cumulative Union: Block A + Block B + Block C"))

    r_abc = {
        "Experiment": "Cumulative_Block_A_plus_B_plus_C",
        "Active_Blockers": "Name(All)+Address(All)+RareToken(DF100_Top2)",
        "Pair_Recall": f"{m_abc['pair_recall']:.4f}",
        "Entity_Recovery": f"{m_abc['entity_recovery']:.4f}",
        "Total_Candidates": m_abc["total_candidates"],
        "Avg_Candidates": f"{m_abc['avg_candidates']:.2f}",
        "Median_Candidates": f"{m_abc['median_candidates']:.0f}",
        "P95_Candidates": f"{m_abc['p95_candidates']:.0f}",
        "P99_Candidates": f"{m_abc['p99_candidates']:.0f}",
        "Max_Candidates": m_abc["max_candidates"],
        "Reduction_Ratio": f"{m_abc['reduction_ratio']:.6f}",
        "Runtime_s": f"{union_time:.2f}",
        "Memory_MB": f"{union_mem:.1f}"
    }
    records.append(r_abc)

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

    # Inspect diagnostics sample
    diag_df = pd.read_csv(diag_path, sep="\t")
    print(f"\nDiagnostics Total Pairs: {len(diag_df):,}")
    print("Provenance Strategy Breakdown:")
    print(diag_df['strategies'].value_counts())

    # Append to persistent report
    rep_df = pd.DataFrame(records)
    rep_file = "reports/blocking_benchmark.tsv"
    existing_rep = pd.read_csv(rep_file, sep="\t") if os.path.exists(rep_file) else pd.DataFrame()
    final_rep = pd.concat([existing_rep, rep_df], ignore_index=True).drop_duplicates(subset=["Experiment"], keep="last")
    final_rep.to_csv(rep_file, sep="\t", index=False)
    print(f"\nAll Phase D results appended to {rep_file}")

if __name__ == "__main__":
    main()
