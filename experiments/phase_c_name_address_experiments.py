"""
phase_c_name_address_experiments.py - Phase C Exact-Name & Address Blocker Experiments

Evaluates:
  1. Exact Name Blockers (A1: raw_clean, A2: core_name, A3: sorted_tokens, A1+A2+A3)
  2. Address Anchor Blockers (B1: PIN, B2: PIN+Num, B3: PIN+Token)
  3. Bucket size statistics and oversized bucket safeguards
  4. Incremental combination of Block A + Block B
  5. Provenance tracking and output verification via validate_submission.py
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
    ExactNameBlocker,
    AddressAnchorBlocker,
    CandidateUnionEngine,
    evaluate_candidate_pairs,
    format_evaluation_summary,
    build_or_load_benchmark
)

def run_experiment(exp_name: str, blocker, s1_df: pd.DataFrame, target_pool: pd.DataFrame, gt_df: pd.DataFrame, total_target: int):
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
    print("PHASE C: EXACT-NAME & ADDRESS ANCHOR BLOCKER TOURNAMENT")
    print("=" * 75)

    s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
    target_pool = pd.concat([s2, s3], ignore_index=True)
    total_target = len(target_pool)
    print(f"Target records pooled: {total_target:,} records ({len(s2):,} S2 + {len(s3):,} S3)")

    records = []

    # =========================================================================
    # PART 1: EXACT NAME INDEPENDENT & INCREMENTAL EXPERIMENTS
    # =========================================================================
    # A1: Raw Normalized Name alone
    cfg_a1 = NameBlockConfig(use_raw_clean=True, use_core_name=False, use_sorted_tokens=False)
    blk_a1 = ExactNameBlocker(name="Block_A1_RawClean", config=cfg_a1)
    cands_a1, m_a1, r_a1 = run_experiment("Block_A1_RawClean", blk_a1, s1, target_pool, gt, total_target)
    records.append(r_a1)

    # A2: Core Name (Suffix Stripped) alone
    cfg_a2 = NameBlockConfig(use_raw_clean=False, use_core_name=True, use_sorted_tokens=False)
    blk_a2 = ExactNameBlocker(name="Block_A2_CoreName", config=cfg_a2)
    cands_a2, m_a2, r_a2 = run_experiment("Block_A2_CoreName", blk_a2, s1, target_pool, gt, total_target)
    records.append(r_a2)

    # A3: Sorted Informative Tokens alone
    cfg_a3 = NameBlockConfig(use_raw_clean=False, use_core_name=False, use_sorted_tokens=True)
    blk_a3 = ExactNameBlocker(name="Block_A3_SortedTokens", config=cfg_a3)
    cands_a3, m_a3, r_a3 = run_experiment("Block_A3_SortedTokens", blk_a3, s1, target_pool, gt, total_target)
    records.append(r_a3)

    # A1 + A2 (Raw + Core)
    cfg_a12 = NameBlockConfig(use_raw_clean=True, use_core_name=True, use_sorted_tokens=False)
    blk_a12 = ExactNameBlocker(name="Block_A1_A2_Raw_Core", config=cfg_a12)
    cands_a12, m_a12, r_a12 = run_experiment("Block_A1_A2_Raw_Core", blk_a12, s1, target_pool, gt, total_target)
    records.append(r_a12)

    # Full Name Blocker: A1 + A2 + A3 (Raw + Core + Sorted)
    cfg_all_a = NameBlockConfig(use_raw_clean=True, use_core_name=True, use_sorted_tokens=True)
    blk_all_a = ExactNameBlocker(name="Block_A_All_Name_Representations", config=cfg_all_a)
    cands_all_a, m_all_a, r_all_a = run_experiment("Block_A_All_Name_Representations", blk_all_a, s1, target_pool, gt, total_target)
    records.append(r_all_a)

    # =========================================================================
    # PART 2: ADDRESS ANCHOR INDEPENDENT EXPERIMENTS & BUCKET AUDIT
    # =========================================================================
    # B1: Postal Code (PIN) alone (with max_bucket_size=500 cap)
    cfg_b1 = AddressBlockConfig(use_pin_only=True, use_pin_primary_num=False, use_pin_name_token=False, max_bucket_size=500)
    blk_b1 = AddressAnchorBlocker(name="Block_B1_PIN_Only", config=cfg_b1)
    cands_b1, m_b1, r_b1 = run_experiment("Block_B1_PIN_Only", blk_b1, s1, target_pool, gt, total_target)
    records.append(r_b1)
    print("  Bucket Diagnostics for B1 (PIN):", blk_b1.get_bucket_diagnostics())

    # B2: Postal Code + Primary Number
    cfg_b2 = AddressBlockConfig(use_pin_only=False, use_pin_primary_num=True, use_pin_name_token=False, max_bucket_size=500)
    blk_b2 = AddressAnchorBlocker(name="Block_B2_PIN_PrimaryNum", config=cfg_b2)
    cands_b2, m_b2, r_b2 = run_experiment("Block_B2_PIN_PrimaryNum", blk_b2, s1, target_pool, gt, total_target)
    records.append(r_b2)
    print("  Bucket Diagnostics for B2 (PIN+Num):", blk_b2.get_bucket_diagnostics())

    # B3: Postal Code + First Informative Name Token
    cfg_b3 = AddressBlockConfig(use_pin_only=False, use_pin_primary_num=False, use_pin_name_token=True, max_bucket_size=500)
    blk_b3 = AddressAnchorBlocker(name="Block_B3_PIN_NameToken", config=cfg_b3)
    cands_b3, m_b3, r_b3 = run_experiment("Block_B3_PIN_NameToken", blk_b3, s1, target_pool, gt, total_target)
    records.append(r_b3)
    print("  Bucket Diagnostics for B3 (PIN+Token):", blk_b3.get_bucket_diagnostics())

    # B2 + B3 Combined Address Blocker
    cfg_b23 = AddressBlockConfig(use_pin_only=False, use_pin_primary_num=True, use_pin_name_token=True, max_bucket_size=500)
    blk_b23 = AddressAnchorBlocker(name="Block_B_Combined_Anchors", config=cfg_b23)
    cands_b23, m_b23, r_b23 = run_experiment("Block_B_Combined_Anchors", blk_b23, s1, target_pool, gt, total_target)
    records.append(r_b23)

    # =========================================================================
    # PART 3: COMBINED BLOCK A + BLOCK B UNION (WITH PROVENANCE)
    # =========================================================================
    print("\n--- Running Cumulative Union: Block A + Block B ---")
    tracemalloc.start()
    t0 = time.time()

    union_engine = CandidateUnionEngine()
    union_engine.add_blocker_results("Block_A_ExactName", cands_all_a)
    union_engine.add_blocker_results("Block_B_AddressAnchors", cands_b23)

    combined_cands = union_engine.get_candidate_dict()
    union_time = time.time() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    union_mem = peak_mem / (1024 * 1024)

    m_ab = evaluate_candidate_pairs(combined_cands, gt, total_target)
    m_ab["runtime_s"] = union_time
    m_ab["memory_mb"] = union_mem
    print(format_evaluation_summary(m_ab, title="Cumulative: Block A (Exact Name) + Block B (Address Anchors)"))

    r_ab = {
        "Experiment": "Cumulative_Block_A_plus_B",
        "Active_Blockers": "Name(Clean+Core+Sorted)+Address(PINNum+PINTok)",
        "Pair_Recall": f"{m_ab['pair_recall']:.4f}",
        "Entity_Recovery": f"{m_ab['entity_recovery']:.4f}",
        "Total_Candidates": m_ab["total_candidates"],
        "Avg_Candidates": f"{m_ab['avg_candidates']:.2f}",
        "Median_Candidates": f"{m_ab['median_candidates']:.0f}",
        "P95_Candidates": f"{m_ab['p95_candidates']:.0f}",
        "P99_Candidates": f"{m_ab['p99_candidates']:.0f}",
        "Max_Candidates": m_ab["max_candidates"],
        "Reduction_Ratio": f"{m_ab['reduction_ratio']:.6f}",
        "Runtime_s": f"{union_time:.2f}",
        "Memory_MB": f"{union_mem:.1f}"
    }
    records.append(r_ab)

    # Export Dual Outputs: official and diagnostics
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
    print(f"\nExported official candidate pairs to {official_path}")
    print(f"Exported internal diagnostics to {diag_path}")

    # Inspect diagnostics sample
    diag_df = pd.read_csv(diag_path, sep="\t", nrows=10)
    print("\nDiagnostics Head:")
    print(diag_df)

    # Append to persistent report
    rep_df = pd.DataFrame(records)
    rep_file = "reports/blocking_benchmark.tsv"
    existing_rep = pd.read_csv(rep_file, sep="\t") if os.path.exists(rep_file) else pd.DataFrame()
    final_rep = pd.concat([existing_rep, rep_df], ignore_index=True).drop_duplicates(subset=["Experiment"], keep="last")
    final_rep.to_csv(rep_file, sep="\t", index=False)
    print(f"\nAll Phase C results appended to {rep_file}")

if __name__ == "__main__":
    main()
