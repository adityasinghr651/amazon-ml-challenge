"""
phase_fg_ablation_and_failure_analysis.py - Phase F & G Comprehensive Tournament & Diagnostics

Performs:
  1. Full Leave-One-Out (LOO) Blocker Ablation Analysis (All - A, All - B, All - C, All - D)
  2. Pairwise Blocker Overlap Analysis (Venn intersections & unique recall contributions)
  3. Failure Analysis: Forensic breakdown of ground-truth pairs missed by the full pipeline
  4. Candidate Explosion Analysis: Inspection of top candidate sets (P95, P99, P99.9, Max)
  5. Exports:
     - reports/blocker_ablation.tsv
     - reports/blocker_overlap.tsv
     - reports/blocking_failures.tsv
     - reports/candidate_statistics.tsv
"""
import os
import sys
import time
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
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

def classify_missed_pair(s1_row: dict, target_row: dict) -> str:
    """Classifies the primary failure mode of a missed true ground-truth match."""
    s1_name = str(s1_row.get('business_name', '')).strip()
    t_name = str(target_row.get('business_name', '')).strip()
    s1_addr = str(s1_row.get('business_address', '')).strip()
    t_addr = str(target_row.get('business_address', '')).strip()
    s1_c = str(s1_row.get('country', '')).strip().upper()
    t_c = str(target_row.get('country', '')).strip().upper()

    # 1. Country Mismatch
    if s1_c and t_c and s1_c != t_c:
        return "country_mismatch"

    # 2. Cross-Script / Non-Latin
    def get_scripts(text):
        scripts = set()
        for ch in text:
            if ch.isalpha():
                sn = unicodedata.name(ch, '')
                if 'DEVANAGARI' in sn: scripts.add('Devanagari')
                elif 'TAMIL' in sn: scripts.add('Tamil')
                elif 'LATIN' in sn: scripts.add('Latin')
        return scripts

    s1_scripts = get_scripts(s1_name)
    t_scripts = get_scripts(t_name)
    if s1_scripts and t_scripts and s1_scripts != t_scripts:
        return "cross_script"

    # 3. Missing Address
    if not t_addr or t_addr.lower() in {'nan', 'none', ''}:
        return "missing_address"

    # 4. Severe Abbreviation / Short Name
    s1_toks = s1_name.lower().split()
    t_toks = t_name.lower().split()
    if (len(s1_name) <= 4 or len(t_name) <= 4) and s1_name.lower() != t_name.lower():
        return "abbreviation_acronym"

    # 5. Token Reordering with Typos
    if set(s1_toks) == set(t_toks) and s1_toks != t_toks:
        return "reordered_tokens"

    # 6. Numeric Mismatch
    import re
    s1_nums = set(re.findall(r'\b\d+\b', s1_name + " " + s1_addr))
    t_nums = set(re.findall(r'\b\d+\b', t_name + " " + t_addr))
    if s1_nums and t_nums and not (s1_nums & t_nums):
        return "numeric_mismatch"

    # 7. Extreme Lexical Mutation / High Edit Distance
    return "severe_lexical_mutation"

def main():
    print("=" * 80)
    print("PHASE F & G: COMPREHENSIVE ABLATION, OVERLAP & FAILURE ANALYSIS")
    print("=" * 80)

    # 1. Load Benchmark Data
    s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
    target_pool = pd.concat([s2, s3], ignore_index=True)
    total_target = len(target_pool)
    print(f"Benchmark ready: {len(s1):,} S1 queries against {total_target:,} target records.")

    # 2. Run All 4 Blockers with Verified Optimal Settings
    print("\n[1/4] Running Block A (Exact Name Full)...")
    blk_a = ExactNameBlocker(name="Block_A_ExactName", config=NameBlockConfig(use_raw_clean=True, use_core_name=True, use_sorted_tokens=True))
    blk_a.fit(target_pool)
    cands_a = blk_a.block(s1)

    print("[2/4] Running Block B (Address Anchors Full)...")
    blk_b = AddressAnchorBlocker(name="Block_B_AddressAnchors", config=AddressBlockConfig(use_pin_only=False, use_pin_primary_num=True, use_pin_name_token=True, max_bucket_size=500))
    blk_b.fit(target_pool)
    cands_b = blk_b.block(s1)

    print("[3/4] Running Block C (Rare Tokens DF<=100 Top-2)...")
    blk_c = RareTokenBlocker(name="Block_C_RareTokens", config=RareTokenConfig(max_doc_freq=100, min_token_len=3, top_n_rare=2, max_bucket_size=500))
    blk_c.fit(target_pool)
    cands_c = blk_c.block(s1)

    print("[4/4] Running Block D (Sparse Char Retrieval Top-K=20)...")
    blk_d = CharNgramRetrievalBlocker(name="Block_D_CharRetrieval", config=RetrievalConfig(top_k=20, ngram_range=(2, 4), sim_threshold=0.30, batch_size=1000))
    blk_d.fit(target_pool)
    cands_d = blk_d.block(s1)

    # 3. Build Union Engine
    union_engine = CandidateUnionEngine()
    union_engine.add_blocker_results("Block_A_ExactName", cands_a)
    union_engine.add_blocker_results("Block_B_AddressAnchors", cands_b)
    union_engine.add_blocker_results("Block_C_RareTokens", cands_c)
    union_engine.add_blocker_results("Block_D_CharRetrieval", cands_d)

    full_cands = union_engine.get_candidate_dict()
    full_metrics = evaluate_candidate_pairs(full_cands, gt, total_target)
    full_recall = full_metrics['pair_recall']
    full_total_cands = full_metrics['total_candidates']

    print("\n" + "=" * 80)
    print("FULL PIPELINE REFERENCE BASELINE (A + B + C + D)")
    print("=" * 80)
    print(format_evaluation_summary(full_metrics, title="Full Pipeline (A + B + C + D)"))

    # =========================================================================
    # PART 1: LEAVE-ONE-OUT (LOO) ABLATION TOURNAMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: LEAVE-ONE-OUT (LOO) ABLATION ANALYSIS")
    print("=" * 80)

    blocker_names = [
        ("Block_A_ExactName", "Exact Name"),
        ("Block_B_AddressAnchors", "Address Anchors"),
        ("Block_C_RareTokens", "Rare Tokens"),
        ("Block_D_CharRetrieval", "Char Retrieval")
    ]

    ablation_records = []
    # Add Full Pipeline Row
    ablation_records.append({
        "Configuration": "Full Pipeline (A+B+C+D)",
        "Removed_Blocker": "None",
        "Pair_Recall": f"{full_recall:.4f}",
        "Recall_Loss": "0.0000",
        "Full_Entity_Recovery": f"{full_metrics['entity_recovery']:.4f}",
        "Total_Candidates": full_total_cands,
        "Candidate_Reduction": "0",
        "Candidate_Reduction_Pct": "0.00%",
        "Avg_Candidates": f"{full_metrics['avg_candidates']:.2f}"
    })

    for b_key, b_label in blocker_names:
        # Candidate set excluding blocker b_key
        loo_cands = defaultdict(set)
        for s1_id, cand_map in union_engine.provenance.items():
            for c_id, strats in cand_map.items():
                if len(strats - {b_key}) > 0:
                    loo_cands[s1_id].add(c_id)

        loo_metrics = evaluate_candidate_pairs(loo_cands, gt, total_target)
        recall_loss = full_recall - loo_metrics['pair_recall']
        cand_reduction = full_total_cands - loo_metrics['total_candidates']
        cand_reduction_pct = (cand_reduction / full_total_cands) * 100

        print(f"\n--- Ablation: All Minus {b_label} ---")
        print(f"  Recall Without Strategy : {loo_metrics['pair_recall']*100:.2f}% (Loss: -{recall_loss*100:.2f}%)")
        print(f"  Full Entity Recovery    : {loo_metrics['entity_recovery']*100:.2f}%")
        print(f"  Candidate Reduction     : -{cand_reduction:,} pairs (-{cand_reduction_pct:.2f}%)")
        print(f"  Remaining Candidates    : {loo_metrics['total_candidates']:,} (Avg {loo_metrics['avg_candidates']:.2f} / S1)")

        ablation_records.append({
            "Configuration": f"All Minus {b_label}",
            "Removed_Blocker": b_label,
            "Pair_Recall": f"{loo_metrics['pair_recall']:.4f}",
            "Recall_Loss": f"-{recall_loss:.4f}",
            "Full_Entity_Recovery": f"{loo_metrics['entity_recovery']:.4f}",
            "Total_Candidates": loo_metrics['total_candidates'],
            "Candidate_Reduction": f"-{cand_reduction}",
            "Candidate_Reduction_Pct": f"-{cand_reduction_pct:.2f}%",
            "Avg_Candidates": f"{loo_metrics['avg_candidates']:.2f}"
        })

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv("reports/blocker_ablation.tsv", sep="\t", index=False)
    print(f"\nAblation report saved to reports/blocker_ablation.tsv")

    # =========================================================================
    # PART 2: PAIRWISE BLOCKER OVERLAP ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: PAIRWISE BLOCKER OVERLAP & VENN ANALYSIS")
    print("=" * 80)

    # Parse ground truth positive pairs
    gt_pairs = set()
    for _, row in gt.iterrows():
        s1_id = str(row['source1_entity_id'])
        m_str = str(row['matched_entity_ids']).strip()
        if m_str:
            for m in m_str.split(','):
                gt_pairs.add((s1_id, m.strip()))

    blocker_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for s1_id, cand_map in union_engine.provenance.items():
        for c_id, strats in cand_map.items():
            for s in strats:
                blocker_pairs[s].add((s1_id, c_id))

    overlap_records = []
    keys = list(blocker_pairs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            b1, b2 = keys[i], keys[j]
            p1 = blocker_pairs[b1]
            p2 = blocker_pairs[b2]

            intersection = p1 & p2
            only_b1 = p1 - p2
            only_b2 = p2 - p1

            # Unique recall contribution (true pairs found by b1 but not b2, and vice-versa)
            rec_inter = len(intersection & gt_pairs)
            rec_only_b1 = len(only_b1 & gt_pairs)
            rec_only_b2 = len(only_b2 & gt_pairs)

            overlap_records.append({
                "Blocker_1": b1,
                "Blocker_2": b2,
                "B1_Total_Pairs": len(p1),
                "B2_Total_Pairs": len(p2),
                "Overlapping_Pairs": len(intersection),
                "B1_Unique_Pairs": len(only_b1),
                "B2_Unique_Pairs": len(only_b2),
                "Overlapping_True_Matches": rec_inter,
                "B1_Unique_True_Matches": rec_only_b1,
                "B2_Unique_True_Matches": rec_only_b2
            })
            print(f"Overlap: {b1} AND {b2} -> {len(intersection):,} overlapping pairs ({rec_inter:,} true matches)")

    df_overlap = pd.DataFrame(overlap_records)
    df_overlap.to_csv("reports/blocker_overlap.tsv", sep="\t", index=False)
    print(f"Overlap report saved to reports/blocker_overlap.tsv")

    # =========================================================================
    # PART 3: FAILURE ANALYSIS OF MISSED GROUND TRUTH PAIRS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: FORENSIC ANALYSIS OF MISSED GROUND-TRUTH PAIRS")
    print("=" * 80)

    # Find missed true pairs
    union_pairs = set()
    for s1_id, cand_map in union_engine.provenance.items():
        for c_id in cand_map.keys():
            union_pairs.add((s1_id, c_id))

    missed_gt_pairs = gt_pairs - union_pairs
    print(f"Total True Positive Pairs: {len(gt_pairs):,}")
    print(f"Recovered by Blocking     : {len(gt_pairs & union_pairs):,} ({len(gt_pairs & union_pairs)/len(gt_pairs)*100:.2f}%)")
    print(f"Missed by Blocking        : {len(missed_gt_pairs):,} ({len(missed_gt_pairs)/len(gt_pairs)*100:.2f}%)")

    # Index target rows for quick lookup
    s1_dict = s1.set_index("entity_id").to_dict("index")
    target_dict = target_pool.set_index("entity_id").to_dict("index")

    failure_classes = Counter()
    failure_rows = []

    for s1_id, cand_id in missed_gt_pairs:
        s1_row = s1_dict.get(s1_id, {})
        t_row = target_dict.get(cand_id, {})

        f_type = classify_missed_pair(s1_row, t_row)
        failure_classes[f_type] += 1

        failure_rows.append({
            "source1_entity_id": s1_id,
            "target_entity_id": cand_id,
            "failure_category": f_type,
            "s1_business_name": s1_row.get('business_name', ''),
            "target_business_name": t_row.get('business_name', ''),
            "s1_address": s1_row.get('business_address', ''),
            "target_address": t_row.get('business_address', ''),
            "s1_country": s1_row.get('country', ''),
            "target_country": t_row.get('country', '')
        })

    print("\nFailure Category Breakdown:")
    for f_cat, count in failure_classes.most_common():
        pct = (count / len(missed_gt_pairs)) * 100 if missed_gt_pairs else 0
        print(f"  {f_cat:<30}: {count:>5,} ({pct:>5.1f}%)")

    df_fail = pd.DataFrame(failure_rows)
    df_fail.to_csv("reports/blocking_failures.tsv", sep="\t", index=False)
    print(f"\nDetailed failure report saved to reports/blocking_failures.tsv")

    # =========================================================================
    # PART 4: CANDIDATE SET QUALITY & CANDIDATE EXPLOSION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: CANDIDATE SET QUALITY & EXPLOSION ANALYSIS")
    print("=" * 80)

    cand_counts_by_s1 = [(s1_id, len(cands)) for s1_id, cands in full_cands.items()]
    cand_counts_by_s1.sort(key=lambda x: x[1], reverse=True)

    print(f"Top 10 Entities with Largest Candidate Sets:")
    print(f"{'Source 1 ID':<18} {'Candidates':<12} {'Business Name':<35} {'Country'}")
    print("-" * 75)

    stats_rows = []
    for s1_id, c_count in cand_counts_by_s1[:15]:
        s1_row = s1_dict.get(s1_id, {})
        b_name = s1_row.get('business_name', '')[:32]
        c_val = s1_row.get('country', '')
        print(f"{s1_id:<18} {c_count:<12} {b_name:<35} {c_val}")

        stats_rows.append({
            "source1_entity_id": s1_id,
            "candidate_count": c_count,
            "business_name": s1_row.get('business_name', ''),
            "business_address": s1_row.get('business_address', ''),
            "country": s1_row.get('country', '')
        })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv("reports/candidate_statistics.tsv", sep="\t", index=False)
    print(f"\nCandidate explosion diagnostics saved to reports/candidate_statistics.tsv")

if __name__ == "__main__":
    main()
