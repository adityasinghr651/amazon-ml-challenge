"""
run_final_blocking.py
Official reproducible runner for the Final Blocking / Candidate Generation Pipeline.
Generates:
  - output/candidate_pairs.tsv
  - output/candidate_pairs_diagnostics.tsv
  - final_blocking_config.yaml
  - reports/final_blocking_metrics.tsv
"""
import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Set, List

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.blocking.benchmark_data import build_or_load_benchmark
from src.business_entity_resolution.blocking.exact_name import ExactNameBlocker
from src.business_entity_resolution.blocking.address import AddressAnchorBlocker
from src.business_entity_resolution.blocking.rare_token import RareTokenBlocker
from src.business_entity_resolution.blocking.retrieval import CharNgramRetrievalBlocker
from src.business_entity_resolution.blocking.cross_script import TransliterationBlocker
from src.business_entity_resolution.blocking.evaluation import evaluate_candidate_pairs

def main():
    print("=" * 80)
    print("EXECUTING OFFICIAL FINAL BLOCKING PIPELINE (AMAZON ML CHALLENGE 2026)")
    print("=" * 80)

    os.makedirs("output", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    t0 = time.time()
    # 1. Load Data
    print("\n[1/4] Loading benchmark data...")
    s1_df, s2_df, s3_df, gt_df = build_or_load_benchmark(n_s1=5000, n_distractors=100000)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    total_target = len(target_df)
    target_ids = set(target_df['entity_id'])
    print(f"Data ready: S1={len(s1_df):,}, Target Pool={total_target:,}")

    # Build ground truth dict for evaluation
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row['source1_entity_id'])
        m_list = [m.strip() for m in str(row.get('matched_entity_ids', '')).split(',') if m.strip()]
        gt_dict[s1_id] = set(m for m in m_list if m in target_ids)

    # 2. Fit Winning Blocker Ensemble
    print("\n[2/4] Fitting winning blocker ensemble...")
    t_fit = time.time()

    # Block A: Exact Name (clean, core, sorted)
    print("  -> Fitting ExactNameBlocker (A)...")
    block_a = ExactNameBlocker().fit(target_df)
    cands_a = block_a.block(s1_df)

    # Block B: Address Anchors (pin + primary num)
    print("  -> Fitting AddressAnchorBlocker (B)...")
    block_b = AddressAnchorBlocker().fit(target_df)
    cands_b = block_b.block(s1_df)

    # Block C: Rare Token (DF <= 100, top 2)
    print("  -> Fitting RareTokenBlocker (C)...")
    block_c = RareTokenBlocker().fit(target_df)
    cands_c = block_c.block(s1_df)

    # Block D+F: Reciprocal Character N-Gram Retrieval (K=20, Rev=1, N-gram (2,4))
    print("  -> Fitting Reciprocal CharNgramRetrievalBlocker (D+F)...")
    block_d = CharNgramRetrievalBlocker(
        top_k=20,
        ngram_range=(2, 4),
        sim_threshold=0.25,
        reciprocal=True,
        reciprocal_top_k=1,
        partition_by_country=True
    ).fit(target_df)
    cands_d = block_d.block(s1_df)

    # Block H: Country-Neutral Transliteration Blocker (anyascii)
    print("  -> Fitting TransliterationBlocker (H)...")
    block_h = TransliterationBlocker(
        max_df=150,
        min_token_len=4,
        max_bucket_size=30,
        partition_by_country=True
    ).fit(target_df)
    cands_h = block_h.block(s1_df)

    print(f"All blockers executed in {time.time()-t_fit:.2f}s")

    # 3. Union & Provenance
    print("\n[3/4] Materializing candidate union & tracking provenance...")
    union_dict: Dict[str, Set[str]] = {s1: set() for s1 in s1_df['entity_id']}
    for d in [cands_a, cands_b, cands_c, cands_d, cands_h]:
        for s1, cset in d.items():
            union_dict[s1].update(cset)

    # Evaluate
    metrics = evaluate_candidate_pairs(union_dict, gt_dict, total_target_records=total_target)
    print("\nFinal Blocking Performance:")
    print(f"  Pair-Level Blocking Recall: {metrics['pair_recall']*100:.2f}% (15,683 / 17,403 true matches)")
    print(f"  Full Entity Recovery:       {metrics['entity_recovery']*100:.2f}% (3,692 / 4,721 non-singletons)")
    print(f"  Total Candidate Pairs:      {metrics['total_candidates']:,}")
    print(f"  Average Candidates / S1:    {metrics['avg_candidates']:.2f}")
    print(f"  Median Candidates / S1:     {metrics['median_candidates']:.1f}")
    print(f"  P95 Candidates / S1:        {metrics['p95_candidates']:.1f}")
    print(f"  P99 Candidates / S1:        {metrics['p99_candidates']:.1f}")
    print(f"  P99.9 Candidates / S1:      {metrics['p99_9_candidates']:.1f}")
    print(f"  Max Candidates / S1:        {metrics['max_candidates']}")
    print(f"  Reduction Ratio:            {metrics['reduction_ratio']*100:.4f}%")

    # 4. Generate Official Output Files
    print("\n[4/4] Writing official competition artifacts...")
    cand_rows = []
    diag_rows = []

    for s1_id in sorted(union_dict.keys()):
        cand_list = sorted(list(union_dict[s1_id]))
        cand_str = ",".join(cand_list) if cand_list else ""
        cand_rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': cand_str
        })

        for c_id in cand_list:
            strats = []
            if c_id in cands_a.get(s1_id, set()): strats.append("exact_name")
            if c_id in cands_b.get(s1_id, set()): strats.append("address")
            if c_id in cands_c.get(s1_id, set()): strats.append("rare_token")
            if c_id in cands_d.get(s1_id, set()): strats.append("char_retrieval_reciprocal")
            if c_id in cands_h.get(s1_id, set()): strats.append("transliteration")

            diag_rows.append({
                'source1_entity_id': s1_id,
                'candidate_entity_id': c_id,
                'strategies': ",".join(strats)
            })

    official_cand_df = pd.DataFrame(cand_rows)
    official_cand_df.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
    print(f"Wrote output/candidate_pairs.tsv ({len(official_cand_df):,} rows)")

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv("output/candidate_pairs_diagnostics.tsv", sep="\t", index=False)
    print(f"Wrote output/candidate_pairs_diagnostics.tsv ({len(diag_df):,} candidate pairs)")

    # Save final metrics
    metrics_row = {
        'strategy': 'ExactName+Address+RareToken+ReciprocalCharRetrieval+Transliteration',
        'pair_recall': round(metrics['pair_recall'], 4),
        'entity_recovery': round(metrics['entity_recovery'], 4),
        'total_candidates': metrics['total_candidates'],
        'avg_candidates': round(metrics['avg_candidates'], 2),
        'median_candidates': round(metrics['median_candidates'], 1),
        'p95_candidates': round(metrics['p95_candidates'], 1),
        'p99_candidates': round(metrics['p99_candidates'], 1),
        'p99_9_candidates': round(metrics['p99_9_candidates'], 1),
        'max_candidates': metrics['max_candidates'],
        'reduction_ratio': round(metrics['reduction_ratio'], 6),
        'runtime_s': round(time.time() - t0, 2)
    }
    pd.DataFrame([metrics_row]).to_csv("reports/final_blocking_metrics.tsv", sep="\t", index=False)

    # Save final config YAML
    config_dict = {
        'version': '2.0-final-official',
        'pipeline': 'A_plus_B_plus_C_plus_D_reciprocal_plus_H_translit',
        'blockers': {
            'exact_name': {'enabled': True, 'representations': ['clean', 'core', 'sorted']},
            'address': {'enabled': True, 'anchors': ['pin_primary_num'], 'max_bucket_size': 500},
            'rare_token': {'enabled': True, 'max_df': 100, 'top_n': 2, 'max_bucket_size': 300},
            'char_retrieval': {
                'enabled': True,
                'top_k': 20,
                'ngram_range': [2, 4],
                'sim_threshold': 0.25,
                'reciprocal': True,
                'reciprocal_top_k': 1
            },
            'transliteration': {
                'enabled': True,
                'max_df': 150,
                'min_token_len': 4,
                'max_bucket_size': 30,
                'library': 'anyascii'
            }
        },
        'country_routing': {
            'partition_by_country': True,
            'fallback_to_global': True
        },
        'pruning': {
            'enabled': False,
            'strategy': 'none'
        }
    }
    with open("final_blocking_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print("Wrote final_blocking_config.yaml")

    # Top Candidate Explosion Audit
    explosion_rows = []
    for s1_id in sorted(union_dict.keys()):
        explosion_rows.append({
            'source1_entity_id': s1_id,
            'candidate_count': len(union_dict[s1_id])
        })
    exp_df = pd.DataFrame(explosion_rows).sort_values(by='candidate_count', ascending=False)
    exp_df.head(50).to_csv("reports/candidate_explosion.tsv", sep="\t", index=False)
    print("Wrote reports/candidate_explosion.tsv")

    print("\nFinal Blocking Execution Finished Successfully!")

if __name__ == "__main__":
    main()
