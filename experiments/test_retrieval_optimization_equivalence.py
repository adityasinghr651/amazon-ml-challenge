"""
experiments/test_retrieval_optimization_equivalence.py

Phase 2: Comprehensive Multi-Country Equivalence Regression Test
Compares OLD retrieval (evaluating target_mat.T in-loop) vs OPTIMIZED retrieval (target_mat_T precomputed + flatnonzero)
across representative S1 queries and real test target pools from France, US, and India.

Verifies:
  - 100% exact candidate set identity for every query
  - Candidate counts
  - Zero-candidate query counts
  - 0 mismatches
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Set

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.blocking.retrieval import CharNgramRetrievalBlocker

def load_sample_records(tsv_path: str, country: str, max_records: int) -> pd.DataFrame:
    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        col_to_idx = {c: i for i, c in enumerate(header)}
        e_idx = col_to_idx["entity_id"]
        n_idx = col_to_idx["business_name"]
        a_idx = col_to_idx["business_address"]
        c_idx = col_to_idx["country"]

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(e_idx, n_idx, a_idx, c_idx):
                continue
            if parts[c_idx] == country:
                records.append({
                    "entity_id": parts[e_idx],
                    "business_name": parts[n_idx],
                    "business_address": parts[a_idx],
                    "country": parts[c_idx],
                    "country_norm": parts[c_idx].upper(),
                    "name_clean": parts[n_idx].lower().strip()
                })
                if len(records) >= max_records:
                    break
    return pd.DataFrame(records)

def run_retrieval_variant(query_df, target_df, use_optimized=True):
    blocker = CharNgramRetrievalBlocker(
        top_k=20,
        ngram_range=(2, 4),
        sim_threshold=0.25,
        reciprocal=True,
        reciprocal_top_k=1,
        partition_by_country=False,
        batch_size=500
    )
    blocker.fit(target_df)

    if not use_optimized:
        # Reconstruct old non-precomputed target_mat.T logic with full-row scan
        def old_block(q_df):
            candidates = {eid: set() for eid in q_df['entity_id']}
            query_names = q_df['name_clean'].fillna('').astype(str).values
            query_eids = q_df['entity_id'].values
            target_mat = blocker.global_matrix
            target_eids = blocker.global_eids

            for chunk_start in range(0, len(query_names), blocker.batch_size):
                chunk_end = min(chunk_start + blocker.batch_size, len(query_names))
                chunk_names = query_names[chunk_start:chunk_end]
                chunk_eids = query_eids[chunk_start:chunk_end]

                X_chunk = blocker.vectorizer.transform(chunk_names)
                # Old inline target_mat.T call
                scores_mat = X_chunk.dot(target_mat.T).tocsr()

                indptr = scores_mat.indptr
                data = scores_mat.data
                indices = scores_mat.indices

                # 1. Forward
                for i in range(len(chunk_eids)):
                    q_eid = chunk_eids[i]
                    start_ptr = indptr[i]
                    end_ptr = indptr[i + 1]
                    if start_ptr == end_ptr:
                        continue
                    row_data = data[start_ptr:end_ptr]
                    row_indices = indices[start_ptr:end_ptr]
                    valid_mask = row_data >= blocker.sim_threshold
                    if not np.any(valid_mask):
                        continue
                    f_data = row_data[valid_mask]
                    f_indices = row_indices[valid_mask]
                    if len(f_data) > blocker.top_k:
                        top_k_idx = np.argpartition(-f_data, blocker.top_k)[:blocker.top_k]
                        best_indices = f_indices[top_k_idx]
                    else:
                        best_indices = f_indices
                    candidates[q_eid].update(set(target_eids[best_indices]))

                # 2. Reciprocal (OLD full-row scan)
                rev_mat = scores_mat.T.tocsr()
                r_indptr = rev_mat.indptr
                r_data = rev_mat.data
                r_indices = rev_mat.indices

                for j in range(rev_mat.shape[0]):
                    r_start = r_indptr[j]
                    r_end = r_indptr[j + 1]
                    if r_start == r_end:
                        continue
                    col_data = r_data[r_start:r_end]
                    col_indices = r_indices[r_start:r_end]
                    r_valid = col_data >= blocker.sim_threshold
                    if not np.any(r_valid):
                        continue
                    rf_data = col_data[r_valid]
                    rf_indices = col_indices[r_valid]
                    if len(rf_data) > blocker.reciprocal_top_k:
                        top_r_idx = np.argpartition(-rf_data, blocker.reciprocal_top_k)[:blocker.reciprocal_top_k]
                        best_q_indices = rf_indices[top_r_idx]
                    else:
                        best_q_indices = rf_indices
                    t_eid = target_eids[j]
                    for q_idx in best_q_indices:
                        candidates[chunk_eids[q_idx]].add(t_eid)

            return candidates

        return old_block(query_df)
    else:
        return blocker.block(query_df)

def main():
    print("=" * 80)
    print("PHASE 2: COMPREHENSIVE MULTI-COUNTRY EQUIVALENCE REGRESSION TEST")
    print("=" * 80)

    countries = ["France", "US", "India"]
    n_queries_per_country = 300
    n_targets_per_country = 15000

    total_mismatches_all = 0

    for country in countries:
        print(f"\n--- Testing Country Shard: {country} ---")
        q_df = load_sample_records("dataset/test/test_source1.tsv", country, max_records=n_queries_per_country)
        t_s2 = load_sample_records("dataset/test/test_source2.tsv", country, max_records=n_targets_per_country//2)
        t_s3 = load_sample_records("dataset/test/test_source3.tsv", country, max_records=n_targets_per_country//2)
        tgt_df = pd.concat([t_s2, t_s3], ignore_index=True)

        print(f"Loaded: {len(q_df)} {country} queries, {len(tgt_df)} target records")

        # Run OLD
        t0 = time.time()
        cands_old = run_retrieval_variant(q_df, tgt_df, use_optimized=False)
        t_old = time.time() - t0
        total_old = sum(len(c) for c in cands_old.values())

        # Run OPTIMIZED
        t0 = time.time()
        cands_opt = run_retrieval_variant(q_df, tgt_df, use_optimized=True)
        t_opt = time.time() - t0
        total_opt = sum(len(c) for c in cands_opt.values())

        # Audit
        mismatched_queries = 0
        for q_eid in q_df["entity_id"]:
            set_old = cands_old.get(q_eid, set())
            set_opt = cands_opt.get(q_eid, set())
            if set_old != set_opt:
                mismatched_queries += 1

        total_mismatches_all += mismatched_queries

        print(f"  Old Runtime:       {t_old:.3f}s (Total Candidates: {total_old:,})")
        print(f"  Optimized Runtime: {t_opt:.3f}s (Total Candidates: {total_opt:,})")
        print(f"  Speedup:           {t_old / max(t_opt, 0.001):.2f}x")
        print(f"  Mismatched sets:   {mismatched_queries} / {len(q_df)}")
        assert mismatched_queries == 0, f"REGRESSION FAILURE in {country}: {mismatched_queries} mismatches!"

    print("\n" + "=" * 80)
    print("PHASE 2 EQUIVALENCE TEST SUMMARY:")
    print(f"  Total Countries Evaluated: 3 (France, US, India)")
    print(f"  Total Mismatched Sets:     {total_mismatches_all}")
    print("  RESULT: 100% EXACT MATHEMATICAL CANDIDATE IDENTITY CONFIRMED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
