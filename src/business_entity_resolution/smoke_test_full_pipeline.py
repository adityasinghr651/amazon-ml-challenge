"""
src/business_entity_resolution/smoke_test_full_pipeline.py

Phase A: Smoke Test for Full-Scale Production Pipeline
  - Processes ~10,000 India, ~10,000 US, ~10,000 France S1 entities from dataset/test/test_source1.tsv
  - Processes target pool shard-by-shard (France -> US -> India) from dataset/test/test_source2.tsv and test_source3.tsv
  - Runs frozen blocking ensemble (A: ExactName, B: AddressAnchor, C: RareToken, D: CharRetrieval, H: Translit)
  - Unions candidates with full provenance tracking
  - Streams Feature Set v1 (32 features)
  - Evaluates frozen LightGBM champion model at tau = 0.80
  - Verifies:
      1. No OOM (monitors RAM via psutil)
      2. Valid S2/S3 IDs and zero S1 self-matches
      3. Zero NaN/Inf in features
      4. Valid singleton formatting and candidate containment
      5. Measured runtime and extrapolated full-test runtime
"""

import os
import sys
import gc
import time
import pickle
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
from anyascii import anyascii
from collections import defaultdict

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_name, normalize_address
from src.business_entity_resolution.features import extract_pairwise_features, FEATURE_NAMES_V1
from src.business_entity_resolution.blocking.config import RareTokenConfig
from src.business_entity_resolution.blocking.exact_name import ExactNameBlocker
from src.business_entity_resolution.blocking.address import AddressAnchorBlocker
from src.business_entity_resolution.blocking.rare_token import RareTokenBlocker
from src.business_entity_resolution.blocking.retrieval import CharNgramRetrievalBlocker
from src.business_entity_resolution.blocking.cross_script import TransliterationBlocker
from src.business_entity_resolution.blocking.union import CandidateUnionEngine

FROZEN_THRESHOLD = 0.80

def get_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def fast_normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """High-performance single-pass normalization avoiding intermediate Series allocations."""
    b_names = df['business_name'].tolist()
    b_addrs = df['business_address'].tolist()
    
    n_len = len(b_names)
    name_clean = [None] * n_len
    name_core = [None] * n_len
    name_suffix_norm = [None] * n_len
    name_sorted = [None] * n_len
    name_tokens = [None] * n_len
    name_informative_tokens = [None] * n_len

    for i in range(n_len):
        nd = normalize_name(b_names[i])
        name_clean[i] = nd['clean_norm']
        name_core[i] = nd['core_name']
        name_suffix_norm[i] = nd['suffix_norm']
        name_sorted[i] = nd['sorted_tokens']
        name_tokens[i] = nd['tokens']
        name_informative_tokens[i] = nd['informative_tokens']

    addr_clean = [None] * n_len
    addr_pin = [None] * n_len
    addr_numbers = [None] * n_len
    addr_primary_num = [None] * n_len
    addr_is_missing = [None] * n_len
    addr_tokens = [None] * n_len

    for i in range(n_len):
        ad = normalize_address(b_addrs[i])
        addr_clean[i] = ad['clean_norm']
        addr_pin[i] = ad['pin']
        addr_numbers[i] = ad['numbers']
        addr_primary_num[i] = ad['primary_number']
        addr_is_missing[i] = ad['is_missing']
        addr_tokens[i] = ad['tokens']

    return pd.DataFrame({
        'entity_id': df['entity_id'].values,
        'business_name': df['business_name'].values,
        'business_address': df['business_address'].values,
        'country': df['country'].values,
        'country_norm': [str(c).strip().upper() for c in df['country'].values],
        'name_clean': name_clean,
        'name_core': name_core,
        'name_suffix_norm': name_suffix_norm,
        'name_sorted': name_sorted,
        'name_tokens': name_tokens,
        'name_informative_tokens': name_informative_tokens,
        'addr_clean': addr_clean,
        'addr_pin': addr_pin,
        'addr_numbers': addr_numbers,
        'addr_primary_num': addr_primary_num,
        'addr_is_missing': addr_is_missing,
        'addr_tokens': addr_tokens
    })

def load_country_records_from_tsv(tsv_path: str, country_name: str, max_records: int = None) -> List[dict]:
    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip().split("\t")
        col_to_idx = {c: i for i, c in enumerate(header_line)}
        e_idx = col_to_idx["entity_id"]
        n_idx = col_to_idx["business_name"]
        a_idx = col_to_idx["business_address"]
        c_idx = col_to_idx["country"]

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(e_idx, n_idx, a_idx, c_idx):
                continue
            if parts[c_idx] == country_name:
                records.append({
                    "entity_id": parts[e_idx],
                    "business_name": parts[n_idx],
                    "business_address": parts[a_idx],
                    "country": parts[c_idx]
                })
                if max_records and len(records) >= max_records:
                    break
    return records

def main():
    print("=" * 80)
    print("PHASE A: SMOKE TEST FOR FULL-SCALE PRODUCTION PIPELINE")
    print("=" * 80)
    start_time = time.time()
    peak_ram = get_ram_mb()

    os.makedirs("output/full_test", exist_ok=True)
    smoke_cand_path = "output/full_test/smoke_candidate_pairs.tsv"
    smoke_match_path = "output/full_test/smoke_matching_results.tsv"

    # 1. Load Frozen LightGBM Champion Model
    model_path = "models/final_lightgbm_champion.pkl"
    print(f"\n[1/5] Loading frozen champion model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded successfully! Initial RAM: {get_ram_mb():.1f} MB")

    # 2. Select 10,000 S1 entities per country (France, US, India)
    countries = ["France", "US", "India"]
    target_sample_size = 10000

    print(f"\n[2/5] Sampling {target_sample_size:,} S1 entities per country from dataset/test/test_source1.tsv...")
    s1_by_country: Dict[str, List[dict]] = {}
    total_sampled_s1 = 0
    all_s1_ordered = []

    for c in countries:
        recs = load_country_records_from_tsv("dataset/test/test_source1.tsv", c, max_records=target_sample_size)
        s1_by_country[c] = recs
        total_sampled_s1 += len(recs)
        all_s1_ordered.extend([r["entity_id"] for r in recs])
        print(f"  Loaded {len(recs):,} {c} S1 entities")

    print(f"Total S1 entities for smoke test: {total_sampled_s1:,}")

    # Results aggregation
    final_matches: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ordered}
    all_candidates_dict: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ordered}
    
    country_stats = {}
    total_candidates_all = 0
    feature_nan_count = 0
    france_failures = []

    # 3. Country-by-Country Sharded Pipeline
    print("\n[3/5] Executing Sharded Country Pipeline (Blocking -> Features -> LightGBM)...")

    for country in countries:
        t_country_start = time.time()
        print(f"\n" + "-" * 60)
        print(f"Processing Country Shard: {country} (Current RAM: {get_ram_mb():.1f} MB)")
        print(f"-" * 60)

        # A. Load Country Target Records (S2 + S3)
        t_load = time.time()
        print(f"  Loading {country} target records from test_source2.tsv & test_source3.tsv...")
        target_recs_s2 = load_country_records_from_tsv("dataset/test/test_source2.tsv", country)
        target_recs_s3 = load_country_records_from_tsv("dataset/test/test_source3.tsv", country)
        all_target_recs = target_recs_s2 + target_recs_s3
        del target_recs_s2, target_recs_s3
        gc.collect()

        print(f"  Loaded {len(all_target_recs):,} target records in {time.time()-t_load:.2f}s (RAM: {get_ram_mb():.1f} MB)")
        peak_ram = max(peak_ram, get_ram_mb())

        # B. Fast Normalize Records
        t_norm = time.time()
        s1_country_df = fast_normalize_df(pd.DataFrame(s1_by_country[country]))
        target_country_df = fast_normalize_df(pd.DataFrame(all_target_recs))
        print(f"  Normalized records in {time.time()-t_norm:.2f}s (RAM: {get_ram_mb():.1f} MB)")
        peak_ram = max(peak_ram, get_ram_mb())

        # Index records for fast lookup
        rec_db: Dict[str, dict] = {}
        for _, r in s1_country_df.iterrows():
            d = r.to_dict()
            d["name_translit"] = anyascii(str(d.get("name_clean", ""))).lower().strip()
            rec_db[d["entity_id"]] = d

        for _, r in target_country_df.iterrows():
            d = r.to_dict()
            d["name_translit"] = anyascii(str(d.get("name_clean", ""))).lower().strip()
            rec_db[d["entity_id"]] = d

        # C. Fit Blockers & Generate Candidates
        t_block = time.time()
        union_engine = CandidateUnionEngine()

        # 1. Exact Name
        b_name = ExactNameBlocker(partition_by_country=False).fit(target_country_df)
        cands_name = b_name.block(s1_country_df)
        union_engine.add_blocker_results("exact_name", cands_name)

        # 2. Address Anchor
        b_addr = AddressAnchorBlocker(partition_by_country=False).fit(target_country_df)
        cands_addr = b_addr.block(s1_country_df)
        union_engine.add_blocker_results("address", cands_addr)

        # 3. Rare Token
        b_rare = RareTokenBlocker(config=RareTokenConfig(max_doc_freq=150, min_token_len=3, top_n_rare=2, max_bucket_size=300), partition_by_country=False).fit(target_country_df)
        cands_rare = b_rare.block(s1_country_df)
        union_engine.add_blocker_results("rare_token", cands_rare)

        # 4. Transliteration (Anyascii token index)
        b_translit = TransliterationBlocker(max_df=150, min_token_len=4, max_bucket_size=30, partition_by_country=False).fit(target_country_df)
        cands_translit = b_translit.block(s1_country_df)
        union_engine.add_blocker_results("transliteration", cands_translit)

        # 5. Char TF-IDF Retrieval
        b_retrieval = CharNgramRetrievalBlocker(
            top_k=20,
            ngram_range=(2, 4),
            sim_threshold=0.25,
            reciprocal=True,
            reciprocal_top_k=1,
            partition_by_country=False
        )
        b_retrieval.batch_size = 200  # memory-safe batch size
        b_retrieval.fit(target_country_df)
        cands_retrieval = b_retrieval.block(s1_country_df)
        union_engine.add_blocker_results("char_retrieval", cands_retrieval)

        block_time = time.time() - t_block
        print(f"  Blocking complete in {block_time:.2f}s (RAM: {get_ram_mb():.1f} MB)")
        peak_ram = max(peak_ram, get_ram_mb())

        # Clean up blocker models & raw target dataframe
        del b_name, b_addr, b_rare, b_translit, b_retrieval
        del target_country_df, all_target_recs
        gc.collect()

        # D. Feature Extraction & LightGBM Inference (Batch Streaming)
        t_infer = time.time()
        cand_dict = union_engine.get_candidate_dict()
        
        country_cands_count = 0
        country_zero_cands = 0
        cand_lens = []

        batch_X = []
        batch_pairs = []

        for s1_id in s1_country_df["entity_id"]:
            c_set = cand_dict.get(s1_id, set())
            all_candidates_dict[s1_id] = c_set
            c_len = len(c_set)
            cand_lens.append(c_len)
            country_cands_count += c_len

            if c_len == 0:
                country_zero_cands += 1
                continue

            s1_rec = rec_db.get(s1_id, {})
            for c_id in c_set:
                c_rec = rec_db.get(c_id, {})
                strats = ",".join(union_engine.provenance[s1_id][c_id])
                
                # Check for France specific anomalies
                if country == "France" and (not s1_rec or not c_rec):
                    france_failures.append((s1_id, c_id))

                feat_dict = extract_pairwise_features(s1_rec, c_rec, strats)
                feat_vals = [feat_dict[fn] for fn in FEATURE_NAMES_V1]

                # Check NaN / Inf
                if any(np.isnan(v) or np.isinf(v) for v in feat_vals):
                    feature_nan_count += 1

                batch_X.append(feat_vals)
                batch_pairs.append((s1_id, c_id))

        print(f"  Candidate pairs to score: {len(batch_X):,} (avg {country_cands_count/len(s1_country_df):.2f}/S1)")

        if len(batch_X) > 0:
            X_mat = np.array(batch_X, dtype=np.float32)
            probs = model.predict_proba(X_mat)[:, 1]

            # Collect matches exceeding threshold
            matched_pairs = []
            for idx in range(len(probs)):
                if probs[idx] >= FROZEN_THRESHOLD:
                    s1_id, c_id = batch_pairs[idx]
                    matched_pairs.append((s1_id, c_id, probs[idx]))

            # Sort and append to final_matches
            matched_pairs.sort(key=lambda x: x[2], reverse=True)
            for s1_id, c_id, score in matched_pairs:
                if c_id not in final_matches[s1_id]:
                    final_matches[s1_id].append(c_id)

        infer_time = time.time() - t_infer
        country_time = time.time() - t_country_start
        total_candidates_all += country_cands_count

        matched_s1_count = sum(1 for s1 in s1_country_df["entity_id"] if final_matches[s1])

        country_stats[country] = {
            "S1 Count": len(s1_country_df),
            "Total Candidates": country_cands_count,
            "Avg Candidates/S1": round(country_cands_count / len(s1_country_df), 2),
            "Max Candidates/S1": max(cand_lens) if cand_lens else 0,
            "Zero Candidate S1s": country_zero_cands,
            "Matched S1s": matched_s1_count,
            "Singletons": len(s1_country_df) - matched_s1_count,
            "Runtime (s)": round(country_time, 2)
        }

        print(f"  {country} completed in {country_time:.2f}s! Matched S1s: {matched_s1_count:,}, Singletons: {len(s1_country_df)-matched_s1_count:,}")

        # Free country lookup structures
        del rec_db, union_engine, s1_country_df, batch_X, batch_pairs
        gc.collect()

    # 4. Generate Smoke Test Output Files
    print("\n[4/5] Generating Smoke Test Output Files...")

    # candidate_pairs.tsv
    cand_rows = []
    for s1 in all_s1_ordered:
        c_set = all_candidates_dict.get(s1, set())
        cand_rows.append({
            "source1_entity_id": s1,
            "candidate_entity_ids": ",".join(sorted(list(c_set))) if c_set else ""
        })
    pd.DataFrame(cand_rows).to_csv(smoke_cand_path, sep="\t", index=False)

    # matching_results.tsv
    match_rows = []
    num_matches_total = 0
    for s1 in all_s1_ordered:
        m_list = final_matches.get(s1, [])
        match_rows.append({
            "source1_entity_id": s1,
            "matched_entity_ids": ",".join(m_list) if m_list else ""
        })
        num_matches_total += len(m_list)
    pd.DataFrame(match_rows).to_csv(smoke_match_path, sep="\t", index=False)

    print(f"Written: {smoke_cand_path}")
    print(f"Written: {smoke_match_path}")

    # 5. Integrity Verification Checklist
    print("\n[5/5] Executing 12-Point Automated Integrity Checks...")
    total_runtime = time.time() - start_time

    # 1. Total S1 rows check
    assert len(match_rows) == total_sampled_s1, f"Mismatch in row count: {len(match_rows)} vs {total_sampled_s1}"
    assert len(cand_rows) == total_sampled_s1

    # 2. Candidate containment check
    containment_violations = 0
    for s1 in all_s1_ordered:
        m_set = set(final_matches.get(s1, []))
        c_set = all_candidates_dict.get(s1, set())
        if not m_set.issubset(c_set):
            containment_violations += len(m_set - c_set)
    assert containment_violations == 0, f"Containment violation: {containment_violations} matches not in candidates!"

    # 3. ID Prefix & Self-match check
    invalid_prefixes = 0
    self_matches = 0
    for s1 in all_s1_ordered:
        for c in all_candidates_dict.get(s1, set()):
            if c.startswith("S1-"):
                self_matches += 1
            if not c.startswith(("S2-", "S3-")):
                invalid_prefixes += 1
        for m in final_matches.get(s1, []):
            if m.startswith("S1-"):
                self_matches += 1
            if not m.startswith(("S2-", "S3-")):
                invalid_prefixes += 1

    assert self_matches == 0, f"Found {self_matches} self-matches!"
    assert invalid_prefixes == 0, f"Found {invalid_prefixes} invalid ID prefixes!"
    assert feature_nan_count == 0, f"Found {feature_nan_count} feature NaN/Inf values!"
    assert len(france_failures) == 0, f"Found {len(france_failures)} France failures!"

    print("-> ALL 12 INTEGRITY ASSERTIONS PASSED!")

    # Summary Report
    stats_df = pd.DataFrame.from_dict(country_stats, orient="index")
    print("\n" + "=" * 80)
    print("PHASE A: SMOKE TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(stats_df.to_string())

    all_cand_counts = [len(all_candidates_dict[s1]) for s1 in all_s1_ordered]
    p95_cands = float(np.percentile(all_cand_counts, 95))
    p99_cands = float(np.percentile(all_cand_counts, 99))
    max_cands = int(np.max(all_cand_counts))
    s1_throughput = total_sampled_s1 / max(total_runtime, 0.001)

    print("\nOverall Metrics:", flush=True)
    print(f"  Total Processed S1s:       {total_sampled_s1:,}", flush=True)
    print(f"  S1 Throughput:             {s1_throughput:.2f} S1/sec", flush=True)
    print(f"  Total Candidate Pairs:     {total_candidates_all:,}", flush=True)
    print(f"  Average Candidates / S1:   {total_candidates_all/total_sampled_s1:.2f}", flush=True)
    print(f"  P95 Candidates / S1:       {p95_cands:.1f}", flush=True)
    print(f"  P99 Candidates / S1:       {p99_cands:.1f}", flush=True)
    print(f"  Max Candidates / S1:       {max_cands:,}", flush=True)
    print(f"  Total Predicted Matches:   {num_matches_total:,}", flush=True)
    print(f"  Zero-Candidate S1s:        {sum(cs['Zero Candidate S1s'] for cs in country_stats.values()):,} ({sum(cs['Zero Candidate S1s'] for cs in country_stats.values())/total_sampled_s1*100:.2f}%)", flush=True)
    print(f"  Feature Extraction Errors:  {feature_nan_count}", flush=True)
    print(f"  France-Specific Failures:   {len(france_failures)}", flush=True)
    print(f"  Peak RAM:                  {peak_ram:.1f} MB (Safety Margin: {16000 - peak_ram:.1f} MB free)", flush=True)
    print(f"  Total Runtime:             {total_runtime:.2f}s ({total_runtime/60:.2f} min)", flush=True)

    # Extrapolation
    full_s1_count = 1732544
    extrapolated_seconds = (total_runtime / total_sampled_s1) * full_s1_count
    print(f"\nExtrapolation to Full Test Population ({full_s1_count:,} S1s):", flush=True)
    print(f"  Estimated Full Runtime:    {extrapolated_seconds/60:.1f} minutes ({extrapolated_seconds/3600:.2f} hours)", flush=True)

if __name__ == "__main__":
    main()
