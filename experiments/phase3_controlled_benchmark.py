"""
experiments/phase3_controlled_benchmark.py

Phase 3: Controlled Performance Benchmark
Measures:
  - 1,000 France S1 queries against 1.43M France target records
  - 1,000 US S1 queries against 3.82M US target records
  - 1,000 India S1 queries against 4.72M India target records

Records:
  - Normalization time
  - TF-IDF construction time
  - Retrieval time
  - Total runtime
  - Queries / sec throughput
  - Candidates / query
  - Peak RAM
"""

import os
import sys
import gc
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Set, List
from anyascii import anyascii

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.business_entity_resolution.normalization import normalize_name, normalize_address
from src.business_entity_resolution.blocking.config import RareTokenConfig
from src.business_entity_resolution.blocking.exact_name import ExactNameBlocker
from src.business_entity_resolution.blocking.address import AddressAnchorBlocker
from src.business_entity_resolution.blocking.rare_token import RareTokenBlocker
from src.business_entity_resolution.blocking.retrieval import CharNgramRetrievalBlocker
from src.business_entity_resolution.blocking.cross_script import TransliterationBlocker
from src.business_entity_resolution.blocking.union import CandidateUnionEngine

def get_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def fast_normalize_df(df: pd.DataFrame) -> pd.DataFrame:
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
    print("PHASE 3: CONTROLLED MULTI-COUNTRY PERFORMANCE BENCHMARK")
    print("=" * 80)

    countries = ["France", "US", "India"]
    n_queries = 1000
    batch_size = 500

    peak_ram = get_ram_mb()
    benchmark_results = []

    for country in countries:
        print(f"\n============================================================")
        print(f"BENCHMARKING SHARD: {country} (Current RAM: {get_ram_mb():.1f} MB)")
        print(f"============================================================")

        t_c_start = time.time()

        # 1. Load Queries
        print(f"  Loading {n_queries} S1 queries for {country}...")
        s1_raw = load_country_records_from_tsv("dataset/test/test_source1.tsv", country, max_records=n_queries)
        s1_df = fast_normalize_df(pd.DataFrame(s1_raw))
        print(f"  Loaded & normalized {len(s1_df)} S1 queries.")

        # 2. Load Targets
        t0 = time.time()
        print(f"  Loading full target records for {country} (test_source2 + test_source3)...")
        t_s2 = load_country_records_from_tsv("dataset/test/test_source2.tsv", country)
        t_s3 = load_country_records_from_tsv("dataset/test/test_source3.tsv", country)
        target_raw = t_s2 + t_s3
        del t_s2, t_s3
        gc.collect()
        t_load = time.time() - t0
        print(f"  Loaded {len(target_raw):,} target records in {t_load:.2f}s (RAM: {get_ram_mb():.1f} MB)")
        peak_ram = max(peak_ram, get_ram_mb())

        # 3. Normalization
        t0 = time.time()
        target_df = fast_normalize_df(pd.DataFrame(target_raw))
        del target_raw
        gc.collect()
        t_norm = time.time() - t0
        print(f"  Normalized {len(target_df):,} targets in {t_norm:.2f}s (RAM: {get_ram_mb():.1f} MB)")
        peak_ram = max(peak_ram, get_ram_mb())

        # 4. TF-IDF Construction
        t0 = time.time()
        print(f"  Fitting CharNgramRetrievalBlocker (TF-IDF construction + precomputing target_mat_T)...")
        b_retrieval = CharNgramRetrievalBlocker(
            top_k=20,
            ngram_range=(2, 4),
            sim_threshold=0.25,
            reciprocal=True,
            reciprocal_top_k=1,
            partition_by_country=False,
            batch_size=batch_size
        )
        b_retrieval.fit(target_df)
        t_tfidf_fit = time.time() - t0
        print(f"  TF-IDF fit + transpose precomputation: {t_tfidf_fit:.2f}s (RAM: {get_ram_mb():.1f} MB)")
        peak_ram = max(peak_ram, get_ram_mb())

        # 5. Retrieval Query Execution
        t0 = time.time()
        print(f"  Executing retrieval queries (batch_size={batch_size})...")
        cands_retrieval = b_retrieval.block(s1_df)
        t_retrieval_query = time.time() - t0
        throughput = len(s1_df) / max(t_retrieval_query, 0.001)
        print(f"  Query retrieval finished in {t_retrieval_query:.2f}s! ({throughput:.2f} queries/sec)")

        # Candidate stats
        c_counts = [len(cands_retrieval[eid]) for eid in s1_df['entity_id']]
        avg_c = np.mean(c_counts)
        p95_c = np.percentile(c_counts, 95)
        p99_c = np.percentile(c_counts, 99)
        max_c = np.max(c_counts)
        zero_c = sum(1 for c in c_counts if c == 0)

        total_country_time = time.time() - t_c_start

        row = {
            "Country": country,
            "Target Pool": len(target_df),
            "Queries": len(s1_df),
            "Norm Time (s)": round(t_norm, 2),
            "TF-IDF Construction (s)": round(t_tfidf_fit, 2),
            "Query Retrieval (s)": round(t_retrieval_query, 2),
            "Total Shard Time (s)": round(total_country_time, 2),
            "Retrieval Throughput (q/s)": round(throughput, 2),
            "Avg Candidates": round(avg_c, 2),
            "P95 Candidates": round(p95_c, 1),
            "P99 Candidates": round(p99_c, 1),
            "Max Candidates": int(max_c),
            "Zero-Candidate Queries": zero_c,
            "Peak RAM (MB)": round(peak_ram, 1)
        }
        benchmark_results.append(row)

        del target_df, b_retrieval, s1_df, cands_retrieval
        gc.collect()

    res_df = pd.DataFrame(benchmark_results)
    print("\n" + "=" * 80)
    print("PHASE 3 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(res_df.to_string(index=False))

    # Extrapolations for Full Test Population (1,732,544 queries)
    # India: 809,986 S1s, US: 663,106 S1s, France: 259,452 S1s
    test_counts = {"India": 809986, "US": 663106, "France": 259452}
    print("\n" + "=" * 80)
    print("EXTRAPOLATION TO FULL 1,732,544 TEST S1 POPULATION")
    print("=" * 80)

    total_extrap_query_time = 0
    total_extrap_setup_time = 0

    for r in benchmark_results:
        c = r["Country"]
        q_count = test_counts[c]
        setup_time = r["Norm Time (s)"] + r["TF-IDF Construction (s)"]
        query_time = (r["Query Retrieval (s)"] / r["Queries"]) * q_count
        total_time = setup_time + query_time
        total_extrap_query_time += query_time
        total_extrap_setup_time += setup_time
        print(f"  {c:7s}: {q_count:,} queries -> Setup: {setup_time/60:.1f}m | Retrieval: {query_time/60:.1f}m | Total: {total_time/60:.1f}m ({total_time/3600:.2f}h)")

    grand_total_time = total_extrap_setup_time + total_extrap_query_time
    print(f"\n  Projected Full Run Time: {grand_total_time/60:.1f} minutes ({grand_total_time/3600:.2f} hours)")
    print(f"  Remaining Deadline:      ~6.0 hours")
    if grand_total_time/3600 < 6.0:
        print(f"  FEASIBILITY STATUS:      FEASIBLE (Completes with {6.0 - grand_total_time/3600:.2f} hours margin)")
    else:
        print(f"  FEASIBILITY STATUS:      NEEDS FURTHER PARALLELIZATION (Exceeds deadline)")

if __name__ == "__main__":
    main()
