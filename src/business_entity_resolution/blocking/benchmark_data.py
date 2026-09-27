"""
benchmark_data.py - Representative & Leak-Free Validation Dataset Generator

Constructs a reproducible, stratified validation split containing:
  - N representative S1 entities with true singleton & country proportions
  - 100% of all true S2 and S3 ground-truth matches for these S1 entities
  - A substantial pool of distractor records from S2 and S3 (e.g. 100,000+ records)
  - Pre-normalized representations cached for lightning-fast experiment iteration
"""
import os
import time
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Set, Optional

from src.business_entity_resolution.io import get_dataset_dir, load_tsv
from src.business_entity_resolution.normalization import process_dataframe

BENCHMARK_CACHE_DIR = "dataset/benchmark"

def build_or_load_benchmark(
    n_s1: int = 5000,
    n_distractors: int = 100000,
    random_seed: int = 42,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds or loads a cached, stratified validation benchmark dataset.

    Returns:
        (s1_val, s2_pool, s3_pool, gt_val)
    """
    os.makedirs(BENCHMARK_CACHE_DIR, exist_ok=True)
    s1_cache = os.path.join(BENCHMARK_CACHE_DIR, f"s1_benchmark_{n_s1}.tsv")
    s2_cache = os.path.join(BENCHMARK_CACHE_DIR, f"s2_benchmark_{n_s1}_{n_distractors}.tsv")
    s3_cache = os.path.join(BENCHMARK_CACHE_DIR, f"s3_benchmark_{n_s1}_{n_distractors}.tsv")
    gt_cache = os.path.join(BENCHMARK_CACHE_DIR, f"gt_benchmark_{n_s1}.tsv")

    if not force_rebuild and all(os.path.exists(p) for p in [s1_cache, s2_cache, s3_cache, gt_cache]):
        print(f"Loading cached benchmark from {BENCHMARK_CACHE_DIR}...")
        s1 = pd.read_csv(s1_cache, sep="\t", dtype=str, keep_default_na=False)
        s2 = pd.read_csv(s2_cache, sep="\t", dtype=str, keep_default_na=False)
        s3 = pd.read_csv(s3_cache, sep="\t", dtype=str, keep_default_na=False)
        gt = pd.read_csv(gt_cache, sep="\t", dtype=str, keep_default_na=False)

        def deserialize_lists(df: pd.DataFrame) -> pd.DataFrame:
            for col in ['name_tokens', 'name_informative_tokens', 'addr_tokens', 'addr_numbers']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: str(x).split() if x else [])
            return df

        s1 = deserialize_lists(s1)
        s2 = deserialize_lists(s2)
        s3 = deserialize_lists(s3)
        print(f"Benchmark loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}, GT={len(gt):,}")
        return s1, s2, s3, gt

    print(f"Building new stratified benchmark (S1={n_s1:,}, Distractors={n_distractors:,})...")
    d_dir = get_dataset_dir()
    train_dir = os.path.join(d_dir, "train")

    # 1. Load Ground Truth & Stratify S1
    gt_df = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))
    s1_raw = load_tsv(os.path.join(train_dir, "train_source1.tsv"))

    # Add country info to GT for stratification
    gt_merged = gt_df.merge(s1_raw[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    gt_merged['is_singleton'] = gt_merged['matched_entity_ids'].str.strip() == ""

    # Stratified sampling on (country, is_singleton)
    s1_sampled = gt_merged.groupby(['country', 'is_singleton'], group_keys=False).apply(
        lambda x: x.sample(frac=n_s1 / len(gt_merged), random_state=random_seed),
        include_groups=False
    ).head(n_s1)

    sampled_s1_ids = set(s1_sampled['source1_entity_id'])
    s1_subset = s1_raw[s1_raw['entity_id'].isin(sampled_s1_ids)].copy()
    gt_subset = gt_df[gt_df['source1_entity_id'].isin(sampled_s1_ids)].copy()

    # 2. Collect ALL true matches for the sampled S1 entities
    true_s2_ids: Set[str] = set()
    true_s3_ids: Set[str] = set()
    for _, row in gt_subset.iterrows():
        m_str = str(row['matched_entity_ids'])
        if m_str.strip():
            for m in m_str.split(','):
                m_clean = m.strip()
                if m_clean.startswith('S2-'):
                    true_s2_ids.add(m_clean)
                elif m_clean.startswith('S3-'):
                    true_s3_ids.add(m_clean)

    print(f"True positive matches to recover: S2={len(true_s2_ids):,}, S3={len(true_s3_ids):,}")

    # 3. Stream S2 & S3, collecting true matches + stratified distractors
    def collect_source_records(filename: str, true_ids: Set[str], n_dist: int) -> pd.DataFrame:
        path = os.path.join(train_dir, filename)
        matched_chunks = []
        distractor_chunks = []
        distractors_collected = 0

        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=100000, keep_default_na=False):
            # Collect true matches
            m_chunk = chunk[chunk['entity_id'].isin(true_ids)]
            if len(m_chunk) > 0:
                matched_chunks.append(m_chunk)

            # Collect distractors
            if distractors_collected < n_dist:
                d_chunk = chunk[~chunk['entity_id'].isin(true_ids)]
                take = min(n_dist - distractors_collected, len(d_chunk))
                if take > 0:
                    distractor_chunks.append(d_chunk.head(take))
                    distractors_collected += take

        all_records = pd.concat(matched_chunks + distractor_chunks, ignore_index=True)
        return all_records.drop_duplicates(subset=['entity_id'])

    half_dist = n_distractors // 2
    print(f"Sampling Source 2 records (target distractors ~{half_dist:,})...")
    s2_subset = collect_source_records("train_source2.tsv", true_s2_ids, half_dist)
    print(f"Sampling Source 3 records (target distractors ~{half_dist:,})...")
    s3_subset = collect_source_records("train_source3.tsv", true_s3_ids, half_dist)

    # 4. Pre-Normalize DataFrames
    print("Normalizing benchmark DataFrames with multi-representation engine...")
    t0 = time.time()
    s1_norm = process_dataframe(s1_subset)
    s2_norm = process_dataframe(s2_subset)
    s3_norm = process_dataframe(s3_subset)
    print(f"Normalization complete in {time.time() - t0:.1f}s.")

    # Convert list columns to JSON or string for Parquet storage
    def serialize_lists(df: pd.DataFrame) -> pd.DataFrame:
        df_c = df.copy()
        for col in df_c.columns:
            if df_c[col].apply(lambda x: isinstance(x, list)).any():
                df_c[col] = df_c[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
        return df_c

    s1_ser = serialize_lists(s1_norm)
    s2_ser = serialize_lists(s2_norm)
    s3_ser = serialize_lists(s3_norm)

    # 5. Cache
    print("Caching benchmark to TSV...")
    s1_ser.to_csv(s1_cache, sep="\t", index=False)
    s2_ser.to_csv(s2_cache, sep="\t", index=False)
    s3_ser.to_csv(s3_cache, sep="\t", index=False)
    gt_subset.to_csv(gt_cache, sep="\t", index=False)

    print(f"Benchmark ready: S1={len(s1_norm):,}, S2={len(s2_norm):,}, S3={len(s3_norm):,}, GT={len(gt_subset):,}")
    return s1_norm, s2_norm, s3_norm, gt_subset
