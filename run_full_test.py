#!/usr/bin/env python3
"""
run_full_test.py - Production End-to-End Test Inference Engine
Amazon ML Challenge 2026: Business Entity Resolution

Processes the FULL official test dataset (1,732,544 S1 entities and ~9.97M target records)
using country-sharded processing, frozen 5-strategy blocking, Feature Set v1 (32 features),
and the frozen LightGBM champion model at calibrated threshold tau = 0.80.

Generates:
  - output/full_test/matching_results.tsv  (Official competition submission)
  - output/full_test/candidate_pairs.tsv   (Full test candidate pairs)

Author: ML Challenge 2026 Team
"""

import os
import sys
import gc
import glob
import time
import pickle
import psutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Any, Optional
from anyascii import anyascii
from collections import defaultdict

# Add project root to sys.path
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
    """Returns current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def auto_detect_test_dir(user_path: Optional[str] = None) -> str:
    """Finds the directory containing test_source1.tsv, supporting local and Kaggle mounts."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates.extend([
        "dataset/test",
        "dataset",
        os.path.join("student_resource", "dataset", "test"),
        os.path.join("..", "dataset", "test"),
    ])
    # Search /kaggle/input if running on Kaggle
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "test_source1.tsv" in files:
                candidates.insert(0, root)
                break

    for c in candidates:
        if os.path.exists(os.path.join(c, "test_source1.tsv")):
            return os.path.abspath(c)

    raise FileNotFoundError(
        f"Could not locate test_source1.tsv in specified path '{user_path}' "
        f"or standard search directories: {candidates}"
    )

def fast_normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, memory-efficient single-pass normalization."""
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

def stream_country_records_from_tsv(tsv_path: str, country_name: str) -> List[dict]:
    """Streams and filters TSV lines for a given country without loading the whole file."""
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
    return records

def main():
    parser = argparse.ArgumentParser(description="Run complete full-test inference pipeline for Amazon ML Challenge 2026.")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    parser.add_argument("--output-dir", default="output/full_test", help="Directory where matching_results.tsv and candidate_pairs.tsv will be saved")
    parser.add_argument("--model-path", default="models/final_lightgbm_champion.pkl", help="Path to serialized LightGBM champion model")
    parser.add_argument("--threshold", type=float, default=FROZEN_THRESHOLD, help="Calibrated decision threshold (default: 0.80)")
    parser.add_argument("--batch-size", type=int, default=500, help="Query batch size for retrieval (default: 500)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running utils/validate_submission.py")
    args = parser.parse_args()

    t_global_start = time.time()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - PRODUCTION FULL TEST INFERENCE")
    print("=" * 80)
    print(f"Timestamp:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Decision Threshold: tau = {args.threshold}")
    print(f"Model Checkpoint:   {args.model_path}")
    print(f"Retrieval Batch:    {args.batch_size}")
    print("=" * 80)

    # 1. Resolve Dataset Directory
    test_dir = auto_detect_test_dir(args.test_dir)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n[1/6] Resolved Test Dataset Directory: {test_dir}")
    print(f"  Source 1: {s1_path}")
    print(f"  Source 2: {s2_path}")
    print(f"  Source 3: {s3_path}")

    # 2. Load Frozen Champion Model
    print(f"\n[2/6] Loading frozen champion model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Champion model not found at {args.model_path}. Ensure models/final_lightgbm_champion.pkl is present.")
    
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Champion model loaded! Features: {model.n_features_in_}, RAM: {get_ram_mb():.1f} MB")

    # 3. Load S1 Entities and Preserve Exact Row Order
    print(f"\n[3/6] Loading test Source 1 entities from {s1_path}...")
    t_s1 = time.time()
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    total_s1 = len(s1_df)
    all_s1_ordered = s1_df['entity_id'].tolist()
    print(f"Loaded {total_s1:,} S1 entities in {time.time()-t_s1:.2f}s (RAM: {get_ram_mb():.1f} MB)")

    # Group S1 by Country
    s1_by_country: Dict[str, pd.DataFrame] = {}
    for country, group in s1_df.groupby("country"):
        s1_by_country[country] = group.copy()
        print(f"  Country '{country}': {len(group):,} S1 entities")

    del s1_df
    gc.collect()

    # Results dictionaries
    final_matches: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ordered}
    all_candidates_dict: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ordered}

    # 4. Country Sharded Execution (France -> US -> India)
    # Processing countries sequentially ensures peak RAM stays <= 8.5 GB
    countries_to_process = [c for c in ["France", "US", "India"] if c in s1_by_country]
    # Add any remaining countries if present
    for c in s1_by_country:
        if c not in countries_to_process:
            countries_to_process.append(c)

    print(f"\n[4/6] Executing Country-Sharded Inference across: {countries_to_process}")

    country_metrics = {}

    for country in countries_to_process:
        t_country_start = time.time()
        s1_c_df = s1_by_country[country]
        n_queries = len(s1_c_df)

        print("\n" + "=" * 60)
        print(f"STARTING COUNTRY SHARD: {country} ({n_queries:,} S1 queries) | RAM: {get_ram_mb():.1f} MB")
        print("=" * 60)

        # A. Stream Target Records for this country
        t_load = time.time()
        print(f"  Streaming {country} target records from test_source2 & test_source3...")
        t_recs_s2 = stream_country_records_from_tsv(s2_path, country)
        t_recs_s3 = stream_country_records_from_tsv(s3_path, country)
        target_recs = t_recs_s2 + t_recs_s3
        del t_recs_s2, t_recs_s3
        gc.collect()

        n_targets = len(target_recs)
        print(f"  Loaded {n_targets:,} target records in {time.time()-t_load:.2f}s (RAM: {get_ram_mb():.1f} MB)")

        # B. Fast Normalization
        t_norm = time.time()
        print(f"  Normalizing {n_queries:,} queries and {n_targets:,} targets...")
        s1_norm_df = fast_normalize_df(s1_c_df)
        target_norm_df = fast_normalize_df(pd.DataFrame(target_recs))
        del target_recs
        gc.collect()
        print(f"  Normalized records in {time.time()-t_norm:.2f}s (RAM: {get_ram_mb():.1f} MB)")

        # In-memory lookup database for feature extraction
        rec_db: Dict[str, dict] = {}
        for _, r in s1_norm_df.iterrows():
            d = r.to_dict()
            d["name_translit"] = anyascii(str(d.get("name_clean", ""))).lower().strip()
            rec_db[d["entity_id"]] = d

        for _, r in target_norm_df.iterrows():
            d = r.to_dict()
            d["name_translit"] = anyascii(str(d.get("name_clean", ""))).lower().strip()
            rec_db[d["entity_id"]] = d

        # C. Candidate Generation Ensemble
        t_block = time.time()
        union_engine = CandidateUnionEngine()

        # 1. Exact Name (clean, core, sorted)
        print("  -> Blocker A: ExactNameBlocker...")
        b_name = ExactNameBlocker(partition_by_country=False).fit(target_norm_df)
        union_engine.add_blocker_results("exact_name", b_name.block(s1_norm_df))
        del b_name; gc.collect()

        # 2. Address Anchor (pin + primary num)
        print("  -> Blocker B: AddressAnchorBlocker...")
        b_addr = AddressAnchorBlocker(partition_by_country=False).fit(target_norm_df)
        union_engine.add_blocker_results("address", b_addr.block(s1_norm_df))
        del b_addr; gc.collect()

        # 3. Rare Token (doc_freq <= 100, top 2 rarest)
        print("  -> Blocker C: RareTokenBlocker...")
        b_rare = RareTokenBlocker(config=RareTokenConfig(max_doc_freq=100, min_token_len=3, top_n_rare=2, max_bucket_size=300), partition_by_country=False).fit(target_norm_df)
        union_engine.add_blocker_results("rare_token", b_rare.block(s1_norm_df))
        del b_rare; gc.collect()

        # 4. Transliteration (anyascii)
        print("  -> Blocker H: TransliterationBlocker...")
        b_translit = TransliterationBlocker(max_df=150, min_token_len=4, max_bucket_size=30, partition_by_country=False).fit(target_norm_df)
        union_engine.add_blocker_results("transliteration", b_translit.block(s1_norm_df))
        del b_translit; gc.collect()

        # 5. Char N-Gram Reciprocal Retrieval (char_wb, n-gram (2,4), sim >= 0.25, top 20 forward, top 1 reverse)
        print("  -> Blocker D+F: CharNgramRetrievalBlocker (reciprocal)...")
        b_retrieval = CharNgramRetrievalBlocker(
            top_k=20,
            ngram_range=(2, 4),
            sim_threshold=0.25,
            reciprocal=True,
            reciprocal_top_k=1,
            partition_by_country=False,
            batch_size=args.batch_size
        )
        b_retrieval.fit(target_norm_df)
        union_engine.add_blocker_results("char_retrieval_reciprocal", b_retrieval.block(s1_norm_df))
        del b_retrieval, target_norm_df
        gc.collect()

        print(f"  Blocking completed in {time.time()-t_block:.2f}s (RAM: {get_ram_mb():.1f} MB)")

        # D. Streaming Feature Extraction & LightGBM Inference
        t_infer = time.time()
        cand_dict = union_engine.get_candidate_dict()
        total_shard_cands = sum(len(cset) for cset in cand_dict.values())
        print(f"  Generated {total_shard_cands:,} candidate pairs (avg {total_shard_cands/n_queries:.2f}/query)")

        # Score in streaming chunks to keep RAM strictly bounded
        CHUNK_SIZE = 50000
        chunk_X = []
        chunk_pairs = []
        matched_pairs_shard = []

        for s1_id in s1_norm_df["entity_id"]:
            c_set = cand_dict.get(s1_id, set())
            all_candidates_dict[s1_id] = c_set
            if not c_set:
                continue

            s1_rec = rec_db.get(s1_id, {})
            for c_id in c_set:
                c_rec = rec_db.get(c_id, {})
                strats = ",".join(union_engine.provenance[s1_id][c_id])
                feat_dict = extract_pairwise_features(s1_rec, c_rec, strats)
                feat_vals = [feat_dict[fn] for fn in FEATURE_NAMES_V1]

                chunk_X.append(feat_vals)
                chunk_pairs.append((s1_id, c_id))

                if len(chunk_X) >= CHUNK_SIZE:
                    X_mat = np.array(chunk_X, dtype=np.float32)
                    probs = model.predict_proba(X_mat)[:, 1]
                    for idx in range(len(probs)):
                        if probs[idx] >= args.threshold:
                            matched_pairs_shard.append((chunk_pairs[idx][0], chunk_pairs[idx][1], probs[idx]))
                    chunk_X = []
                    chunk_pairs = []

        # Process remainder
        if chunk_X:
            X_mat = np.array(chunk_X, dtype=np.float32)
            probs = model.predict_proba(X_mat)[:, 1]
            for idx in range(len(probs)):
                if probs[idx] >= args.threshold:
                    matched_pairs_shard.append((chunk_pairs[idx][0], chunk_pairs[idx][1], probs[idx]))
            del chunk_X, chunk_pairs

        # Sort matches by score descending and assign to final_matches
        matched_pairs_shard.sort(key=lambda x: x[2], reverse=True)
        for s1_id, c_id, score in matched_pairs_shard:
            if c_id not in final_matches[s1_id]:
                final_matches[s1_id].append(c_id)

        shard_duration = time.time() - t_country_start
        matched_s1_count = sum(1 for s1 in s1_norm_df["entity_id"] if final_matches[s1])
        print(f"  Shard {country} finished in {shard_duration:.2f}s! Matched S1s: {matched_s1_count:,} ({matched_s1_count/n_queries*100:.2f}%)")

        country_metrics[country] = {
            "S1_Count": n_queries,
            "Targets": n_targets,
            "Total_Candidates": total_shard_cands,
            "Avg_Candidates": round(total_shard_cands / n_queries, 2),
            "Matched_S1": matched_s1_count,
            "Runtime_s": round(shard_duration, 2)
        }

        # Clear memory
        del rec_db, union_engine, s1_norm_df, cand_dict, matched_pairs_shard
        gc.collect()

    # 5. Write Production Artifacts
    print("\n[5/6] Generating Final Output Files adhering to competition schema...")
    os.makedirs(args.output_dir, exist_ok=True)
    match_out_path = os.path.join(args.output_dir, "matching_results.tsv")
    cand_out_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print(f"  Writing {match_out_path}...")
    num_singletons = 0
    num_matched = 0
    total_matched_pairs = 0

    with open(match_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ordered:
            m_list = final_matches.get(s1, [])
            if m_list:
                f_match.write(f"{s1}\t{','.join(m_list)}\n")
                num_matched += 1
                total_matched_pairs += len(m_list)
            else:
                f_match.write(f"{s1}\t\n")
                num_singletons += 1

    print(f"  Wrote {total_s1:,} rows to {match_out_path}")
    print(f"    Matched Entities: {num_matched:,} ({num_matched/total_s1*100:.2f}%)")
    print(f"    Singletons:       {num_singletons:,} ({num_singletons/total_s1*100:.2f}%)")
    print(f"    Total Pairs:      {total_matched_pairs:,}")

    print(f"  Writing {cand_out_path}...")
    with open(cand_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_s1_ordered:
            c_set = all_candidates_dict.get(s1, set())
            c_str = ",".join(sorted(list(c_set))) if c_set else ""
            f_cand.write(f"{s1}\t{c_str}\n")

    print(f"  Wrote {total_s1:,} rows to {cand_out_path}")

    # 6. Execute Official Submission Validator
    if not args.skip_validation:
        print("\n[6/6] Executing Official Amazon Submission Validator...")
        validator_script = "utils/validate_submission.py"
        if os.path.exists(validator_script):
            val_cmd = [
                sys.executable,
                validator_script,
                "--matching", match_out_path,
                "--candidate", cand_out_path,
                "--test-dir", test_dir
            ]
            print(f"Running: {' '.join(val_cmd)}")
            res = subprocess.run(val_cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print("Validator STDERR:\n", res.stderr)
            if res.returncode == 0:
                print("-> PASS: Official submission validator passed with Exit Code 0!")
            else:
                print(f"-> WARNING: Validator returned exit code {res.returncode}")
        else:
            print(f"Validator script {validator_script} not found; skipping.")

    total_time = time.time() - t_global_start
    print("\n" + "=" * 80)
    print("FULL TEST PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total Execution Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Artifacts Created:")
    print(f"  - {match_out_path}")
    print(f"  - {cand_out_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
