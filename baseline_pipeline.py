"""
Minimal Baseline Pipeline — Business Entity Resolution (Amazon ML Challenge 2026)

Rule-based approach using Sparse Matrices for extremely fast processing:
  1. Block by country (exact, case-insensitive)
  2. Normalize business_name: lowercase, strip punctuation, collapse whitespace
  3. Sparse Matrix Token-level Jaccard similarity (only evaluates candidates with >= 1 shared token)
  4. Pick best candidate per source (source2, source3) per source1 entity
  5. Threshold at Jaccard > 0.85

Outputs:
  output/train/candidate_pairs.tsv
  output/train/matching_results.tsv
  output/test/candidate_pairs.tsv
  output/test/matching_results.tsv
"""

import os
import re
import sys
import time
import gc
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

THRESHOLD = 0.85
CHUNK_SIZE = 100

def normalize_series(series):
    """Vectorized Pandas normalization: lowercase, strip punct, collapse space."""
    s = series.fillna("").str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def load_source(path):
    print(f"  Loading {os.path.basename(path)} …", end=" ", flush=True)
    t0 = time.time()
    # Read ONLY necessary columns to avoid out of memory
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "country"], keep_default_na=False)
    print(f"rows={len(df):>10,}  ({time.time() - t0:.1f}s)")
    df["country"] = df["country"].str.lower().str.strip()
    df["norm_name"] = normalize_series(df["business_name"])
    # Free memory of raw business_name
    df.drop(columns=["business_name"], inplace=True)
    return df

def process_country_chunk(s1_chunk, X_s1_chunk, s2_eids, X_s2, len_s1_chunk, len_s2):
    """
    Computes Jaccard for a chunk of S1 against all S2.
    Returns:
       candidate_lists: list of lists of candidate eids
       best_matches: list of (best_eid, best_score)
    """
    if X_s2.shape[0] == 0:
        return [[] for _ in range(X_s1_chunk.shape[0])], [(None, -1.0) for _ in range(X_s1_chunk.shape[0])]
        
    # intersection: dense @ sparse -> dense matrix directly
    # Shape: (chunk_size, N_s2)
    # Using dense avoids SciPy trying to allocate gigabytes for sparse index arrays
    intersections = (X_s1_chunk.toarray() @ X_s2.T)
    
    # Unions: |A| + |B| - |A ∩ B|
    # chunk_len_s1 is (chunk_size, 1), len_s2 is (N_s2,)
    unions = len_s1_chunk[:, None] + len_s2 - intersections
    
    # Jaccard
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccards = intersections / unions
        jaccards[unions == 0] = 0.0
        
    candidate_lists = []
    best_matches = []
    
    for i in range(X_s1_chunk.shape[0]):
        # Candidates are those that share >= 1 token
        cand_mask = intersections[i] > 0
        cand_indices = np.where(cand_mask)[0]
        
        cands = [s2_eids[idx] for idx in cand_indices]
        candidate_lists.append(cands)
        
        if len(cand_indices) > 0:
            # Find best
            best_idx = np.argmax(jaccards[i])
            best_score = jaccards[i, best_idx]
            # Must ensure best_idx is actually a candidate (intersection > 0)
            if cand_mask[best_idx]:
                best_matches.append((s2_eids[best_idx], best_score))
            else:
                best_matches.append((None, -1.0))
        else:
            best_matches.append((None, -1.0))
            
    return candidate_lists, best_matches

def run_pipeline(s1_df, s2_df, s3_df, threshold=THRESHOLD):
    countries = sorted(list(set(s1_df["country"].unique()) | 
                            set(s2_df["country"].unique()) | 
                            set(s3_df["country"].unique())))
    
    candidate_pairs = {}
    matching_results = {}
    
    for country in countries:
        print(f"\n  --- Processing Country: '{country}' ---")
        t0 = time.time()
        
        c_s1 = s1_df[s1_df["country"] == country]
        c_s2 = s2_df[s2_df["country"] == country]
        c_s3 = s3_df[s3_df["country"] == country]
        
        if len(c_s1) == 0:
            continue
            
        print(f"    S1: {len(c_s1):,}, S2: {len(c_s2):,}, S3: {len(c_s3):,}")
        
        # Vectorize
        vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
        # Fit ONLY on S1. Tokens in S2/S3 not present in S1 can never match anyway!
        vectorizer.fit(c_s1["norm_name"])
        
        X_s1 = vectorizer.transform(c_s1["norm_name"])
        X_s2 = vectorizer.transform(c_s2["norm_name"])
        X_s3 = vectorizer.transform(c_s3["norm_name"])
        
        len_s1 = np.array(X_s1.sum(axis=1)).flatten()
        len_s2 = np.array(X_s2.sum(axis=1)).flatten()
        len_s3 = np.array(X_s3.sum(axis=1)).flatten()
        
        s1_eids = c_s1["entity_id"].values
        s2_eids = c_s2["entity_id"].values
        s3_eids = c_s3["entity_id"].values
        
        for start in range(0, X_s1.shape[0], CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, X_s1.shape[0])
            X_s1_chunk = X_s1[start:end]
            len_s1_chunk = len_s1[start:end]
            eids_chunk = s1_eids[start:end]
            
            # Match S2
            cands_s2, best_s2 = process_country_chunk(X_s1_chunk, X_s1_chunk, s2_eids, X_s2, len_s1_chunk, len_s2)
            
            # Match S3
            cands_s3, best_s3 = process_country_chunk(X_s1_chunk, X_s1_chunk, s3_eids, X_s3, len_s1_chunk, len_s3)
            
            # Record
            for i, eid in enumerate(eids_chunk):
                all_cands = cands_s2[i] + cands_s3[i]
                candidate_pairs[eid] = ",".join(all_cands)
                
                matches = []
                s2_eid, s2_score = best_s2[i]
                if s2_eid and s2_score > threshold:
                    matches.append(s2_eid)
                    
                s3_eid, s3_score = best_s3[i]
                if s3_eid and s3_score > threshold:
                    matches.append(s3_eid)
                    
                matching_results[eid] = ",".join(matches)
                
        print(f"    Completed '{country}' in {time.time()-t0:.1f}s")
        gc.collect()

    return candidate_pairs, matching_results

def write_outputs(candidate_pairs, matching_results, out_dir, s1_eids):
    os.makedirs(out_dir, exist_ok=True)
    
    cp_path = os.path.join(out_dir, "candidate_pairs.tsv")
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_eids:
            f.write(f"{eid}\t{candidate_pairs.get(eid, '')}\n")
            
    mr_path = os.path.join(out_dir, "matching_results.tsv")
    with open(mr_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_eids:
            f.write(f"{eid}\t{matching_results.get(eid, '')}\n")
            
    return mr_path

def evaluate(mr_path, gt_path):
    print(f"\n  Loading predictions and ground truth …")
    pred = pd.read_csv(mr_path, sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    
    pred_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for _, row in pred.iterrows()}
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for _, row in gt.iterrows()}
    
    f05_scores, prec_scores, rec_scores = [], [], []
    for eid in gt_dict:
        true_set = gt_dict[eid]
        pred_set = pred_dict.get(eid, set())
        
        if not true_set and not pred_set:
            f05_scores.append(1.0); prec_scores.append(1.0); rec_scores.append(1.0)
            continue
        if not true_set and pred_set:
            f05_scores.append(0.0); prec_scores.append(0.0); rec_scores.append(0.0)
            continue
        if true_set and not pred_set:
            f05_scores.append(0.0); prec_scores.append(0.0); rec_scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        prec = tp / len(pred_set)
        rec = tp / len(true_set)
        f05 = 0.0 if (prec+rec)==0 else (1.25*prec*rec)/(0.25*prec+rec)
        
        f05_scores.append(f05); prec_scores.append(prec); rec_scores.append(rec)
        
    n = len(f05_scores)
    print(f"\n  === Train Evaluation (macro-avg over {n:,} entities) ===")
    print(f"  Precision : {sum(prec_scores)/n:.4f}")
    print(f"  Recall    : {sum(rec_scores)/n:.4f}")
    print(f"  F0.5      : {sum(f05_scores)/n:.4f}")

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("STEP 1: Loading TRAIN data")
    print("=" * 70)
    train_s1 = load_source(os.path.join(base, "dataset", "train", "train_source1.tsv"))
    train_s2 = load_source(os.path.join(base, "dataset", "train", "train_source2.tsv"))
    train_s3 = load_source(os.path.join(base, "dataset", "train", "train_source3.tsv"))
    
    print("\n" + "=" * 70)
    print("STEP 2: Running pipeline on TRAIN split")
    print("=" * 70)
    train_cands, train_matches = run_pipeline(train_s1, train_s2, train_s3)
    train_mr = write_outputs(train_cands, train_matches, os.path.join(base, "output", "train"), train_s1["entity_id"])
    
    del train_cands, train_s2, train_s3
    gc.collect()
    
    evaluate(train_mr, os.path.join(base, "dataset", "train", "train_ground_truth.tsv"))
    del train_s1, train_matches
    gc.collect()
    
    print("\n" + "=" * 70)
    print("STEP 3: Loading TEST data")
    print("=" * 70)
    test_s1 = load_source(os.path.join(base, "dataset", "test", "test_source1.tsv"))
    test_s2 = load_source(os.path.join(base, "dataset", "test", "test_source2.tsv"))
    test_s3 = load_source(os.path.join(base, "dataset", "test", "test_source3.tsv"))
    
    print("\n" + "=" * 70)
    print("STEP 4: Running pipeline on TEST split")
    print("=" * 70)
    test_cands, test_matches = run_pipeline(test_s1, test_s2, test_s3)
    write_outputs(test_cands, test_matches, os.path.join(base, "output", "test"), test_s1["entity_id"])
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
