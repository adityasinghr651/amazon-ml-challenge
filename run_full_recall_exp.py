import pandas as pd
import numpy as np
import time
import sys
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.getcwd())
from src.business_entity_resolution.normalization import normalize_name, normalize_address

def load_gt(path):
    gt_dict = {}
    total_with_match = 0
    # pandas handles messy lines much better than standard csv
    df = pd.read_csv(path, sep='\t', dtype=str, na_filter=False)
    for _, row in df.iterrows():
        s1_id = str(row['source1_entity_id'])
        matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
        if matches:
            gt_dict[s1_id] = set(matches)
            total_with_match += 1
    return gt_dict, total_with_match

def run_memory_safe_eval():
    s1_path = 'dataset/train/train_source1.tsv'
    s2_path = 'dataset/train/train_source2.tsv'
    s3_path = 'dataset/train/train_source3.tsv'
    gt_path = 'dataset/train/train_ground_truth.tsv'
    
    print("Loading Ground Truth...", flush=True)
    gt_dict, total_with_match = load_gt(gt_path)
    print(f"S1 records with >= 1 true match: {total_with_match}", flush=True)
    
    index_a = defaultdict(list)
    index_b = defaultdict(list)
    token_df_counts = defaultdict(int)
    
    s23_names_for_tfidf = []
    s23_ids_for_tfidf = []
    entity_dict = {}
    
    print("Pass 1: Building Index A, B and Term Frequencies...", flush=True)
    count = 0
    for path in [s2_path, s3_path]:
        for chunk in pd.read_csv(path, sep='\t', chunksize=100000, dtype=str, na_filter=False, on_bad_lines='skip'):
            for _, row in chunk.iterrows():
                eid = str(row['entity_id'])
                n_res = normalize_name(str(row.get('business_name', '')))
                a_res = normalize_address(str(row.get('business_address', '')))
                
                name_clean = n_res['clean_norm']
                pin = a_res['pin']
                num = a_res['primary_number']
                
                entity_dict[eid] = {
                    'name_toks': set(n_res['tokens']),
                    'addr_toks': set(a_res['tokens'])
                }
                
                if name_clean:
                    index_a[name_clean].append(eid)
                    s23_names_for_tfidf.append(name_clean)
                    s23_ids_for_tfidf.append(eid)
                    
                if pin and num:
                    index_b[(pin, num)].append(eid)
                    
                for t in set(n_res['informative_tokens']):
                    token_df_counts[t] += 1
                    
                count += 1
            print(f"  Processed {count} S2/S3 records...", flush=True)
                    
    frequencies = list(token_df_counts.values())
    cutoff_freq = np.percentile(frequencies, 25) if frequencies else 0
    
    print(f"Pass 2: Building Index C (Rare tokens cutoff <= {cutoff_freq})...", flush=True)
    index_c = defaultdict(list)
    for path in [s2_path, s3_path]:
        for chunk in pd.read_csv(path, sep='\t', chunksize=100000, dtype=str, na_filter=False, on_bad_lines='skip'):
            for _, row in chunk.iterrows():
                eid = str(row['entity_id'])
                n_res = normalize_name(str(row.get('business_name', '')))
                for t in set(n_res['informative_tokens']):
                    if token_df_counts[t] <= cutoff_freq:
                        index_c[t].append(eid)

    print("Building Index D (TF-IDF ANN)...", flush=True)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_df=0.9, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(s23_names_for_tfidf)
    nn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=4)
    nn.fit(tfidf_matrix)
    
    print("Evaluating against S1...", flush=True)
    stats = {
        "A+B": {"fully_recovered": 0, "total_cands": 0, "zero_cands": 0, "zero_cands_with_match": 0, "missed_examples": []},
        "A+B+C": {"fully_recovered": 0, "total_cands": 0, "zero_cands": 0, "zero_cands_with_match": 0, "missed_examples": []},
        "A+B+C+D": {"fully_recovered": 0, "total_cands": 0, "zero_cands": 0, "zero_cands_with_match": 0, "missed_examples": []}
    }
    
    total_s1 = 0
    
    for chunk in pd.read_csv(s1_path, sep='\t', chunksize=10000, dtype=str, na_filter=False, on_bad_lines='skip'):
        batch_s1_names = []
        batch_s1_rows = []
        for _, row in chunk.iterrows():
            batch_s1_rows.append(row)
            batch_s1_names.append(normalize_name(str(row.get('business_name', '')))['clean_norm'])
            
        d_cands_batch = [[] for _ in range(len(batch_s1_rows))]
        if len(batch_s1_names) > 0:
            s1_tfidf = vectorizer.transform(batch_s1_names)
            distances, indices = nn.kneighbors(s1_tfidf)
            for i in range(len(batch_s1_rows)):
                cands = []
                for j, dist in enumerate(distances[i]):
                    if dist <= 0.5:
                        cands.append(s23_ids_for_tfidf[indices[i][j]])
                d_cands_batch[i] = cands
                
        for i, row in enumerate(batch_s1_rows):
            s1_id = str(row['entity_id'])
            has_match = s1_id in gt_dict
            gt_matches = gt_dict.get(s1_id, set())
            
            n_res = normalize_name(str(row.get('business_name', '')))
            a_res = normalize_address(str(row.get('business_address', '')))
            
            name_clean = n_res['clean_norm']
            pin = a_res['pin']
            num = a_res['primary_number']
            informative_tokens = n_res['informative_tokens']
            
            cands_A = set(index_a[name_clean]) if name_clean in index_a else set()
            cands_B = set(index_b[(pin, num)]) if pin and num and (pin, num) in index_b else set()
            cands_AB = cands_A | cands_B
            
            cands_C = set()
            for t in informative_tokens:
                if t in index_c:
                    cands_C.update(index_c[t])
                    
            cands_ABC = cands_AB | cands_C
            
            cands_D = set(d_cands_batch[i])
            cands_ABCD = cands_ABC | cands_D
            
            if len(cands_ABCD) > 50:
                s1_nt = set(n_res['tokens'])
                s1_at = set(a_res['tokens'])
                scored = []
                for c in cands_ABCD:
                    if c in entity_dict:
                        c_data = entity_dict[c]
                        n_u = len(s1_nt | c_data['name_toks'])
                        n_s = len(s1_nt & c_data['name_toks']) / n_u if n_u > 0 else 0
                        a_s = len(s1_at & c_data['addr_toks'])
                        scored.append((n_s + (a_s * 0.2), c))
                    else:
                        scored.append((0, c))
                scored.sort(key=lambda x: x[0], reverse=True)
                cands_ABCD = {x[1] for x in scored[:20]}
                
            for phase, cands in [("A+B", cands_AB), ("A+B+C", cands_ABC), ("A+B+C+D", cands_ABCD)]:
                stats[phase]["total_cands"] += len(cands)
                if len(cands) == 0:
                    stats[phase]["zero_cands"] += 1
                    if has_match:
                        stats[phase]["zero_cands_with_match"] += 1
                        if len(stats[phase]["missed_examples"]) < 3:
                            stats[phase]["missed_examples"].append(s1_id)
                            
                if has_match and gt_matches.issubset(cands):
                    stats[phase]["fully_recovered"] += 1

        total_s1 += len(batch_s1_rows)
        print(f"  Evaluated {total_s1} S1 records...", flush=True)
            
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    for phase in ["A+B", "A+B+C", "A+B+C+D"]:
        r = stats[phase]
        recall = r["fully_recovered"] / total_with_match if total_with_match > 0 else 0
        avg_cand = r["total_cands"] / total_s1 if total_s1 > 0 else 0
        print(f"\n--- {phase} ---")
        print(f"Candidate Recall: {recall:.4f} ({r['fully_recovered']}/{total_with_match} true matches recovered)")
        print(f"Avg Candidates/S1: {avg_cand:.2f}")
        print(f"Zero Candidates Total: {r['zero_cands']}")
        print(f"Zero Candidates (with true match): {r['zero_cands_with_match']}")
        print(f"Missed True Matches Examples: {r['missed_examples']}")

if __name__ == "__main__":
    run_memory_safe_eval()
