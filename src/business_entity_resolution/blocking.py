import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Any
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def build_index_a(combined_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Builds exact normalized name index."""
    index = defaultdict(list)
    valid_df = combined_df[combined_df['name_clean'].astype(bool)]
    for idx, row in valid_df.iterrows():
        index[row['name_clean']].append(row['entity_id'])
    return index

def build_index_b(combined_df: pd.DataFrame) -> Dict[tuple, List[str]]:
    """Builds address anchor index (pin_code, primary_number)."""
    index = defaultdict(list)
    valid_df = combined_df[combined_df['addr_pin'].astype(bool) & combined_df['addr_primary_num'].astype(bool)]
    for idx, row in valid_df.iterrows():
        index[(row['addr_pin'], row['addr_primary_num'])].append(row['entity_id'])
    return index

def build_index_c(combined_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Builds rare token inverted index."""
    percentile_cutoff = config.get('rare_token_percentile', 25)
    
    token_df_counts = defaultdict(int)
    valid_df = combined_df[combined_df['name_informative_tokens'].astype(bool)]
    
    for tokens in valid_df['name_informative_tokens']:
        if isinstance(tokens, list):
            for t in set(tokens):
                token_df_counts[t] += 1
                
    if not token_df_counts:
        return {}
        
    frequencies = list(token_df_counts.values())
    cutoff_freq = np.percentile(frequencies, percentile_cutoff)
    
    index = defaultdict(list)
    for idx, row in valid_df.iterrows():
        tokens = row['name_informative_tokens']
        if isinstance(tokens, list):
            for t in tokens:
                if token_df_counts[t] <= cutoff_freq:
                    index[t].append(row['entity_id'])
    return index

def build_index_d(combined_df: pd.DataFrame, config: Dict[str, Any]):
    """Builds TF-IDF character n-gram ANN index."""
    valid_df = combined_df[combined_df['name_clean'].astype(bool)].copy()
    if valid_df.empty:
        return None, None, []
        
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_df=0.9, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(valid_df['name_clean'])
    
    nn = NearestNeighbors(n_neighbors=config.get('ann_neighbors', 10), metric='cosine', n_jobs=-1)
    nn.fit(tfidf_matrix)
    
    return vectorizer, nn, valid_df['entity_id'].tolist()

def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Generates candidates using union of indexes A, B, C, D and prunes them."""
    active_blocks = config.get('active_blocks', ['A', 'B', 'C', 'D'])
    combined_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    index_a = build_index_a(combined_df) if 'A' in active_blocks else {}
    index_b = build_index_b(combined_df) if 'B' in active_blocks else {}
    index_c = build_index_c(combined_df, config) if 'C' in active_blocks else {}
    vectorizer_d, nn_d, d_entity_ids = build_index_d(combined_df, config) if 'D' in active_blocks else (None, None, [])
    
    # Pre-build lookup for pruning
    entity_dict = {}
    if config.get('enable_pruning', True):
        for idx, row in combined_df.iterrows():
            entity_dict[row['entity_id']] = {
                'name_toks': set(row['name_tokens']) if isinstance(row['name_tokens'], list) else set(),
                'addr_toks': set(row['addr_tokens']) if isinstance(row['addr_tokens'], list) else set()
            }
            
    # Batch ANN query
    d_candidates = []
    if 'D' in active_blocks and vectorizer_d is not None:
        valid_s1 = s1_df['name_clean'].fillna('')
        s1_tfidf = vectorizer_d.transform(valid_s1)
        distances, indices = nn_d.kneighbors(s1_tfidf)
        
        max_dist = config.get('ann_max_distance', 0.5)
        for i in range(len(s1_df)):
            cands = []
            for j, dist in enumerate(distances[i]):
                if dist <= max_dist:
                    cands.append(d_entity_ids[indices[i][j]])
            d_candidates.append(cands)
    else:
        d_candidates = [[]] * len(s1_df)
        
    results = []
    for i, (idx, row) in enumerate(s1_df.iterrows()):
        candidates = set()
        
        if 'A' in active_blocks:
            name_clean = row['name_clean']
            if name_clean and name_clean in index_a:
                candidates.update(index_a[name_clean])
                
        if 'B' in active_blocks:
            pin = row['addr_pin']
            num = row['addr_primary_num']
            if pin and num and (pin, num) in index_b:
                candidates.update(index_b[(pin, num)])
                
        if 'C' in active_blocks:
            tokens = row['name_informative_tokens']
            if isinstance(tokens, list):
                for t in tokens:
                    if t in index_c:
                        candidates.update(index_c[t])
                        
        if 'D' in active_blocks:
            candidates.update(d_candidates[i])
            
        # 3. PRUNING / RERANKING
        if config.get('enable_pruning', True) and len(candidates) > config.get('max_candidates_before_pruning', 50):
            s1_nt = set(row['name_tokens']) if isinstance(row['name_tokens'], list) else set()
            s1_at = set(row['addr_tokens']) if isinstance(row['addr_tokens'], list) else set()
            
            scored_cands = []
            for cand in candidates:
                if cand in entity_dict:
                    c_data = entity_dict[cand]
                    c_nt = c_data['name_toks']
                    c_at = c_data['addr_toks']
                    
                    n_union = len(s1_nt | c_nt)
                    n_score = len(s1_nt & c_nt) / n_union if n_union > 0 else 0
                    a_score = len(s1_at & c_at)
                    
                    score = n_score + (a_score * 0.2)
                    scored_cands.append((score, cand))
                else:
                    scored_cands.append((0, cand))
                    
            scored_cands.sort(key=lambda x: x[0], reverse=True)
            top_k = config.get('prune_top_k', 20)
            candidates = {c[1] for c in scored_cands[:top_k]}
            
        results.append({
            "source1_entity_id": row['entity_id'],
            "candidate_entity_ids": ",".join(list(candidates))
        })
        
    return pd.DataFrame(results)

def measure_blocking_performance(candidates_df: pd.DataFrame, ground_truth_df: pd.DataFrame, total_s2s3_records: int) -> dict:
    candidates_dict = {}
    for idx, row in candidates_df.iterrows():
        cands = [c.strip() for c in str(row['candidate_entity_ids']).split(",") if c.strip()]
        candidates_dict[row['source1_entity_id']] = set(cands)
        
    gt_dict = {}
    for idx, row in ground_truth_df.iterrows():
        gt_matches = [m.strip() for m in str(row['matched_entity_ids']).split(",") if m.strip()]
        if gt_matches:
            gt_dict[row['source1_entity_id']] = set(gt_matches)
            
    total_with_matches = len(gt_dict)
    fully_recovered = 0
    
    for s1_id, gt_matches in gt_dict.items():
        if s1_id in candidates_dict:
            pred_matches = candidates_dict[s1_id]
            if gt_matches.issubset(pred_matches):
                fully_recovered += 1
                
    recall = fully_recovered / total_with_matches if total_with_matches > 0 else 0
    
    cand_counts = [len(cands) for cands in candidates_dict.values()]
    if cand_counts:
        avg_cands = np.mean(cand_counts)
        median_cands = np.median(cand_counts)
        p95_cands = np.percentile(cand_counts, 95)
        max_cands = np.max(cand_counts)
    else:
        avg_cands = median_cands = p95_cands = max_cands = 0
        
    reduction_ratio = 1 - (avg_cands / total_s2s3_records) if total_s2s3_records > 0 else 0
    
    return {
        "candidate_recall": recall,
        "avg_candidates": avg_cands,
        "median_candidates": median_cands,
        "p95_candidates": p95_cands,
        "max_candidates": max_cands,
        "reduction_ratio": reduction_ratio
    }
