import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from typing import Dict, Any, List

def compute_name_features(row) -> dict:
    n1 = str(row.get('s1_name_clean', ''))
    n2 = str(row.get('s2_name_clean', ''))
    
    if not n1 or not n2:
        return {
            'name_exact_match': 0.0,
            'name_jaro_winkler': 0.0,
            'name_levenshtein_ratio': 0.0,
            'name_token_jaccard': 0.0,
            'name_length_ratio': 0.0
        }
        
    t1 = set(row.get('s1_name_tokens', [])) if isinstance(row.get('s1_name_tokens'), list) else set()
    t2 = set(row.get('s2_name_tokens', [])) if isinstance(row.get('s2_name_tokens'), list) else set()
    
    union = len(t1 | t2)
    jaccard = len(t1 & t2) / union if union > 0 else 0.0
    
    return {
        'name_exact_match': 1.0 if n1 == n2 else 0.0,
        'name_jaro_winkler': fuzz.jaro_winkler_normalized(n1, n2),
        'name_levenshtein_ratio': fuzz.ratio(n1, n2) / 100.0,
        'name_token_jaccard': jaccard,
        'name_length_ratio': min(len(n1), len(n2)) / max(len(n1), len(n2))
    }

def compute_address_features(row) -> dict:
    a1 = str(row.get('s1_addr_clean', ''))
    a2 = str(row.get('s2_addr_clean', ''))
    
    # Missing value handling - explicitly treat missing addresses as neutral (np.nan), not 0
    if not a1 or not a2 or a1 == 'nan' or a2 == 'nan':
        return {
            'addr_exact_match': np.nan,
            'addr_jaro_winkler': np.nan,
            'addr_token_jaccard': np.nan,
            'addr_pin_match': np.nan,
            'addr_num_match': np.nan,
            'addr_missing': 1.0
        }
        
    t1 = set(row.get('s1_addr_tokens', [])) if isinstance(row.get('s1_addr_tokens'), list) else set()
    t2 = set(row.get('s2_addr_tokens', [])) if isinstance(row.get('s2_addr_tokens'), list) else set()
    
    union = len(t1 | t2)
    jaccard = len(t1 & t2) / union if union > 0 else 0.0
    
    p1, p2 = str(row.get('s1_addr_pin', '')), str(row.get('s2_addr_pin', ''))
    pin_match = 1.0 if p1 and p2 and p1 == p2 and p1 != 'nan' else 0.0 if p1 and p2 and p1 != 'nan' and p2 != 'nan' else np.nan
    
    num1, num2 = str(row.get('s1_addr_primary_num', '')), str(row.get('s2_addr_primary_num', ''))
    num_match = 1.0 if num1 and num2 and num1 == num2 and num1 != 'nan' else 0.0 if num1 and num2 and num1 != 'nan' and num2 != 'nan' else np.nan
    
    return {
        'addr_exact_match': 1.0 if a1 == a2 else 0.0,
        'addr_jaro_winkler': fuzz.jaro_winkler_normalized(a1, a2),
        'addr_token_jaccard': jaccard,
        'addr_pin_match': pin_match,
        'addr_num_match': num_match,
        'addr_missing': 0.0
    }

def generate_features(pairs_df: pd.DataFrame, s1_df: pd.DataFrame, combined_s2s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of candidate pairs (source1_entity_id, candidate_entity_id),
    joins with their respective normalized features and computes pairwise similarity features.
    """
    print("Joining candidates with raw features...", flush=True)
    df = pairs_df.merge(s1_df.add_prefix('s1_'), left_on='source1_entity_id', right_on='s1_entity_id', how='left')
    df = df.merge(combined_s2s3_df.add_prefix('s2_'), left_on='candidate_entity_id', right_on='s2_entity_id', how='left')
    
    print("Computing Name pairwise features...", flush=True)
    name_feats = df.apply(compute_name_features, axis=1, result_type='expand')
    
    print("Computing Address pairwise features...", flush=True)
    addr_feats = df.apply(compute_address_features, axis=1, result_type='expand')
    
    print("Computing Cross-field features...", flush=True)
    cross_feats = pd.DataFrame()
    
    # Country Match
    c1 = df['s1_country'].fillna('').astype(str).str.lower()
    c2 = df['s2_country'].fillna('').astype(str).str.lower()
    cross_feats['country_match'] = np.where((c1 == '') | (c2 == ''), np.nan, (c1 == c2).astype(float))
    
    # Cross-field: Name Similarity * Address Similarity
    # Fill NA address jaro with 0.5 (neutral) so name similarity isn't completely destroyed when address is missing
    cross_feats['name_x_addr'] = name_feats['name_jaro_winkler'] * addr_feats['addr_jaro_winkler'].fillna(0.5)
    
    final_feats = pd.concat([
        df[['source1_entity_id', 'candidate_entity_id']], 
        name_feats, 
        addr_feats, 
        cross_feats
    ], axis=1)
    
    return final_feats
