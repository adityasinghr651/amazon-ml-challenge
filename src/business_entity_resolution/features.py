"""
Phase 4: features.py
Pairwise feature engineering between S1 and S2/S3 candidates.
"""

import rapidfuzz

def extract_features(s1_record, candidate_record):
    """
    Extract name, address, and cross-field similarity features.
    Returns a dictionary of features.
    """
    s1_name = str(s1_record.get('business_name', '') or '').lower()
    c_name = str(candidate_record.get('business_name', '') or '').lower()
    s1_addr = str(s1_record.get('business_address', '') or '').lower()
    c_addr = str(candidate_record.get('business_address', '') or '').lower()
    
    name_lev = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_name, c_name)
    name_jaro = rapidfuzz.distance.JaroWinkler.similarity(s1_name, c_name)
    addr_lev = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_addr, c_addr) if s1_addr and c_addr else 0.0
    
    return {
        'name_lev': name_lev,
        'name_jaro': name_jaro,
        'addr_lev': addr_lev,
        'name_exact': float(s1_name == c_name and s1_name != ''),
        'country_match': float(str(s1_record.get('country', '')).lower() == str(candidate_record.get('country', '')).lower())
    }
