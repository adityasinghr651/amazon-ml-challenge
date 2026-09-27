"""
Phase 4: features.py
Comprehensive Pairwise Feature Engineering Suite (Feature Set v1).
Computes 32 high-precision lexical, token, address, cross-script transliteration,
missingness, interaction, and blocking provenance signals.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz, distance
from anyascii import anyascii

# Feature names in deterministic sorted order
FEATURE_NAMES_V1 = [
    # Name Features (13)
    "name_exact",
    "name_core_exact",
    "name_sorted_exact",
    "name_lev",
    "name_jaro",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_wratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_rare_token_overlap",
    "name_token_count_diff",
    "name_length_ratio",
    # Address Features (7)
    "addr_lev",
    "addr_token_sort_ratio",
    "addr_token_jaccard",
    "addr_pin_match",
    "addr_num_match",
    "addr_length_ratio",
    "addr_either_missing",
    # Cross-Script / Transliteration Features (2)
    "name_translit_lev",
    "name_translit_exact",
    # Cross-Field & Interactions (4)
    "country_match",
    "cross_name_addr",
    "name_high_addr_missing",
    "name_exact_addr_match",
    # Provenance Signals (6)
    "prov_exact_name",
    "prov_address",
    "prov_rare_token",
    "prov_retrieval_reciprocal",
    "prov_transliteration",
    "prov_count"
]

def _safe_str(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "") else s

def _safe_list(val) -> list:
    if isinstance(val, list):
        return val
    elif isinstance(val, str) and val.strip():
        return val.strip().split()
    return []

def _token_jaccard(toks_a: list, toks_b: list) -> float:
    if not toks_a and not toks_b:
        return 0.0
    set_a = set(toks_a)
    set_b = set(toks_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union

def _token_overlap(toks_a: list, toks_b: list) -> float:
    if not toks_a or not toks_b:
        return 0.0
    set_a = set(toks_a)
    set_b = set(toks_b)
    min_len = min(len(set_a), len(set_b))
    if min_len == 0:
        return 0.0
    return len(set_a & set_b) / min_len

def extract_pairwise_features(
    s1_rec: Dict[str, Any],
    c_rec: Dict[str, Any],
    strategies_str: str = ""
) -> Dict[str, float]:
    """
    Extract Feature Set v1 (32 features) for a single candidate pair.
    """
    # 1. Names
    s1_name = _safe_str(s1_rec.get("name_clean") or s1_rec.get("business_name"))
    c_name = _safe_str(c_rec.get("name_clean") or c_rec.get("business_name"))

    s1_core = _safe_str(s1_rec.get("name_core"))
    c_core = _safe_str(c_rec.get("name_core"))

    s1_sorted = _safe_str(s1_rec.get("name_sorted"))
    c_sorted = _safe_str(c_rec.get("name_sorted"))

    s1_toks = _safe_list(s1_rec.get("name_tokens"))
    c_toks = _safe_list(c_rec.get("name_tokens"))

    s1_info = _safe_list(s1_rec.get("name_informative_tokens"))
    c_info = _safe_list(c_rec.get("name_informative_tokens"))

    # Name similarity metrics
    has_names = bool(s1_name and c_name)
    name_exact = float(s1_name == c_name and has_names)
    name_core_exact = float(s1_core == c_core and s1_core != "")
    name_sorted_exact = float(s1_sorted == c_sorted and s1_sorted != "")

    if has_names:
        name_lev = distance.Levenshtein.normalized_similarity(s1_name, c_name)
        name_jaro = distance.JaroWinkler.similarity(s1_name, c_name)
        name_token_sort_ratio = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
        name_token_set_ratio = fuzz.token_set_ratio(s1_name, c_name) / 100.0
        name_wratio = fuzz.WRatio(s1_name, c_name) / 100.0
        name_token_jaccard = _token_jaccard(s1_toks, c_toks)
        name_token_overlap = _token_overlap(s1_toks, c_toks)
    else:
        name_lev = 0.0
        name_jaro = 0.0
        name_token_sort_ratio = 0.0
        name_token_set_ratio = 0.0
        name_wratio = 0.0
        name_token_jaccard = 0.0
        name_token_overlap = 0.0

    # Informative / rare token overlap
    info_a = set(s1_info)
    info_b = set(c_info)
    max_info = max(len(info_a), len(info_b))
    name_rare_token_overlap = len(info_a & info_b) / max_info if max_info > 0 else 0.0

    # Token and length differences
    max_toks = max(len(s1_toks), len(c_toks))
    name_token_count_diff = abs(len(s1_toks) - len(c_toks)) / max_toks if max_toks > 0 else 0.0
    max_len = max(len(s1_name), len(c_name))
    name_length_ratio = min(len(s1_name), len(c_name)) / max_len if max_len > 0 else 0.0

    # 2. Addresses
    s1_addr = _safe_str(s1_rec.get("addr_clean") or s1_rec.get("business_address"))
    c_addr = _safe_str(c_rec.get("addr_clean") or c_rec.get("business_address"))

    s1_addr_toks = _safe_list(s1_rec.get("addr_tokens"))
    c_addr_toks = _safe_list(c_rec.get("addr_tokens"))

    s1_pin = _safe_str(s1_rec.get("addr_pin"))
    c_pin = _safe_str(c_rec.get("addr_pin"))

    s1_num = _safe_str(s1_rec.get("addr_primary_num"))
    c_num = _safe_str(c_rec.get("addr_primary_num"))

    s1_m_val = s1_rec.get("addr_is_missing")
    c_m_val = c_rec.get("addr_is_missing")
    s1_missing = (s1_m_val is True or str(s1_m_val).lower() == 'true') or len(s1_addr) < 3
    c_missing = (c_m_val is True or str(c_m_val).lower() == 'true') or len(c_addr) < 3
    addr_either_missing = float(s1_missing or c_missing)

    both_have_addr = not (s1_missing or c_missing)
    if both_have_addr:
        addr_lev = distance.Levenshtein.normalized_similarity(s1_addr, c_addr)
        addr_token_sort_ratio = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        addr_token_jaccard = _token_jaccard(s1_addr_toks, c_addr_toks)
        max_addr_len = max(len(s1_addr), len(c_addr))
        addr_length_ratio = min(len(s1_addr), len(c_addr)) / max_addr_len if max_addr_len > 0 else 0.0
    else:
        addr_lev = 0.0
        addr_token_sort_ratio = 0.0
        addr_token_jaccard = 0.0
        addr_length_ratio = 0.0

    addr_pin_match = float(s1_pin == c_pin and s1_pin != "")
    addr_num_match = float(s1_num == c_num and s1_num != "")

    # 3. Transliteration / Cross-Script
    s1_trans = s1_rec.get("name_translit") or anyascii(s1_name).lower().strip()
    c_trans = c_rec.get("name_translit") or anyascii(c_name).lower().strip()

    if s1_trans and c_trans:
        name_translit_lev = distance.Levenshtein.normalized_similarity(s1_trans, c_trans)
        name_translit_exact = float(s1_trans == c_trans)
    else:
        name_translit_lev = 0.0
        name_translit_exact = 0.0

    # 4. Cross-Field Interactions
    s1_country = _safe_str(s1_rec.get("country")).lower()
    c_country = _safe_str(c_rec.get("country")).lower()
    country_match = float(s1_country == c_country and s1_country != "")

    cross_name_addr = name_lev * addr_lev
    name_high_addr_missing = float(name_lev > 0.85 and addr_either_missing == 1.0)
    name_exact_addr_match = float(name_exact == 1.0 and addr_lev > 0.60)

    # 5. Provenance Signals
    strat = str(strategies_str or "")
    prov_exact_name = 1.0 if "exact_name" in strat else 0.0
    prov_address = 1.0 if "address" in strat else 0.0
    prov_rare_token = 1.0 if "rare_token" in strat else 0.0
    prov_retrieval_reciprocal = 1.0 if ("char_retrieval" in strat or "reciprocal" in strat) else 0.0
    prov_transliteration = 1.0 if "transliteration" in strat else 0.0
    prov_count = float(prov_exact_name + prov_address + prov_rare_token + prov_retrieval_reciprocal + prov_transliteration)

    return {
        "name_exact": name_exact,
        "name_core_exact": name_core_exact,
        "name_sorted_exact": name_sorted_exact,
        "name_lev": name_lev,
        "name_jaro": name_jaro,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_wratio": name_wratio,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_rare_token_overlap": name_rare_token_overlap,
        "name_token_count_diff": name_token_count_diff,
        "name_length_ratio": name_length_ratio,
        "addr_lev": addr_lev,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_pin_match": addr_pin_match,
        "addr_num_match": addr_num_match,
        "addr_length_ratio": addr_length_ratio,
        "addr_either_missing": addr_either_missing,
        "name_translit_lev": name_translit_lev,
        "name_translit_exact": name_translit_exact,
        "country_match": country_match,
        "cross_name_addr": cross_name_addr,
        "name_high_addr_missing": name_high_addr_missing,
        "name_exact_addr_match": name_exact_addr_match,
        "prov_exact_name": prov_exact_name,
        "prov_address": prov_address,
        "prov_rare_token": prov_rare_token,
        "prov_retrieval_reciprocal": prov_retrieval_reciprocal,
        "prov_transliteration": prov_transliteration,
        "prov_count": prov_count
    }

# Backward compatibility alias
extract_features = extract_pairwise_features
