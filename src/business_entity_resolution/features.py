"""
Phase 4: features.py
Pairwise feature engineering between S1 and S2/S3 candidates.

All features read from normalization.py's output columns:
  name_clean, name_core, name_sorted, name_tokens, name_informative_tokens,
  addr_clean, addr_pin, addr_primary_num, addr_is_missing, addr_tokens.

TF-IDF vectorizers are fitted once externally via fit_tfidf_vectorizers()
and passed in — never refitted per pair.
"""

import numpy as np
import rapidfuzz
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


# ============================================================
# TF-IDF Vectorizer Management (fit once, use many)
# ============================================================

def fit_tfidf_vectorizers(
    all_name_strings: List[str],
    all_addr_strings: List[str],
    name_cfg: Optional[Dict] = None,
    addr_cfg: Optional[Dict] = None,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """
    Fit TF-IDF vectorizers over the full corpus of unique normalized strings.
    Call this ONCE on the training split, then use the fitted vectorizers
    for both train and val/holdout transforms.

    Parameters
    ----------
    all_name_strings : list of all normalized name strings (name_clean).
    all_addr_strings : list of all normalized address strings (addr_clean).
    name_cfg, addr_cfg : dicts with keys analyzer, ngram_range, max_df, min_df.

    Returns
    -------
    (name_vectorizer, addr_vectorizer) — fitted TfidfVectorizer instances.
    """
    ncfg = name_cfg or {"analyzer": "char_wb", "ngram_range": [2, 4], "max_df": 0.9, "min_df": 2}
    acfg = addr_cfg or {"analyzer": "char_wb", "ngram_range": [2, 4], "max_df": 0.9, "min_df": 2}

    name_vec = TfidfVectorizer(
        analyzer=ncfg["analyzer"],
        ngram_range=tuple(ncfg["ngram_range"]),
        max_df=ncfg["max_df"],
        min_df=ncfg["min_df"],
    )
    addr_vec = TfidfVectorizer(
        analyzer=acfg["analyzer"],
        ngram_range=tuple(acfg["ngram_range"]),
        max_df=acfg["max_df"],
        min_df=acfg["min_df"],
    )

    # Filter empty strings — TF-IDF can't handle them
    name_corpus = [s for s in all_name_strings if s.strip()]
    addr_corpus = [s for s in all_addr_strings if s.strip()]

    if name_corpus:
        name_vec.fit(name_corpus)
    if addr_corpus:
        addr_vec.fit(addr_corpus)

    return name_vec, addr_vec


def _sparse_cosine_sim(v1: csr_matrix, v2: csr_matrix) -> float:
    """Cosine similarity between two sparse row vectors."""
    dot = v1.dot(v2.T).toarray().item()
    n1 = np.sqrt(v1.dot(v1.T).toarray().item())
    n2 = np.sqrt(v2.dot(v2.T).toarray().item())
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _token_jaccard(tokens_a: list, tokens_b: list) -> float:
    """Jaccard similarity between two token lists."""
    if not tokens_a and not tokens_b:
        return 0.0
    set_a = set(tokens_a) if isinstance(tokens_a, list) else set()
    set_b = set(tokens_b) if isinstance(tokens_b, list) else set()
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def _token_overlap(tokens_a: list, tokens_b: list) -> float:
    """Overlap coefficient: |A∩B| / min(|A|, |B|)."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a) if isinstance(tokens_a, list) else set()
    set_b = set(tokens_b) if isinstance(tokens_b, list) else set()
    min_len = min(len(set_a), len(set_b))
    if min_len == 0:
        return 0.0
    return len(set_a & set_b) / min_len


def _safe_str(val) -> str:
    """Safely convert to string, handling NaN/None."""
    if val is None:
        return ""
    s = str(val)
    return "" if s.lower() in ("nan", "none", "") else s


def _safe_list(val) -> list:
    """Safely convert to list."""
    if isinstance(val, list):
        return val
    return []


# ============================================================
# Main Feature Extraction
# ============================================================

def extract_features(
    s1_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    name_vectorizer: Optional[TfidfVectorizer] = None,
    addr_vectorizer: Optional[TfidfVectorizer] = None,
) -> Dict[str, float]:
    """
    Extract 20+ pairwise similarity features from normalized record dicts.

    Parameters
    ----------
    s1_record, candidate_record : dicts with normalized columns from process_dataframe().
    name_vectorizer, addr_vectorizer : fitted TF-IDF vectorizers (optional).

    Returns
    -------
    Dict of feature_name -> float value. Feature names are deterministic and sorted.
    """
    # --- Get normalized fields ---
    s1_name_clean = _safe_str(s1_record.get("name_clean", ""))
    c_name_clean = _safe_str(candidate_record.get("name_clean", ""))

    s1_name_core = _safe_str(s1_record.get("name_core", ""))
    c_name_core = _safe_str(candidate_record.get("name_core", ""))

    s1_name_sorted = _safe_str(s1_record.get("name_sorted", ""))
    c_name_sorted = _safe_str(candidate_record.get("name_sorted", ""))

    s1_name_tokens = _safe_list(s1_record.get("name_tokens", []))
    c_name_tokens = _safe_list(candidate_record.get("name_tokens", []))

    s1_informative = _safe_list(s1_record.get("name_informative_tokens", []))
    c_informative = _safe_list(candidate_record.get("name_informative_tokens", []))

    s1_addr_clean = _safe_str(s1_record.get("addr_clean", ""))
    c_addr_clean = _safe_str(candidate_record.get("addr_clean", ""))

    s1_addr_tokens = _safe_list(s1_record.get("addr_tokens", []))
    c_addr_tokens = _safe_list(candidate_record.get("addr_tokens", []))

    s1_addr_pin = _safe_str(s1_record.get("addr_pin", ""))
    c_addr_pin = _safe_str(candidate_record.get("addr_pin", ""))

    s1_addr_num = _safe_str(s1_record.get("addr_primary_num", ""))
    c_addr_num = _safe_str(candidate_record.get("addr_primary_num", ""))

    s1_addr_missing = bool(s1_record.get("addr_is_missing", False))
    c_addr_missing = bool(candidate_record.get("addr_is_missing", False))

    s1_country = _safe_str(s1_record.get("country", "")).lower().strip()
    c_country = _safe_str(candidate_record.get("country", "")).lower().strip()

    features = {}

    # ---- NAME FEATURES (10) ----
    # 1. Exact match on cleaned name
    features["name_exact"] = float(s1_name_clean == c_name_clean and s1_name_clean != "")
    # 2. Normalized Levenshtein on clean name
    features["name_lev"] = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_name_clean, c_name_clean)
    # 3. Jaro-Winkler on clean name
    features["name_jaro"] = rapidfuzz.distance.JaroWinkler.similarity(s1_name_clean, c_name_clean)
    # 4. Core name (suffix-stripped) Levenshtein
    features["name_core_lev"] = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_name_core, c_name_core)
    # 5. Sorted-token name Levenshtein (handles word-order permutations)
    features["name_sorted_lev"] = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_name_sorted, c_name_sorted)
    # 6. Name token Jaccard
    features["name_token_jaccard"] = _token_jaccard(s1_name_tokens, c_name_tokens)
    # 7. Name token overlap coefficient
    features["name_token_overlap"] = _token_overlap(s1_name_tokens, c_name_tokens)
    # 8. Rare/informative token overlap ratio
    s1_info_set = set(s1_informative) if s1_informative else set()
    c_info_set = set(c_informative) if c_informative else set()
    max_info = max(len(s1_info_set), len(c_info_set))
    features["name_rare_token_overlap"] = len(s1_info_set & c_info_set) / max_info if max_info > 0 else 0.0
    # 9. Token count difference (normalized)
    max_toks = max(len(s1_name_tokens), len(c_name_tokens))
    features["name_token_count_diff"] = abs(len(s1_name_tokens) - len(c_name_tokens)) / max_toks if max_toks > 0 else 0.0
    # 10. Name length ratio
    max_len = max(len(s1_name_clean), len(c_name_clean))
    features["name_length_ratio"] = min(len(s1_name_clean), len(c_name_clean)) / max_len if max_len > 0 else 0.0

    # ---- ADDRESS FEATURES (6) ----
    # 11. Address Levenshtein
    if s1_addr_clean and c_addr_clean:
        features["addr_lev"] = rapidfuzz.distance.Levenshtein.normalized_similarity(s1_addr_clean, c_addr_clean)
    else:
        features["addr_lev"] = 0.0
    # 12. Address token Jaccard
    features["addr_token_jaccard"] = _token_jaccard(s1_addr_tokens, c_addr_tokens)
    # 13. PIN/postal code exact match
    features["addr_pin_match"] = float(s1_addr_pin == c_addr_pin and s1_addr_pin != "")
    # 14. Primary number (house/building) exact match
    features["addr_num_match"] = float(s1_addr_num == c_addr_num and s1_addr_num != "")
    # 15. Address length ratio
    max_addr_len = max(len(s1_addr_clean), len(c_addr_clean))
    features["addr_length_ratio"] = min(len(s1_addr_clean), len(c_addr_clean)) / max_addr_len if max_addr_len > 0 else 0.0
    # 16. Either address missing indicator
    features["addr_either_missing"] = float(s1_addr_missing or c_addr_missing)

    # ---- CROSS-FIELD FEATURES (3) ----
    # 17. Country exact match
    features["country_match"] = float(s1_country == c_country and s1_country != "")
    # 18. Name similarity × Address similarity (cross signal)
    features["cross_name_addr"] = features["name_lev"] * features["addr_lev"]
    # 19. High name similarity + missing address (special regime)
    features["name_high_addr_missing"] = float(features["name_lev"] > 0.8 and (s1_addr_missing or c_addr_missing))

    # ---- TF-IDF COSINE FEATURES (2, optional) ----
    if name_vectorizer is not None and s1_name_clean and c_name_clean:
        try:
            v1 = name_vectorizer.transform([s1_name_clean])
            v2 = name_vectorizer.transform([c_name_clean])
            features["name_tfidf_cosine"] = _sparse_cosine_sim(v1, v2)
        except Exception:
            features["name_tfidf_cosine"] = 0.0
    else:
        features["name_tfidf_cosine"] = 0.0

    if addr_vectorizer is not None and s1_addr_clean and c_addr_clean:
        try:
            v1 = addr_vectorizer.transform([s1_addr_clean])
            v2 = addr_vectorizer.transform([c_addr_clean])
            features["addr_tfidf_cosine"] = _sparse_cosine_sim(v1, v2)
        except Exception:
            features["addr_tfidf_cosine"] = 0.0
    else:
        features["addr_tfidf_cosine"] = 0.0

    return features


def extract_features_batch(
    pairs_df,
    records_db: Dict[str, Dict],
    name_vectorizer: Optional[TfidfVectorizer] = None,
    addr_vectorizer: Optional[TfidfVectorizer] = None,
):
    """
    Extract features for a DataFrame of (source1_entity_id, candidate_entity_id, label) pairs.

    Returns
    -------
    X : np.ndarray of shape (n_pairs, n_features)
    y : np.ndarray of shape (n_pairs,)
    meta : list of (s1_id, c_id) tuples
    feature_names : list of feature name strings (sorted)
    """
    X, y, meta = [], [], []
    feature_names = None

    for _, row in pairs_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        c_id = str(row["candidate_entity_id"])
        label = int(row.get("label", 0))

        if s1_id not in records_db or c_id not in records_db:
            continue

        feat = extract_features(
            records_db[s1_id], records_db[c_id],
            name_vectorizer=name_vectorizer,
            addr_vectorizer=addr_vectorizer,
        )

        if feature_names is None:
            feature_names = sorted(feat.keys())

        X.append([feat[k] for k in feature_names])
        y.append(label)
        meta.append((s1_id, c_id))

    if not X:
        return np.array([]).reshape(0, 0), np.array([]), [], []

    return np.array(X), np.array(y), meta, feature_names
