"""
normalization.py - High-Performance Multi-Representation Normalization Engine

Preserves multi-script information (Indic Hindi/Tamil, French diacritics, Latin),
standardizes legal suffixes (US, India, France), removes URL/domain artifacts,
extracts structural address anchors (PINs, building numbers, road markers),
and computes token-sorted and core representations without information destruction.
"""
import re
import unicodedata
from typing import Dict, List, Any, Optional
import pandas as pd

# 1. Comprehensive Legal Suffix Mappings across US, India, and France
LEGAL_SUFFIX_MAP = {
    # India / Commonwealth
    r'\bpvt\s*\.?\s*ltd\s*\.?\b': 'private limited',
    r'\bpvt\s*\.?\b': 'private',
    r'\bltd\s*\.?\b': 'limited',
    r'\bllp\s*\.?\b': 'limited liability partnership',
    # US / Global
    r'\bcorp\s*\.?\b': 'corporation',
    r'\binc\s*\.?\b': 'incorporated',
    r'\bllc\s*\.?\b': 'limited liability company',
    r'\bco\s*\.?\b': 'company',
    # France (15% of test data)
    r'\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\s*\.?\b': 'societe a responsabilite limitee',
    r'\bs\s*\.?\s*a\s*\.?\s*s\s*\.?\s*u\s*\.?\b': 'societe par actions simplifiee unipersonnelle',
    r'\bs\s*\.?\s*a\s*\.?\s*s\s*\.?\b': 'societe par actions simplifiee',
    r'\bs\s*\.?\s*a\s*\.?\b': 'societe anonyme',
    r'\be\s*\.?\s*u\s*\.?\s*r\s*\.?\s*l\s*\.?\b': 'entreprise unipersonnelle a responsabilite limitee'
}

# Generic business words that should be excluded when generating informative token keys
GENERIC_BUSINESS_WORDS = {
    'incorporated', 'corporation', 'company', 'limited', 'private',
    'llc', 'inc', 'corp', 'ltd', 'pvt', 'llp', 'sarl', 'sas', 'sa', 'sasu', 'eurl',
    'and', 'the', 'of', 'in', 'at', 'by', 'for', 'with', 'a', 'an',
    'services', 'service', 'solutions', 'enterprises', 'enterprise',
    'holdings', 'holding', 'group', 'international', 'global', 'technologies',
    'consulting', 'management', 'store', 'shop', 'mart', 'market',
    'india', 'us', 'usa', 'france', 'center', 'centre'
}

# Address abbreviation mappings
ADDRESS_ABBREV_MAP = {
    r'\bst\s*\.?\b': 'street',
    r'\brd\s*\.?\b': 'road',
    r'\bave\s*\.?\b': 'avenue',
    r'\bblvd\s*\.?\b': 'boulevard',
    r'\bapt\s*\.?\b': 'apartment',
    r'\bbldg\s*\.?\b': 'building',
    r'\bste\s*\.?\b': 'suite',
    r'\bfl\s*\.?\b': 'floor',
    r'\bhwy\s*\.?\b': 'highway',
    r'\bln\s*\.?\b': 'lane',
    r'\bdr\s*\.?\b': 'drive',
    r'\bct\s*\.?\b': 'court',
    r'\bpl\s*\.?\b': 'place',
    r'\bpk\s*\.?\b': 'park',
    r'\bnagar\b': 'nagar',
    r'\bmarg\b': 'marg'
}

def clean_unicode_text(text: str) -> str:
    """
    Standardizes Unicode characters (NFKC) and converts to lowercase.
    Preserves all international scripts (Tamil, Devanagari, French accents)
    without lossy ASCII truncation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    norm = unicodedata.normalize('NFKC', text)
    return norm.lower().strip()

def strip_web_domains(text: str) -> str:
    """Removes web URL protocols and top-level domain extensions."""
    t = text
    t = re.sub(r'https?://', '', t)
    t = re.sub(r'\bwww\.', '', t)
    t = re.sub(r'\.(com|in|fr|org|net|co|io|biz|info)\b', '', t)
    return t

def normalize_name(name: str) -> Dict[str, Any]:
    """
    Normalizes business names into multi-representation structure:
      - raw: original string
      - clean_norm: lowercased, punctuation-normalized, unicode-preserved
      - suffix_norm: legal suffixes standardized
      - core_name: legal suffixes removed (for strong blocking & core alignment)
      - sorted_tokens: alphabetically sorted non-generic tokens (handles word reordering)
      - tokens: list of all cleaned word tokens
      - informative_tokens: high-information tokens excluding generic stop words
    """
    raw = str(name).strip() if isinstance(name, str) else ""
    if not raw:
        return {
            "raw": "",
            "clean_norm": "",
            "suffix_norm": "",
            "core_name": "",
            "sorted_tokens": "",
            "tokens": [],
            "informative_tokens": []
        }

    # 1. Universal Base Cleaning
    clean = clean_unicode_text(raw)
    clean = clean.replace('&', ' and ')
    clean = strip_web_domains(clean)
    
    # Standardize punctuation to spaces (Unicode-aware word character preservation)
    clean = re.sub(r'[^\w\s]', ' ', clean, flags=re.UNICODE)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # 2. Suffix Standardization
    suffix_norm = clean
    for pat, repl in LEGAL_SUFFIX_MAP.items():
        suffix_norm = re.sub(pat, repl, suffix_norm)
    suffix_norm = re.sub(r'\s+', ' ', suffix_norm).strip()

    # 3. Core Name (Complete Legal Suffix Removal)
    core_name = clean
    for pat in LEGAL_SUFFIX_MAP.keys():
        core_name = re.sub(pat, ' ', core_name)
    # Also strip expanded forms if present
    for exp_form in ['private limited', 'limited liability company', 'limited liability partnership',
                     'societe a responsabilite limitee', 'societe par actions simplifiee']:
        core_name = core_name.replace(exp_form, ' ')
    core_name = re.sub(r'\s+', ' ', core_name).strip()
    if not core_name:
        core_name = clean  # Fallback if name was only a suffix

    # 4. Token Ingestion & Sorting
    tokens = clean.split()
    informative_tokens = [t for t in tokens if t not in GENERIC_BUSINESS_WORDS and len(t) >= 2]
    
    tokens_to_sort = informative_tokens if informative_tokens else tokens
    sorted_tokens = " ".join(sorted(tokens_to_sort))

    return {
        "raw": raw,
        "clean_norm": clean,
        "suffix_norm": suffix_norm,
        "core_name": core_name,
        "sorted_tokens": sorted_tokens,
        "tokens": tokens,
        "informative_tokens": informative_tokens
    }

def normalize_address(address: str) -> Dict[str, Any]:
    """
    Normalizes business addresses and extracts structural components:
      - pin: 5 or 6 digit postal code
      - numbers: all numeric tokens (building, floor, street numbers)
      - primary_number: leading numeric anchor
      - is_missing: boolean indicator for missing addresses (~3.3% of target records)
      - clean_norm: standardized address text with unified road abbreviations
    """
    raw = str(address).strip() if isinstance(address, str) else ""
    if not raw or raw.lower() in {"nan", "null", "none"}:
        return {
            "raw": "",
            "clean_norm": "",
            "pin": "",
            "numbers": [],
            "primary_number": "",
            "tokens": [],
            "is_missing": True
        }

    clean = clean_unicode_text(raw)
    
    # 1. Extract PIN / Postal Code (US: 5 digits, India: 6 digits, France: 5 digits)
    pins = re.findall(r'\b\d{5,6}\b', clean)
    pin_str = pins[0] if pins else ""
    
    # 2. Extract all structural numeric tokens
    numbers = re.findall(r'\b\d+\b', clean)
    primary_num = numbers[0] if numbers else ""
    
    # 3. Punctuation standardization
    clean = re.sub(r'[^\w\s]', ' ', clean, flags=re.UNICODE)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # 4. Address road/street abbreviation standardization
    for pat, repl in ADDRESS_ABBREV_MAP.items():
        clean = re.sub(pat, repl, clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    tokens = clean.split()
    return {
        "raw": raw,
        "clean_norm": clean,
        "pin": pin_str,
        "numbers": numbers,
        "primary_number": primary_num,
        "tokens": tokens,
        "is_missing": False
    }

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the multi-representation normalization engine across a DataFrame.
    Expands clean, core, sorted, and structural anchor columns.
    """
    if df is None or df.empty:
        return df

    names_norm = df['business_name'].apply(normalize_name)
    addrs_norm = df['business_address'].apply(normalize_address)

    df_out = df.copy()
    
    df_out['name_clean'] = names_norm.apply(lambda x: x['clean_norm'])
    df_out['name_core'] = names_norm.apply(lambda x: x['core_name'])
    df_out['name_suffix_norm'] = names_norm.apply(lambda x: x['suffix_norm'])
    df_out['name_sorted'] = names_norm.apply(lambda x: x['sorted_tokens'])
    df_out['name_tokens'] = names_norm.apply(lambda x: x['tokens'])
    df_out['name_informative_tokens'] = names_norm.apply(lambda x: x['informative_tokens'])

    df_out['addr_clean'] = addrs_norm.apply(lambda x: x['clean_norm'])
    df_out['addr_pin'] = addrs_norm.apply(lambda x: x['pin'])
    df_out['addr_numbers'] = addrs_norm.apply(lambda x: x['numbers'])
    df_out['addr_primary_num'] = addrs_norm.apply(lambda x: x['primary_number'])
    df_out['addr_is_missing'] = addrs_norm.apply(lambda x: x['is_missing'])
    df_out['addr_tokens'] = addrs_norm.apply(lambda x: x['tokens'])

    if 'country' in df_out.columns:
        df_out['country_norm'] = df_out['country'].astype(str).str.strip().str.upper()

    return df_out
