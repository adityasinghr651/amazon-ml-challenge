"""
Phase 2: normalization.py
Implements the Normalization Engine for both names and addresses.
Builds multiple representations without destroying original information.
"""

import re
import unicodedata

def clean_text_base(text: str) -> str:
    """Applies universal base cleaning: unicode normalization & lowercase."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Unicode normalization to remove accents and standardize characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.lower()

def normalize_name(name: str) -> dict:
    """
    Normalizes a business name and returns multiple representations.
    """
    raw = str(name).strip() if isinstance(name, str) else ""
    if not raw:
        return {"raw": "", "normalized": "", "sorted_tokens": "", "tokens": []}
    
    norm = clean_text_base(raw)
    
    # 1. & ↔ and normalization
    norm = norm.replace('&', ' and ')
    
    # 2. Punctuation normalization (replace non-alphanumeric with spaces)
    norm = re.sub(r'[^a-z0-9]', ' ', norm)
    
    # 3. Whitespace normalization
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    # 4. Legal suffix abbreviation normalization
    suffix_map = {
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bco\b': 'company',
        r'\bltd\b': 'limited',
        r'\bllc\b': 'limited liability company',
        r'\bpvt ltd\b': 'private limited',
        r'\bpvt\b': 'private'
    }
    for pat, repl in suffix_map.items():
        norm = re.sub(pat, repl, norm)
        
    # Final whitespace cleanup just in case
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    tokens = norm.split()
    sorted_tokens = " ".join(sorted(tokens))
    
    return {
        "raw": raw,
        "normalized": norm,
        "sorted_tokens": sorted_tokens,
        "tokens": tokens
    }

def normalize_address(address: str) -> dict:
    """
    Normalizes a business address and extracts structural components (PIN, numbers).
    """
    raw = str(address).strip() if isinstance(address, str) else ""
    if not raw:
        return {"raw": "", "normalized": "", "pin": "", "numbers": "", "tokens": []}
    
    norm = clean_text_base(raw)
    
    # Extract potential PIN/postal codes (5-6 digits) before removing all punctuation
    # PIN codes are usually 5 digits in US or 6 digits in India
    pins = re.findall(r'\b\d{5,6}\b', norm)
    pin_str = " ".join(pins)
    
    # Extract all numeric tokens (building numbers, street numbers, etc.)
    # We do this before substituting abbreviations to capture numbers clearly
    numbers = " ".join(re.findall(r'\b\d+\b', norm))
    
    # Punctuation to space
    norm = re.sub(r'[^a-z0-9]', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    # Common address abbreviation handling
    abbrev_map = {
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bapt\b': 'apartment',
        r'\bbldg\b': 'building',
        r'\bste\b': 'suite',
        r'\bfl\b': 'floor',
        r'\bhwy\b': 'highway'
    }
    for pat, repl in abbrev_map.items():
        norm = re.sub(pat, repl, norm)
        
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    return {
        "raw": raw,
        "normalized": norm,
        "pin": pin_str,
        "numbers": numbers,
        "tokens": norm.split()
    }

def process_dataframe(df):
    """
    Applies the normalization engine to a dataframe, returning a new dataframe
    with expanded feature columns.
    """
    import pandas as pd
    
    if df is None or df.empty:
        return df
        
    # Apply normalization to the entire series
    names_norm = df['business_name'].apply(normalize_name)
    addrs_norm = df['business_address'].apply(normalize_address)
    
    # Expand dictionaries into DataFrame columns
    df_out = df.copy()
    
    df_out['name_norm'] = names_norm.apply(lambda x: x['normalized'])
    df_out['name_sorted'] = names_norm.apply(lambda x: x['sorted_tokens'])
    df_out['name_tokens'] = names_norm.apply(lambda x: x['tokens'])
    
    df_out['addr_norm'] = addrs_norm.apply(lambda x: x['normalized'])
    df_out['addr_pin'] = addrs_norm.apply(lambda x: x['pin'])
    df_out['addr_numbers'] = addrs_norm.apply(lambda x: x['numbers'])
    df_out['addr_tokens'] = addrs_norm.apply(lambda x: x['tokens'])
    
    if 'country' in df_out.columns:
        df_out['country_norm'] = df_out['country'].astype(str).str.strip().str.upper()
        
    return df_out
