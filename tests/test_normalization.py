"""
tests/test_normalization.py - Unit tests for Phase 2 Normalization Engine
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.business_entity_resolution.normalization import (
    normalize_name,
    normalize_address,
    clean_unicode_text,
    strip_web_domains
)

def test_unicode_preservation():
    # Tamil name should NOT become empty
    tamil_name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
    norm = normalize_name(tamil_name)
    assert len(norm["clean_norm"]) > 0, "Tamil script was destroyed by normalization!"
    
    # Hindi name should NOT become empty
    hindi_name = "एसएस फूड प्राइवेट लिमिटेड"
    norm_hindi = normalize_name(hindi_name)
    assert len(norm_hindi["clean_norm"]) > 0, "Hindi script was destroyed by normalization!"
    
    # French accents preserved/standardized
    french_name = "Dréxkor S.A.R.L."
    norm_fr = normalize_name(french_name)
    assert "drexkor" in norm_fr["clean_norm"] or "dréxkor" in norm_fr["clean_norm"]
    assert "societe a responsabilite limitee" in norm_fr["suffix_norm"]

def test_legal_suffix_normalization():
    # US suffixes
    res = normalize_name("Acme Corp.")
    assert "corporation" in res["suffix_norm"]
    assert res["core_name"] == "acme"
    
    # India suffixes
    res_in = normalize_name("Raj Investments Pvt. Ltd.")
    assert "private limited" in res_in["suffix_norm"]
    assert res_in["core_name"] == "raj investments"
    
    res_llp = normalize_name("Apex Logistics LLP")
    assert "limited liability partnership" in res_llp["suffix_norm"]
    assert res_llp["core_name"] == "apex logistics"
    
    # France suffixes
    res_fr = normalize_name("Boulangerie Moderne SAS")
    assert "societe par actions simplifiee" in res_fr["suffix_norm"]
    assert res_fr["core_name"] == "boulangerie moderne"

def test_web_domain_stripping():
    res = normalize_name("maurewilliamscolombier.com")
    assert res["clean_norm"] == "maurewilliamscolombier"
    
    res_www = normalize_name("www.globaltech.in")
    assert res_www["clean_norm"] == "globaltech"

def test_token_sorting():
    res1 = normalize_name("Reliable Scientific Power")
    res2 = normalize_name("Power Reliable Scientific")
    assert res1["sorted_tokens"] == res2["sorted_tokens"]

def test_address_normalization_and_anchors():
    addr = "6(29), C.I.T. Colony, 2nd Main Road, Chennai 600004"
    norm_addr = normalize_address(addr)
    
    assert norm_addr["pin"] == "600004"
    assert "6" in norm_addr["numbers"] and "29" in norm_addr["numbers"]
    assert "road" in norm_addr["clean_norm"]
    assert not norm_addr["is_missing"]

def test_missing_address():
    norm_empty = normalize_address("")
    assert norm_empty["is_missing"] is True
    assert norm_empty["clean_norm"] == ""
    assert norm_empty["pin"] == ""
