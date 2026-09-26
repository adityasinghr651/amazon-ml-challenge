# Phase 2: Multi-Representation Normalization Engine Report

> **Amazon ML Challenge 2026: Business Entity Resolution**  
> **Status:** Complete  
> **Date:** September 2026  
> **Module:** [`src/business_entity_resolution/normalization.py`](file:///d:/ML%20Challenge%202026/amazon/src/business_entity_resolution/normalization.py)

---

## 1. Objectives & Architectural Principles

Phase 2 builds a high-performance Normalization Engine designed to produce multiple standardized representations of names and addresses **without destroying discriminative identity information**.

Key design principles established:
1. **Unicode Script & Diacritic Preservation**:
   - Standardized via `unicodedata.normalize('NFKC', text)`.
   - Explicitly avoids destructive ASCII downcasting (`encode('ascii', 'ignore')`), ensuring Indian native scripts (Tamil, Hindi / Devanagari) and French accents (`é`, `è`, `ç`, `ô`) retain full integrity.
2. **Multi-Region Legal Suffix Handling**:
   - Standardizes suffixes across all target jurisdictions:
     - **US**: `inc` $\rightarrow$ `incorporated`, `corp` $\rightarrow$ `corporation`, `llc` $\rightarrow$ `limited liability company`, `co` $\rightarrow$ `company`.
     - **India**: `pvt ltd` / `pvt. ltd.` $\rightarrow$ `private limited`, `llp` $\rightarrow$ `limited liability partnership`.
     - **France (15% of Test Data)**: `sarl` $\rightarrow$ `societe a responsabilite limitee`, `sas`, `sasu`, `sa`, `eurl`.
   - Computes two complementary representations:
     - `suffix_norm`: legal forms expanded and standardized.
     - `core_name`: legal suffixes completely stripped to align entities when one source includes a legal form and another omits it.
3. **Web Domain / URL Stripping**:
   - Strips `http(s)://`, `www.`, and domain extensions (`.com`, `.in`, `.fr`, `.org`, `.net`) to match entities formatted as URLs (`maurewilliamscolombier.com` $\rightarrow$ `maurewilliamscolombier`).
4. **Token Ordering Invariance**:
   - Computes `sorted_tokens` from informative tokens to neutralize word-order transpositions (`Reliable Scientific Power` $\leftrightarrow$ `Power Reliable Scientific`).
5. **Address Structural Anchor Extraction**:
   - Extracts 5–6 digit postal/PIN codes (`pin`).
   - Extracts building, plot, and street numeric tokens (`numbers`, `primary_number`).
   - Standardizes road/street abbreviations (`rd` $\rightarrow$ `road`, `st` $\rightarrow$ `street`, `ave` $\rightarrow$ `avenue`, `blvd` $\rightarrow$ `boulevard`, `nagar`, `marg`).
   - Explicitly tracks `is_missing` boolean flag (~3.3% of target records lack addresses).

---

## 2. Output Schema & Representations

```
Input Record:
  business_name:    "Raj Investments Pvt. Ltd."
  business_address: "6(29), C.I.T. Colony, 2nd Main Rd, Chennai 600004"
  country:          "India"

Normalized Representations:
  name_clean:              "raj investments pvt ltd"
  name_core:               "raj investments"
  name_suffix_norm:        "raj investments private limited"
  name_sorted:             "investments raj"
  name_informative_tokens: ["raj", "investments"]
  
  addr_clean:              "6 29 c i t colony 2nd main road chennai 600004"
  addr_pin:                "600004"
  addr_numbers:            ["6", "29", "2", "600004"]
  addr_primary_num:        "6"
  addr_is_missing:         False
  country_norm:            "INDIA"
```

---

## 3. Unit Test Verification

Automated unit tests in [`tests/test_normalization.py`](file:///d:/ML%20Challenge%202026/amazon/tests/test_normalization.py) verified:
- [x] Unicode script preservation (Tamil, Hindi, French accents).
- [x] Legal suffix expansion and core name extraction across US, India, and France.
- [x] Web domain and URL prefix/suffix removal.
- [x] Informative token alphabetical sorting.
- [x] Structural PIN and numeric address anchor extraction.
- [x] Missing address boolean flag assignment.

---

## 4. Downstream Utility for Phase 3 Blocking

These normalized representations directly feed the upcoming blocking keys:
- **Block A**: Exact match on `name_clean`, `name_core`, and `name_sorted`.
- **Block B**: Inverted index on `name_informative_tokens` (bounded by document frequency).
- **Block C**: Structural composite keys (`addr_pin + "_" + name_prefix`, `addr_primary_num + "_" + name_prefix`).
