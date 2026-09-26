# Phase 1: Dataset Forensics & Exploratory Analysis Report

> **Amazon ML Challenge 2026: Business Entity Resolution**  
> **Status:** Complete  
> **Date:** September 2026  
> **Dataset Location:** `dataset/` (Root)

---

## 1. Executive Summary & Core Observations

Phase 1 dataset forensics analyzed all 7 competition TSV files across the training and test splits. The findings directly establish the operational constraints and design requirements for the downstream blocking, feature engineering, and modeling stages:

1. **Massive Search Space (Scale Reality)**:
   - Training contains **2,206,821** Source 1 entities and over **10.3 million** records in Source 2 & Source 3.
   - Testing contains **1,732,544** Source 1 reference entities and over **9.96 million** records in Source 2 & Source 3.
   - Unconstrained Cartesian product space in testing exceeds **17.2 Trillion pairs**. Scalable, high-reduction blocking is essential.
2. **Ground Truth Structure & Match Multiplicity**:
   - **Singletons (0 matches)** represent **123,247 entities (5.58%)** of the training reference set. Under macro $F_{0.5}$, singletons must output empty string `""` (earning 1.0; predicting any false match gives 0.0).
   - **Multi-matches (94.42%)**: Entities with matches have an average of **3.66 matches** (max 11). The model must support multi-label candidate assignment per $S_1$ entity.
3. **Open-Set Country Domain Shift (France)**:
   - Training covers only `US` (59.98%) and `India` (40.02%).
   - Testing introduces `France` as **14.98% of $S_1$ entities (259,452 records)**, `India` as 46.75%, and `US` as 38.27%.
   - The pipeline must use **dynamic open-set string partitioning** by country without hard-coded country lists or one-hot vectors.
4. **Noise & Transliteration Archetypes**:
   - Indian entities in Source 2 and Source 3 frequently appear in native Indic scripts (Tamil, Hindi / Devanagari) or mixed script forms while Source 1 is in English Latin script.
   - Substantial legal suffix variations (`LLC`, `Inc`, `Pvt Ltd`, `Private Limited`, `SARL`).
   - Web domains substituted for business names (`companyname.com`).
   - Missing addresses present in ~2.7%–3.4% of Source 2 & Source 3 records.

---

## 2. Table-by-Table Forensic Profiles

### **Training Set**

#### A. Source 1 (Reference Entities) — `train_source1.tsv`
- **Total Records:** 2,206,821
- **Columns:** `entity_id, business_name, business_address, country`
- **Missing Values:** `entity_id`: 0 (0.0%), `business_name`: 0 (0.0%), `business_address`: 0 (0.0%), `country`: 0 (0.0%)
- **Name Length (chars):** Min=3, Mean=24.0, Median=24, P90=34, P95=37, P99=42, Max=105
- **Name Word Tokens:** Min=1, Mean=3.5, Median=4, P95=5, Max=16
- **Address Length (chars):** Min=11, Mean=52.1, Median=41, P90=90, P95=103, P99=124, Max=256
- **Address Word Tokens:** Min=2, Mean=8.0, Median=7, P95=15, Max=43
- **Country Distribution:** `US`: 1,323,633 (59.98%), `India`: 883,188 (40.02%)

#### B. Source 2 (Noisy Input) — `train_source2.tsv`
- **Total Records:** 5,034,616
- **Missing Values:** `business_address`: 168,967 empty (3.36%), all other columns 0% missing
- **Name Length (chars):** Min=2, Mean=25.1, Median=25, P90=37, P95=40, P99=48, Max=104
- **Address Length (chars):** Min=8, Mean=47.8, Median=37, P90=84, P95=97, P99=118, Max=249
- **Country Distribution:** `US`: 3,016,817 (59.92%), `India`: 2,017,799 (40.08%)

#### C. Source 3 (Noisy Input) — `train_source3.tsv`
- **Total Records:** 5,285,603
- **Missing Values:** `business_address`: 175,916 empty (3.33%), all other columns 0% missing
- **Name Length (chars):** Min=2, Mean=25.2, Median=25, P90=37, P95=42, P99=50, Max=123
- **Address Length (chars):** Min=2, Mean=48.3, Median=42, P90=78, P95=92, P99=116, Max=240
- **Country Distribution:** `US`: 3,170,056 (59.98%), `India`: 2,115,547 (40.02%)

---

### **Ground Truth Analysis — `train_ground_truth.tsv`**

- **Total S1 Reference Entities:** 2,206,821
- **Singletons (0 matches):** 123,247 (5.58%)
- **Entities with $\ge 1$ match:** 2,083,574 (94.42%)
- **Total Positive Match Links:** 7,638,365
- **Matches per S1 Record:** Mean=3.46, Median=3, Max=11

```
Histogram of Matches per S1 Entity:
  - 0 matches:  123,247 entities ( 5.58%)
  - 1 match:    119,157 entities ( 5.40%)
  - 2 matches:  375,212 entities (17.00%)
  - 3 matches:  530,841 entities (24.05%)
  - 4 matches:  484,115 entities (21.94%)
  - 5 matches:  321,957 entities (14.59%)
  - 6+ matches: 252,292 entities (11.43%)
```

- **Source 2 Match Coverage:** 3,693,619 true matches (73.36% of $S_2$). 1,340,997 records (26.64%) are non-matching distractors.
- **Source 3 Match Coverage:** 3,944,746 true matches (74.63% of $S_3$). 1,340,857 records (25.37%) are non-matching distractors.

---

### **Test Set Characterization**

#### A. Source 1 (Test Reference) — `test_source1.tsv`
- **Total Records:** 1,732,544
- **Missing Values:** 0% missing across all columns
- **Country Breakdown:**
  - `India`: 809,986 (46.75%)
  - `US`: 663,106 (38.27%)
  - `France`: 259,452 (14.98%)

#### B. Source 2 (Test Noisy) — `test_source2.tsv`
- **Total Records:** 4,887,273
- **Missing Values:** `business_address`: 129,408 empty (2.65%)
- **Country Breakdown:** `India`: 47.32%, `US`: 38.29%, `France`: 14.39%

#### C. Source 3 (Test Noisy) — `test_source3.tsv`
- **Total Records:** 5,082,316
- **Missing Values:** `business_address`: 136,098 empty (2.68%)
- **Country Breakdown:** `India`: 47.32%, `US`: 38.28%, `France`: 14.40%

---

## 3. Empirical Noise & Pattern Analysis

| S1 Entity | S1 Name | S1 Address | Matched ID | Matched Name | Matched Address | Identified Pattern |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S1-965667` | Maure Williams Colombier Inc | 85 Wayne Avenue, Ticonderoga, NY | `S2-681193310` | Maure Wilblims Colombier Inc | *(empty)* | Typo in Name, Missing Address |
| `S1-965667` | Maure Williams Colombier Inc | 85 Wayne Avenue, Ticonderoga, NY | `S3-11291185` | maurewilliamscolombier.com | Wayne Ave, Ticonderoga Townshiip, NY | URL format Name, Address Abbrev & Typo |
| `S1-55344266` | Raj Investments LLP | 6(29), C.I.T. Colony, 2Nd Main Road | `S2-249013014` | ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி | 6(29), C.I.T. COLONY, 2ND MAIN ROAD | Tamil Transliteration, Exact Address |
| `S1-55344266` | Raj Investments LLP | 6(29), C.I.T. Colony, 2Nd Main Road | `S3-478195123` | Raj Investments எல்எல்பி | 6(29), C.i.t. Colony, 2Nd Main Road | Mixed Script Name, Exact Address |
| `S1-343815751` | Dahlia Power Reliable Scientific LLC | 630 45th Terrace, Kansas City, MO | `S2-790675320` | Dahlia Power Reliable | KANSAS CITY, MO, 630 45ND TERRACE | Name Truncation, Address Reordering |
| `S1-656753428` | Ss Food Private Limited | Af-684, Nandgram Near Mother India | `S2-153058913` | एसएस फूड प्राइवेट लिमिटेड | AF-0684, NANDGRAM NEAR MOTHER INDIA | Hindi Devanagari Script, Zero-padded Number |
| `S1-656753428` | Ss Food Private Limited | Af-684, Nandgram Near Mother India | `S3-679606215` | एसएस फूड प्राइवेट लिमिटेड | Af-684, Ghaziabad, UP | Hindi Script, Landmark vs City Address |

---

## 4. Architectural & Algorithmic Implications

1. **Multi-Script & Character-Level Matching**:
   - Word token matching fails on Tamil/Hindi transliterations when compared against English reference names.
   - Address numbers (e.g. `6(29)`, `Af-684`, `85`) and city names serve as critical invariant anchors across script barriers.
   - Character 3-gram cosine similarity provides non-zero affinity across partial spelling variations.
2. **Missing Address Representation**:
   - Since ~3% of $S_2/S_3$ records lack addresses, missing address must NOT result in similarity = 0.0. The feature extractor must emit an explicit `is_address_missing` indicator feature and compute unpenalized name-only similarity branches.
3. **Hard Negative Mining Necessity**:
   - Because 25–27% of $S_2$ and $S_3$ are non-matching distractors sharing the same cities and generic words, the classifier must be trained on hard negatives mined from blocking collisions.
4. **Precision-Heavy Decision Threshold**:
   - Macro $F_{0.5}$ weights precision $2\times$ over recall. Low-confidence candidates must be rejected to protect singletons (5.58%) from scoring 0.0.
