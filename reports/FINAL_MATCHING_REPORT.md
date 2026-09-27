# Amazon ML Challenge 2026: Final Matching & Decision Layer Report

**Project**: Large-Scale Cross-Source Business Entity Resolution  
**Phase**: Phase 4 & Phase 5 — Pairwise Matching Model, Calibrated Decision Layer & Official Submission  
**Date**: September 27, 2026  
**Status**: **FROZEN & VALIDATED (Ready for Submission)**  
**Submission Artifact**: [`output/matching_results.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/matching_results.tsv)  
**Candidate Artifact**: [`output/candidate_pairs.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/candidate_pairs.tsv)  

---

## Executive Summary

We have completed the pairwise matching and decision layer for the Amazon ML Challenge 2026. Working under strict frozen candidate constraints (`output/candidate_pairs.tsv` at **90.12% pair recall** across 474,960 candidate pairs), we systematically engineered, benchmarked, and validated our matching pipeline across 4 rigorous stages:

1. **Stage 1 (Baselines & Feature Extraction)**: Extracted **Feature Set v1** (32 dense features spanning lexical, phonetic, cross-script transliteration, address token/pin overlap, and blocking provenance). Evaluated Logistic Regression (Plain Macro $F_{0.5} = 93.91\%$) and Random Forest (100 trees, Macro $F_{0.5} = 94.48\%$).
2. **Stage 2 (LightGBM Optimization & Ranking Analysis)**: Deployed LightGBM with 63 leaves and depth 8. Achieved **94.59% Macro $F_{0.5}$** at optimal threshold $\tau = 0.80$ with 10× faster inference than Random Forest. Empirically demonstrated that Top-1 exclusivity causes catastrophic collapse ($F_{0.5}$ dropped from 94.59% to 72.40%) due to real-world multi-match topology.
3. **Stage 3 (Hard-Negative Mining & 5-Fold CV)**: Conducted leak-free out-of-fold training error analysis. Proved that over **99.95%** of candidate negatives are naturally suppressed by LightGBM, and manual HNM produced negligible gain (+0.03% within variance noise). Executed **5-Fold Entity-Stratified Cross-Validation**, demonstrating a robust **95.00% Mean Macro $F_{0.5}$** ($\pm 0.27\%$), **96.90% Precision**, and **96.49% Singleton Accuracy**.
4. **Final Inference & Submission Generation**: Trained the frozen champion LightGBM model on the complete benchmark dataset, applied the calibrated threshold $\tau = 0.80$, generated `output/matching_results.tsv`, and verified 100% compliance via `utils/validate_submission.py` (**PASS, exit code 0**).

> **Important Metric Clarification**: The **95.00% Macro $F_{0.5}$** figure represents the **5-fold Cross-Validation mean** across entity-stratified benchmark folds, not a guaranteed final test leaderboard score. It demonstrates strong, low-variance generalization on unseen entities.

---

## 1. Frozen Candidate Generation Foundation

The upstream blocking pipeline was frozen without modification:
- **Pair Recall**: 90.12% (15,683 / 17,403 true matches recovered)
- **Candidate Pool**: 474,960 candidate pairs across 5,000 Source 1 entities (avg 94.99 candidates/entity)
- **Provenance**: Multi-strategy union of normalized exact name, address anchor, rare-token inverted index, character 3-gram TF-IDF retrieval, and cross-script transliteration.
- **Candidate Integrity**: Stored in [`output/candidate_pairs.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/candidate_pairs.tsv) and diagnostics in [`output/candidate_pairs_diagnostics.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/candidate_pairs_diagnostics.tsv).

---

## 2. Feature Set v1 Architecture (32 Features)

Every candidate pair $(S_1, S_{2/3})$ is vectorized into 32 leakage-safe features ([`src/business_entity_resolution/features.py`](file:///d:/ML%20Challenge%202026/amazon/src/business_entity_resolution/features.py)):

### A. Name Matchers (13 Features)
- Exact normalized equality (`name_exact`), core company root equality (`name_core_exact`), sorted tokens equality (`name_sorted_exact`).
- Fuzzy string distance ratios via RapidFuzz: normalized Levenshtein (`name_lev`), Jaro-Winkler (`name_jaro`), Token Sort Ratio (`name_token_sort_ratio`), Token Set Ratio (`name_token_set_ratio`), Weighted Ratio (`name_wratio`).
- Token set analytics: Jaccard overlap (`name_token_jaccard`), absolute token overlap count (`name_token_overlap`), rare token overlap (`name_rare_token_overlap`).
- Structural metrics: token length difference (`name_token_count_diff`), string length ratio (`name_length_ratio`).

### B. Address & Geospatial Matchers (7 Features)
- Normalized Levenshtein distance (`addr_lev`), token sort ratio (`addr_token_sort_ratio`), token Jaccard similarity (`addr_token_jaccard`).
- PIN / Postal Code exact match indicator (`addr_pin_match`).
- Street / Building number set overlap (`addr_num_match`).
- Address length ratio (`addr_length_ratio`), missing address flag (`addr_either_missing`).

### C. Cross-Script / Multilingual Matchers (2 Features)
- Transliterated Levenshtein similarity (`name_translit_lev`) via `anyascii`.
- Exact transliterated match flag (`name_translit_exact`).

### D. Cross-Field & Interaction Features (4 Features)
- Country code equality (`country_match`).
- Cross-field name-in-address containment (`cross_name_addr`).
- High-confidence name match with missing address flag (`name_high_addr_missing`).
- Exact name + address match interaction (`name_exact_addr_match`).

### E. Blocking Provenance (6 Features)
- One-hot provenance indicators: `prov_exact_name`, `prov_address`, `prov_rare_token`, `prov_retrieval_reciprocal`, `prov_transliteration`.
- Multi-strategy retrieval count (`prov_count`).

**Feature Quality Audit**: Verified via [`reports/feature_audit.tsv`](file:///d:/ML%20Challenge%202026/amazon/reports/feature_audit.tsv) with **0 missing values** and **0 infinities** across all 474,960 candidate pairs.

---

## 3. Progressive Model Evaluation Benchmark

All models were evaluated on the identical 80/20 entity-level split (4,000 train S1s / 1,000 val S1s, seed=42) preserving all singletons (62 singletons = 6.20%):

| Stage | Model Configuration | Optimal $\tau$ | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy | Matched Recall | Training Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 Baseline** | Logistic Regression (Plain) | 0.55 | 93.91% | 96.30% | 85.43% | 91.94% | 85.00% | 1.12s |
| **Stage 1 Baseline** | Logistic Regression (Balanced) | 0.95 | 93.78% | 95.80% | 86.47% | 88.71% | 86.32% | 1.87s |
| **Stage 1 Baseline** | Random Forest (100 trees, depth 10) | 0.60 | 94.48% | 96.77% | 86.32% | 93.55% | 85.84% | 38.54s |
| **Stage 2 Grid** | LightGBM (31 leaves, depth 6) | 0.80 | 94.55% | 96.67% | 86.97% | 91.94% | 86.64% | 3.55s |
| **Stage 2 Champion** | **LightGBM (63 leaves, depth 8)** | **0.80** | **94.59%** | **96.70%** | **87.00%** | **91.94%** | **86.68%** | **3.73s** |
| **Stage 2 Grid** | LightGBM (Regularized, L1/L2) | 0.75 | 94.55% | 96.69% | 86.91% | 91.94% | 86.58% | 4.88s |
| **Stage 2 Grid** | LightGBM (min_child_samples=100) | 0.80 | 94.57% | 96.69% | 86.97% | 91.94% | 86.64% | 3.52s |

---

## 4. Key Empirical Findings

### A. Top Feature Importance (by Gain)
The top discriminators identified by the winning LightGBM model ([`reports/stage2_feature_importances.tsv`](file:///d:/ML%20Challenge%202026/amazon/reports/stage2_feature_importances.tsv)):
1. `addr_token_jaccard` (Gain: 813,539.1) — **Dominant discriminator**: Resolves distractor pairs that share common company names.
2. `addr_token_sort_ratio` (Gain: 163,261.2) — Secondary address verification.
3. `name_jaro` (Gain: 32,233.1) — Primary name string distance metric.
4. `cross_name_addr` (Gain: 23,248.5) — Identifies cases where entity names appear within addresses.
5. `name_wratio` (Gain: 22,897.4) — Weighted ratio accounting for token reordering and length differences.
6. `prov_retrieval_reciprocal` (Gain: 8,110.1) — Reciprocal TF-IDF rank from candidate blocking.

### B. Decision Layer Analysis: Why Pure Thresholding Won
We tested three decision rules on the validation set ([`reports/stage2_decision_ranking.tsv`](file:///d:/ML%20Challenge%202026/amazon/reports/stage2_decision_ranking.tsv)):
- **Pure Calibrated Threshold ($\tau = 0.80$)**: **94.59% Macro $F_{0.5}$** (Optimal).
- **Top-1 Only Requirement**: **Collapsed to 72.40% Macro $F_{0.5}$** (Recall fell from 87.00% to 35.89%). In commercial entity resolution, a single legal entity frequently owns multiple branches, locations, and subsidiary records in Source 2 and Source 3. Enforcing 1-to-1 matching destroys multi-match recall.
- **Top-1 + Margin Rules**: Requiring secondary matches to be within a tight delta $\Delta < 0.20$ of the top candidate discarded valid multi-matches without improving precision. Pure probability calibration remains mathematically superior.

### C. Hard-Negative Mining Findings
Out of 367,711 training negative pairs, only 178 false positives had OOF probabilities $\ge 0.50$ (only 8 singleton FPs). Upweighting or duplicating hard negatives shifted the optimal threshold ($\tau = 0.70$–$0.75$) but did not yield statistically meaningful improvements (+0.03% to -0.11%). Per experimental protocol, HNM was stopped to avoid threshold instability.

---

## 5. 5-Fold Entity-Stratified Cross-Validation Results

Full 5-fold entity-isolated cross-validation was conducted across all 5,000 Source 1 benchmark entities ([`reports/stage3_cv_summary.tsv`](file:///d:/ML%20Challenge%202026/amazon/reports/stage3_cv_summary.tsv)):

| Fold | Train Entities | Val Entities | Macro $F_{0.5}$ | Precision | Recall | Singleton Accuracy | Matched Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 4,000 | 1,000 | 94.59% | 96.70% | 87.00% | 91.94% | 86.68% |
| **Fold 2** | 4,000 | 1,000 | 94.91% | 96.67% | 88.44% | 98.08% | 87.91% |
| **Fold 3** | 4,000 | 1,000 | 95.30% | 97.38% | 87.81% | 100.00% | 87.10% |
| **Fold 4** | 4,000 | 1,000 | 95.09% | 96.87% | 88.56% | 98.31% | 87.95% |
| **Fold 5** | 4,000 | 1,000 | 95.09% | 96.89% | 88.51% | 94.12% | 88.21% |
| **Mean** | **4,000** | **1,000** | **95.00%** | **96.90%** | **88.06%** | **96.49%** | **87.57%** |
| **Std ($\sigma$)** | — | — | **0.27%** | **0.28%** | **0.67%** | **3.33%** | **0.65%** |

- **Cross-Validation Macro $F_{0.5}$ Mean**: **95.00%** ($\pm 0.27\%$).
- **Cross-Validation Precision Mean**: **96.90%** ($\pm 0.28\%$), demonstrating high precision alignment with the official competition metric.
- **Cross-Validation Singleton Accuracy Mean**: **96.49%** ($\pm 3.33\%$).

---

## 6. Official Submission Verification & Integrity Audit

The final submission file [`output/matching_results.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/matching_results.tsv) was generated via `src/business_entity_resolution/generate_final_submission.py`.

### A. Output Distribution
- **Total S1 Entities**: Exactly 5,000 rows.
- **Predicted Singletons (no matches, empty string)**: 380 entities (7.60%).
- **Predicted Matched Entities**: 4,620 entities (92.40%).
- **Total Matched Pairs**: 15,560 pairs (average 3.37 matches per matched S1).

### B. Integrity Audit Checklist
- [x] **Entity Coverage**: All 5,000 S1 entities represented exactly once in deterministic order.
- [x] **Schema Compliance**: Tab-separated format with header `source1_entity_id\tmatched_entity_ids`.
- [x] **Candidate Containment**: 100% of predicted matches originate from frozen `output/candidate_pairs.tsv` (0 unauthorized pairs).
- [x] **ID Integrity**: All matched IDs possess valid `S2-` or `S3-` prefixes. 0 `S1-` self-matches.
- [x] **Uniqueness**: 0 duplicate S1 rows, 0 duplicate candidate IDs within any row.
- [x] **Clean Data**: 0 NaN, None, or infinite values.
- [x] **Official Scorer Validation**: Executed `utils/validate_submission.py`:
  ```
  ML Challenge 2026 — submission validator
    test dir: dataset/benchmark_test_shim
    required S1 entities: 5000
    matching_results.tsv: 5000 rows (380 empty, 4620 non-empty).
    candidate_pairs.tsv: 5000 rows (0 empty, 5000 non-empty).
  PASS — no blocking issues found. Safe to submit.
  ```

---

## 7. Submission Package Manifest

The final submission package consists of:
1. [`output/matching_results.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/matching_results.tsv) — Final matching decisions (the scored leaderboard file).
2. [`output/candidate_pairs.tsv`](file:///d:/ML%20Challenge%202026/amazon/output/candidate_pairs.tsv) — Frozen candidate pairs (90.12% blocking recall).
3. [`configs/final_matching.yaml`](file:///d:/ML%20Challenge%202026/amazon/configs/final_matching.yaml) — Production configuration parameters.
4. [`experiments/experiment_log.csv`](file:///d:/ML%20Challenge%202026/amazon/experiments/experiment_log.csv) — Complete audit trail of all 148 benchmark evaluations.
5. [`src/business_entity_resolution/`](file:///d:/ML%20Challenge%202026/amazon/src/business_entity_resolution/) — Clean, modular, reproducible Python codebase.
