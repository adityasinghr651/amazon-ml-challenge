# Amazon ML Challenge 2026: Business Entity Resolution — Project State

> **Last Updated:** Step 0–6 Implementation (September 2026)  
> **Core Optimization Mandate:** **TWO-OBJECTIVE OPTIMIZATION**  
> 1. **High-Quality Entity Matching:** Macro-averaged $F_{0.5}$ (Precision-heavy: $2\times$ weight on precision; singletons protected)  
> 2. **Efficient, Scalable Blocking / Candidate Generation:** Smallest practical candidate set per $S_1$ entity while preserving high candidate recall ceiling and large reduction ratio ($>99.99\%$).  
> **Key Outputs:** `output/matching_results.tsv` (Leaderboard Score) & `output/candidate_pairs.tsv` (Audit & Final Ranking Evaluation)

---

## 1. Project Phase Tracker

```
┌─────────┬─────────────────────────────────────────────────┬────────────┬──────────────────────────────────────┐
│ Phase   │ Description                                     │ Status     │ Artifact / Key Output                │
├─────────┼─────────────────────────────────────────────────┼────────────┼──────────────────────────────────────┤
│ Phase 0 │ Environment & Repository Infrastructure         │ [DONE]     │ requirements.txt, config.yaml, utils │
│ Phase 1 │ Comprehensive Dataset Forensics & Noise Profile │ [DONE]     │ docs/phase_reports/phase_01_*.md     │
│ Phase 2 │ Multi-Representation Normalization Engine       │ [DONE]     │ docs/phase_reports/phase_02_*.md     │
│ Phase 3 │ High-Recall Multi-Blocker & candidate_pairs.tsv │ [DONE]     │ blocking.py (improved pruning)       │
│ Phase 4 │ Pairwise Feature Engineering (21 features)      │ [DONE]     │ features.py (TF-IDF fit-once)        │
│ Phase 5 │ Baseline ML Model (LR/RF/LightGBM)             │ [DONE]     │ training.py, run_phase5.py           │
│ Phase 6 │ Two-Band Threshold & Singleton Decision Layer   │ [DONE]     │ decision.py                          │
│ Phase 7 │ Final Inference, Validation PASS & Packaging    │ [READY]    │ inference.py (built, needs run)      │
└─────────┴─────────────────────────────────────────────────┴────────────┴──────────────────────────────────────┘
```

---

## 2. What Changed (Optimization Roadmap Implementation)

### Step 0 — Repo Unblocked
- `requirements.txt`: added `psutil`, `joblib`, `pyyaml`, pinned `lightgbm==4.5.0`
- `config.yaml`: central config with all hyperparameters (blocking, LightGBM, thresholds, splits)
- `utils.py`: config loader, experiment logger, blocking config extractor

### Step 1 — Fixed Candidate Pipeline
- `preprocessing.py`: reusable functions for ground-truth parsing, labeled-pair generation (anti-join pattern), entity-grouped 3-way splits (train/val/holdout)
- `blocking.py`: pruning now uses normalized fields (`name_core` tokens, `addr_pin` match bonus)
- `run_phase5.py`: complete rewrite — uses proper blocking (not baseline), no true-match injection
- `run_blocking_exp.py`: rewritten with entity-grouped 10% sampling, match/singleton decomposition, F0.5 ceiling

### Step 2 — Expanded Features (21 features)
- `features.py`: 10 name features, 6 address features, 3 cross-field features, 2 TF-IDF cosine features
- All features read from normalization output columns (not raw strings)
- TF-IDF vectorizers fit once via `fit_tfidf_vectorizers()`, shared across all pairs

### Step 3 — LightGBM + Hard Negatives
- `training.py`: `train_lgbm()` reads hyperparams from config.yaml
- `mine_hard_negatives()`: selects top-k most similar non-matching candidates per S1 (train split only)

### Step 4 — Two-Band Threshold
- `decision.py`: `predict_matches_two_band(t_high, t_low)` — borderline defaults to non-match
- `optimize_two_band_threshold()`: joint grid search on val split

### Step 5 — Inference Pipeline
- `inference.py`: imports exact same functions from training path, no reimplementation
- 1% profiling step before full run, with runtime/memory feasibility checks
- CLI-driven with argparse

### Step 6 — Error Analysis
- `evaluation.py`: `decomposed_error_analysis()` separates blocking misses from classifier mistakes
- `print_error_analysis()`: actionable recommendation (focus on blocking vs model)

---

## 3. Key Data Realities

| Metric / Dimension | Training Set | Testing Set |
| :--- | :--- | :--- |
| **Source 1 (Reference)** | 2,206,821 records | 1,732,544 records |
| **Source 2 (Noisy)** | 5,034,616 records | 4,887,273 records |
| **Source 3 (Noisy)** | 5,285,603 records | 5,082,316 records |
| **Ground Truth Links** | 7,638,365 pairs | *Unlabeled* |
| **Singletons (0 match)** | 123,247 (5.58%) | Unknown |
| **Country: France** | 0.00% (Unseen) | 14.98% |

---

## 4. Next Steps (Before Submission)

1. **Run `run_blocking_exp.py`** on 10% sample → get real blocking recall + candidate counts
2. **Run `run_phase5.py`** → train LightGBM on real candidates, get first trustworthy F0.5
3. **Run `inference.py`** on test data → generate submission files
4. **Validate** with `utils/validate_submission.py`
5. **Error analysis** on holdout split → decide whether to improve blocking or model
