# Amazon ML Challenge 2026: Business Entity Resolution — Project State

> **Last Updated:** Phase 1 Complete (September 2026)  
> **Target Metric:** Macro-averaged $F_{0.5}$ (Precision-heavy: $2\times$ precision weight)  
> **Key Outputs:** `output/matching_results.tsv` (Leaderboard), `output/candidate_pairs.tsv` (Audit/Final Ranking)

---

## 1. Project Phase Tracker

```
┌─────────┬─────────────────────────────────────────────────┬────────────┬──────────────────────────────────────┐
│ Phase   │ Description                                     │ Status     │ Artifact / Key Output                │
├─────────┼─────────────────────────────────────────────────┼────────────┼──────────────────────────────────────┤
│ Phase 0 │ Environment & Repository Infrastructure         │ [DONE]     │ requirements.txt, io.py, evaluation.py│
│ Phase 1 │ Comprehensive Dataset Forensics & Noise Profile │ [DONE]     │ docs/phase_reports/phase_01_*.md     │
│ Phase 2 │ Multi-Representation Normalization Engine       │ [CURRENT]  │ src/business_entity_resolution/norm* │
│ Phase 3 │ High-Recall Multi-Blocker & candidate_pairs.tsv  │ [PENDING]  │ src/business_entity_resolution/block*│
│ Phase 4 │ Pairwise Feature Engineering (Vectorized)       │ [PENDING]  │ src/business_entity_resolution/feat* │
│ Phase 5 │ Baseline ML Model & Hard Negative Mining        │ [PENDING]  │ src/business_entity_resolution/train*│
│ Phase 6 │ Macro F0.5 Threshold & Singleton Decision Layer │ [PENDING]  │ src/business_entity_resolution/dec*  │
│ Phase 7 │ Final Inference, Validation PASS & Packaging    │ [PENDING]  │ <team>_submission.zip                │
└─────────┴─────────────────────────────────────────────────┴────────────┴──────────────────────────────────────┘
```

---

## 2. Quantitative Baseline & Key Data Realities

| Metric / Dimension | Training Set | Testing Set | Notes & Constraints |
| :--- | :--- | :--- | :--- |
| **Source 1 (Reference)** | 2,206,821 records | 1,732,544 records | Deduplicated ground-truth reference |
| **Source 2 (Noisy)** | 5,034,616 records | 4,887,273 records | ~3.3% missing addresses |
| **Source 3 (Noisy)** | 5,285,603 records | 5,082,316 records | ~3.3% missing addresses |
| **Ground Truth Links** | 7,638,365 pairs | *Unlabeled* | Average 3.46 matches / S1 |
| **Singletons (0 match)** | **123,247 (5.58%)** | Unknown | Must predict empty string `""` |
| **Country: US** | 59.98% | 38.27% | Open-set dynamic partitioning |
| **Country: India** | 40.02% | 46.75% | Multi-script transliterations (Tamil, Hindi) |
| **Country: France** | **0.00% (Unseen)** | **14.98% (259.5k)** | Must NEVER hardcode country filter |
| **Cartesian Pair Space** | ~22.7 Trillion pairs | ~17.2 Trillion pairs | Blocking is mandatory |

---

## 3. Directory Layout & Document Map

```
amazon-ml-challenge/
├── dataset/                                           # Active dataset root
│   ├── train/                                         # train_source1, 2, 3, ground_truth TSVs
│   └── test/                                          # test_source1, 2, 3 TSVs
├── docs/
│   ├── PROJECT_STATE.md                                # Central project state tracker
│   └── phase_reports/
│       ├── phase_01_dataset_forensics.md              # Phase 1 forensics findings & noise statistics
│       └── ... (future phase documentation)
├── src/business_entity_resolution/
│   ├── __init__.py
│   ├── io.py                                          # Strict TSV reading/writing with auto-path resolution
│   ├── forensics.py                                   # Dataset profiling engine
│   ├── normalization.py                               # Multi-script, legal suffix, address normalizer
│   ├── blocking.py                                    # Multi-pass blocking & candidate generator
│   ├── features.py                                    # Vectorized pairwise string/token similarities
│   ├── training.py                                    # LightGBM classifier & hard negative trainer
│   ├── decision.py                                    # Macro F0.5 threshold & singleton handler
│   ├── evaluation.py                                  # Exact competition Macro F0.5 metric harness
│   └── utils.py                                       # Helper utilities
├── results/
│   └── dataset_forensics_report.md                    # Generated Phase 1 markdown output
├── output/                                            # Generated submissions & candidates
│   ├── matching_results.tsv                           # Final matched entity pairs (Leaderboard)
│   └── candidate_pairs.tsv                            # Blocking candidate set (Audit & Final Ranking)
├── utils/
│   └── validate_submission.py                         # Official submission format validator
├── student_resource/                                  # Original resource package backup
├── requirements.txt                                   # Pinned dependencies
├── PROJECT_CONTEXT.md                                 # Competition law & project requirements
└── README.md
```

---

## 4. Key Rules & Technical Constraints

1. **Precision-First Mindset**: $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$. Precision is weighted $2\times$ over recall. A false merge penalizes the score drastically.
2. **Strict Offline Compliance**: Zero external lookups, geocoding APIs, Google Maps, or external databases. Model size $\le 8\text{B}$ parameters, MIT/Apache-2.0 license.
3. **Format Integrity**: Every $S_1$ entity must appear once. Candidate pairs must be a superset of final matches. Output validated via `validate_submission.py`.
