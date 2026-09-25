"For new team members: run `pip install -r requirements.txt`, read this file fully 
before touching any code, follow the phase order strictly, never skip dataset 
forensics before designing blocking/model logic, and always run 
utils/validate_submission.py before treating any output as final. This file is 
project law — do not ask the project owner to re-explain anything covered here."

=====================================================================
1. COMPETITION OVERVIEW
=====================================================================
- Task: business records come from 3 independent sources with NO shared identifiers.
  Source 1 = deduplicated reference entities. Source 2 & 3 = noisy business records.
  For every Source 1 entity, identify ALL corresponding records from Source 2 and/or 
  Source 3 representing the same real-world business. A Source 1 entity can have 
  zero, one, or multiple matches. This is a large-scale Entity Resolution / Record 
  Linkage problem.

- Data files:
  Training: dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, 
            train_ground_truth.tsv
  Testing:  dataset/test/test_source1.tsv, test_source2.tsv, test_source3.tsv
  ALWAYS read with pd.read_csv(path, sep="\t") — never assume comma-separated.
  Source columns: entity_id, business_name, business_address, country
  Ground truth columns: source1_entity_id, matched_entity_ids (comma-separated S2/S3 IDs)

- Country handling: training countries = US, India. Test additionally contains France 
  (unseen in training). Country MUST be treated as an open-set string feature — 
  NEVER hard-code the pipeline to {US, India}. Every test Source 1 entity must appear 
  in the final output regardless of country.

- Noise types to expect:
  Business names: abbreviations, legal suffix differences (Corporation/Corp/Corp.), 
    (Private Limited/Pvt Ltd/Pvt. Ltd.), punctuation differences, typos, 
    transliteration, word-order changes, DBA/trade names.
  Addresses: Rd vs Road, St vs Street, missing PIN, missing state, missing address 
    components, landmarks, municipal numbering differences, component reordering, 
    transliteration, inconsistent formatting.

=====================================================================
2. EVALUATION METRIC
=====================================================================
- Leaderboard metric: macro-averaged F0.5 = (1.25 × Precision × Recall) / 
  (0.25 × Precision + Recall). This is PRECISION-HEAVY — false positive merges are 
  especially harmful. Singletons are extremely important: if true answer is [] and we 
  predict [], full credit; if we incorrectly predict a match for a singleton, that 
  entity scores ZERO. DO NOT blindly maximize recall — the final decision layer must 
  optimize actual macro F0.5, not accuracy or raw recall.

=====================================================================
3. BLOCKING / CANDIDATE GENERATION — MANDATORY, HEAVILY GRADED
=====================================================================
- This is NOT a simple fuzzy matching problem. Cannot compare every S1 × every S2/S3 
  (dataset too large). Must build: S1 entity → blocking → small candidate set from 
  S2/S3 → ML matcher → final matches.
- Candidate generation determines the recall ceiling — if the true match isn't 
  generated as a candidate, the model can never recover it. Candidate recall is one 
  of the most important metrics in the whole project.
- MUST produce output/candidate_pairs.tsv representing the EXACT candidate set the 
  matching model saw immediately before matching (not an early broad blocking dump). 
  Every final match must exist in this candidate set. Amazon inspects: candidate 
  recall, candidate set size, reduction ratio, blocking implementation, scalability, 
  and the code that produced the candidates. Smaller candidate sets per S1 entity are 
  explicitly preferred, beyond just leaderboard quality.
- Blocking strategies to design and test independently, then union + prune:
  Block A — exact normalized name
  Block B — exact strong address components (postal/number/token-derived keys, only 
            if discovered from the dataset itself)
  Block C — name token signatures (rare/high-information tokens)
  Block D — character n-gram retrieval (TF-IDF or equivalent sparse representation)
  Block E — phonetic representation (only if experiments show usefulness)
  Block F — name + address composite keys
  Block G — approximate retrieval (only where computationally justified)
- For every blocking experiment measure: candidate recall, avg/median/P95/max 
  candidates per S1, reduction ratio, runtime, memory. Optimization target = HIGH 
  candidate recall + LOW candidate volume — never sacrifice large true-match recall 
  just to shrink candidate sets.

=====================================================================
4. REQUIRED OUTPUT FILES — EXACT FORMAT
=====================================================================
output/matching_results.tsv — columns exactly: source1_entity_id  matched_entity_ids
  - one row per test S1 entity (every one must appear, exactly once)
  - matched IDs only from S2/S3, no duplicates, empty for singleton/no-match
  - IDs must exist in test data
output/candidate_pairs.tsv — columns exactly: source1_entity_id  candidate_entity_ids
  - every S1 gets exactly one row
  - candidate IDs must belong to S2/S3, exist in test data, no duplicates, and must 
    contain every final predicted match for that S1

Final submission package structure:
TEAM_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/src/, README.md, requirements.txt
└── Documentation_template.md
Pipeline must be fully reproducible end-to-end (raw data → preprocessing → blocking → 
features → model → decision layer → both TSVs) with NO hidden manual steps.

=====================================================================
5. HARD RULES — NEVER VIOLATE
=====================================================================
FAIR PLAY: use ONLY the provided challenge data and training labels. STRICTLY NO: 
Google search, Google Maps, business directories, government registries, business 
registration databases, geocoding APIs, external entity-resolution APIs, external 
business datasets, external identity databases, internet-based enrichment, or any 
external data augmentation. Pipeline must be fully defensible as an offline ML 
solution.

MODEL LICENSE: final model must be MIT or Apache-2.0 licensed, ≤8B parameters. Before 
recommending any pretrained model, explicitly verify: model, license, parameter 
count, purpose, whether it's actually necessary. Do not assume an LLM is 
automatically useful.

NEVER DO (full list):
1. Use external business lookup.  2. Use Google Maps.  3. Use geocoding.  
4. Scrape websites for business identities.  5. Download external business datasets.  
6. Hard-code France.  7. Assume country values are only US/India.  
8. Use full Cartesian joins.  9. Submit invalid TSV files.  
10. Predict only one match per S1.  11. Auto-choose a candidate even at low confidence.  
12. Optimize accuracy instead of macro F0.5.  13. Ignore singleton entities.  
14. Throw away candidate_pairs.tsv.  15. Hide manual post-processing.  
16. Use an incompatible model license.  17. Leak validation labels into feature 
generation.  18. Overfit to the public leaderboard.  19. Claim an improvement without 
measuring it.  20. Build a complicated model before establishing a strong baseline.

=====================================================================
6. ENGINEERING PHILOSOPHY / PIPELINE ORDER
=====================================================================
DATA UNDERSTANDING → BASELINE → BLOCKING → FEATURE ENGINEERING → MATCHING MODEL → 
THRESHOLD OPTIMIZATION → ERROR ANALYSIS → BLOCKING IMPROVEMENT → MODEL IMPROVEMENT → 
FINAL PIPELINE. Every major improvement must be supported by validation results. 
Never say "this should improve the score" — always say "Experiment X improved 
validation macro F0.5 from A to B while candidate recall changed from C to D."

=====================================================================
7. DATASET FORENSICS (Phase 1 — run first on real data)
=====================================================================
- Dataset size: S1/S2/S3 row counts for train and test separately
- Missing values: count + percentage for business_name, business_address, country
- Length statistics (min, mean, median, P90, P95, P99, max) for names and addresses
- Country distribution (train and test separately), special attention to unseen 
  test countries
- Ground truth analysis: % S1 singletons, avg/median/max matches per S1, S2 match 
  coverage, S3 match coverage, number of positive pairs, number of negative/distractor 
  records
- Noise analysis: inspect ACTUAL examples of exact matches, abbreviation matches, 
  typo matches, reordered names, address variation, missing address components, 
  transliteration, landmark addresses, difficult false positives. DO NOT infer 
  patterns before inspecting real data.

=====================================================================
8. NORMALIZATION ENGINE
=====================================================================
Build multiple representations, don't destroy information: raw / normalized / 
tokenized / character n-grams, for BOTH name and address fields.
Names: unicode normalization, lowercase, punctuation normalization, whitespace 
normalization, & ↔ and, legal suffix normalization, abbreviation normalization, 
repeated-character normalization, token sorting where appropriate. Do NOT blindly 
remove meaningful words (e.g. "ABC Hotel" vs "ABC Hotel Resort" vs "ABC Hotel & 
Resort" must NOT automatically become identical).
Addresses: unicode normalization, lowercase, punctuation cleanup, whitespace 
normalization, common abbreviation handling, tokenization, numeric token extraction, 
PIN/postal code extraction where structurally present, house/building number 
extraction, state/city-like token handling only where safely inferred from text. 
No external geographic knowledge.

=====================================================================
9. PAIRWISE FEATURE ENGINEERING
=====================================================================
Name features: exact equality, normalized equality, token Jaccard, token overlap, 
character Jaccard, Levenshtein similarity, edit distance, Jaro/Jaro-Winkler, TF-IDF 
cosine similarity, character n-gram cosine, token count difference, length ratio, 
common token count, rare-token overlap. Run ablations — don't use every feature 
blindly.
Address features: normalized exact similarity, token Jaccard, token overlap, 
character similarity, TF-IDF cosine, numeric token overlap, postal-code match, 
house-number match, address length ratio, common token count. Treat missing values 
explicitly — missing address must NOT automatically mean dissimilar.
Cross-field features: name_similarity × address_similarity, name_exact AND 
address_partial, country_match, name_high_address_high, name_high_address_missing — 
only where there's a logical basis.

=====================================================================
10. MODELING
=====================================================================
Start interpretable/efficient: Model 1 = Logistic Regression, Model 2 = Random 
Forest, Model 3 = Gradient Boosting/LightGBM/XGBoost. Only license-compliant models. 
Architecture: candidate pair → feature vector → gradient boosted classifier → 
P(match) → decision layer. Do not use a huge neural model just because 8B params is 
allowed.

HARD NEGATIVE MINING: random negatives are too easy. Generate difficult negatives — 
same country, similar name, similar address, common tokens, but actually different 
businesses. Train classifier to distinguish hard positive vs hard negative. Validate 
whether it actually improves F0.5 before keeping it.

=====================================================================
11. VALIDATION DESIGN
=====================================================================
No test ground truth exists — build validation from training data. Do NOT randomly 
split individual candidate pairs without considering entity structure — split at the 
S1-entity level so all associated ground truth matches stay together (leakage-safe). 
Validation pipeline must mimic test inference exactly: validation S1 + candidate 
S2/S3 → blocking → features → model → threshold → predictions → macro F0.5. Never 
evaluate using info unavailable at test time.

=====================================================================
12. METRICS TO REPORT PER EXPERIMENT
=====================================================================
Macro F0.5, Precision, Recall, Singleton accuracy, Positive-match precision, 
Candidate recall, Avg candidates/S1, P95 candidates/S1, Candidate reduction ratio, 
Runtime, Memory usage. Maintain an experiment table: Experiment, Blocking, Features, 
Model, Threshold, Candidate Recall, Precision, Recall, F0.5, Avg Candidates, Runtime. 
Keep full version history — never overwrite past experiments.

=====================================================================
13. THRESHOLD OPTIMIZATION & SINGLETON/MULTI-MATCH HANDLING
=====================================================================
Never use threshold=0.5 blindly — search thresholds (e.g. 0.50 to 0.95) optimizing 
macro F0.5, not accuracy. Investigate confidence regimes (very high → match, medium 
→ require supporting evidence, low → reject) only if validation shows improvement.
Singleton detection: explicitly model that an S1 may have NO match. A model that 
always picks the highest-scoring candidate is dangerous — if all candidate scores are 
weak, return []. Threshold should consider absolute confidence of the best candidate, 
not just relative ranking. Analyze singleton-specific false positives.
Multiple matches: an S1 can match multiple S2/S3 records — this is NOT a simple 
one-to-one classification problem. Decision layer must allow multiple matches when 
evidence supports it, while not indiscriminately attaching an S2/S3 record to 
unrelated S1 entities. Only implement global constraints if confirmed useful by 
actual data.

=====================================================================
14. ERROR ANALYSIS (after every major model)
=====================================================================
False positives — categorize: same generic business name, same city, similar 
address, common token collision, over-aggressive normalization, blocking ambiguity, 
model overconfidence.
False negatives — categorize: typo, transliteration, missing address, DBA name, weak 
blocking, rare spelling, very short name.
Build an error taxonomy and use it to drive the next experiment.

=====================================================================
15. ADVANCED ARCHITECTURE (only if baseline plateaus)
=====================================================================
Consider: sparse retrieval + character-level retrieval + semantic representation + 
gradient boosted pair classifier + calibrated decision layer + constraint/decision 
layer. A small licensed embedding model may be considered ONLY if: license compliant, 
runs within resources, improves validation F0.5, doesn't rely on external data, and 
is reproducible. Do not add unnecessary complexity.

=====================================================================
16. COMPUTATIONAL STRATEGY
=====================================================================
Design for scale — never materialize huge pairwise Cartesian products. Always 
estimate N_S1×N_S2 and N_S1×N_S3 before any such operation. Use: chunked reading, 
categorical/string optimization, Arrow/Parquet where appropriate, sparse matrices, 
inverted indexes, dictionary encoding, vectorized operations, batch inference, 
memory-aware joins.

=====================================================================
17. PROJECT STRUCTURE
=====================================================================
amazon-ml-challenge/
├── data/train/, data/test/
├── notebooks/01_dataset_forensics.ipynb, 02_blocking_experiments.ipynb, 
│   03_feature_analysis.ipynb, 04_model_experiments.ipynb, 05_error_analysis.ipynb
├── src/business_entity_resolution/
│   io.py, preprocessing.py, normalization.py, blocking.py, features.py, 
│   training.py, inference.py, evaluation.py, decision.py, utils.py
├── experiments/experiment_log.csv, results/
├── output/matching_results.tsv, candidate_pairs.tsv
├── README.md, requirements.txt, Documentation_template.md

Code quality: modular, typed where useful, deterministic/reproducible seeds, clear 
function names, no giant notebook-only implementation, logging, configuration-driven 
parameters, no hardcoded machine-specific paths, no hidden preprocessing, no manual 
post-processing, comments for non-obvious logic, unit tests for critical 
transformations, output validation before submission.

=====================================================================
18. SUBMISSION VALIDATION & WORKFLOW
=====================================================================
Before every leaderboard submission, run: 
python3 utils/validate_submission.py --matching output/matching_results.tsv 
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
Must get PASS before spending a submission. Limit: 5 submissions/day for 3 days — 
submissions must be deliberate experiments, not random attempts.
Validator details: stdlib-only Python 3.8+, reads only output files + test source 
files (never ground truth, never computes score). Checks TAB-separated format with 
exact headers, every test S1 present exactly once, no duplicate rows, no duplicate 
IDs within a list, no self-matches or wrong-prefix IDs, and warns (never fails) if 
matched IDs aren't a subset of candidate_pairs.tsv. candidate_pairs.tsv is optional 
for the validator run but required in the final submission zip. ID-existence check 
against test_source2/3.tsv is OFF by default (memory-heavy on ~1.7M entities) — 
enable with --check-ids; a nonexistent ID only lowers score, doesn't reject 
submission.

Dev workflow: Antigravity (write/refactor/debug code, primary source of truth) → 
GitHub (version control, push code) → Colab (OPTIONAL — only for heavy compute if 
local resources are insufficient) → download final TSVs → submission portal. Never 
put confidential challenge data into a public GitHub repo.

Leaderboard strategy: public leaderboard is only a subset of test set — do NOT 
overfit to it. Use submissions to validate whether internal-validation improvements 
generalize. Track: Submission ID, Git commit, Experiment ID, Validation F0.5, Public 
score, Changes, Hypothesis, Result. Never change something solely because public 
score moved, unless internal validation supports it. Private leaderboard determines 
final ranking.

=====================================================================
19. TECH STACK & SCOPE
=====================================================================
Python 3.10+, pandas, numpy, rapidfuzz, scikit-learn (TF-IDF, LogReg, RandomForest), 
LightGBM (final matcher, MIT licensed), plain YAML/JSON config + CSV experiment log, 
pytest. NO frontend — this is a backend/batch ML pipeline only (raw data in → TSV 
files out). Colab is optional; Antigravity + local execution is the primary flow.

=====================================================================
20. PIPELINE PHASES (8 phases — build in this order)
=====================================================================
Phase 0 — Setup: repo structure, config system, io.py, skeleton tested on synthetic 
data
Phase 1 — Dataset Forensics (see section 7)
Phase 2 — Normalization Engine (see section 8)
Phase 3 — Blocking System + candidate_pairs.tsv + recall measurement (see section 3)
Phase 4 — Feature Engineering (see section 9)
Phase 5 — Baseline Model: LogReg/RandomForest, leakage-safe validation split, first 
F0.5 number
Phase 6 — Model Improvement: LightGBM, hard negative mining, threshold optimization, 
singleton handling, multi-match decision layer
Phase 7 — Error Analysis + Final Packaging: FP/FN taxonomy, final TSVs, validator 
PASS, zip, methodology document (executive summary, problem analysis, methodology, 
candidate generation, matching model, error analysis, results, reproducibility, 
conclusion)

=====================================================================
21. NON-NEGOTIABLE GOAL
=====================================================================
Our ONLY objective is to WIN this competition. Optimizing macro F0.5 is the central, 
continuous goal at every single phase — never settle for "good enough" or a quick 
implementation over a correct, validated, high-performing one. Every design decision 
must be justified by measured validation results, never assumptions. Precision 
matters more than recall (F0.5 is precision-heavy) — never let recall-chasing hurt 
singleton accuracy. Final success requires ALL of: high macro F0.5 + high precision + 
strong candidate recall + small candidate sets + good singleton handling + scalable 
computation + zero external lookup + compliant model license + reproducible code + 
valid TSV outputs + complete final ZIP + defensible methodology.
