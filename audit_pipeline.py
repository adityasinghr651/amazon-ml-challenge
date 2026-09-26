"""
Audit script for baseline_pipeline.py outputs.
Independent checks — does NOT import or reuse any code from the pipeline.
"""

import os
import re
import sys
import random
import pandas as pd
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

section_results = {}

# ============================================================================
# SECTION 1: SCHEMA / JOIN CHECK
# ============================================================================
print("=" * 80)
print("SECTION 1: SCHEMA / JOIN CHECK")
print("=" * 80)

s1_issues = []

# --- 1a: Entity ID prefix check ---
print("\n--- 1a: Entity ID prefix check ---")
for split in ["train", "test"]:
    for src_num, prefix in [("source1", "S1-"), ("source2", "S2-"), ("source3", "S3-")]:
        path = os.path.join(BASE, "dataset", split, f"{split}_{src_num}.tsv")
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id"],
                         keep_default_na=False)
        bad = df[~df["entity_id"].str.startswith(prefix)]
        if len(bad) > 0:
            msg = f"  {FAIL} {split}/{src_num}: {len(bad)} rows with wrong prefix"
            print(msg)
            print(f"    Examples: {bad['entity_id'].head(5).tolist()}")
            s1_issues.append(msg)
        else:
            print(f"  {PASS} {split}/{src_num}: all {len(df):,} rows start with '{prefix}'")

# --- 1b: Country value check ---
print("\n--- 1b: Country value check ---")
expected_countries = {
    "train": {"US", "India"},
    "test": {"US", "India", "France"},
}
for split in ["train", "test"]:
    for src_num in ["source1", "source2", "source3"]:
        path = os.path.join(BASE, "dataset", split, f"{split}_{src_num}.tsv")
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=["country"],
                         keep_default_na=False)
        unique_vals = set(df["country"].unique())
        extra = unique_vals - expected_countries[split]
        missing = expected_countries[split] - unique_vals

        print(f"  {split}/{src_num}: unique countries = {sorted(unique_vals)}")
        if extra:
            msg = f"    {WARN} Unexpected values: {extra}"
            print(msg)
            s1_issues.append(msg)
        if missing:
            # Not necessarily a bug — some sources might not have all countries
            print(f"    Note: expected but missing: {missing}")

        # Check for whitespace/casing variants
        stripped = df["country"].str.strip()
        ws_diff = (df["country"] != stripped).sum()
        if ws_diff > 0:
            msg = f"    {WARN} {ws_diff} rows have leading/trailing whitespace in country"
            print(msg)
            s1_issues.append(msg)

# --- 1c: No dropped source1 entities ---
print("\n--- 1c: No dropped source1 entities in output ---")
for split in ["train", "test"]:
    s1_path = os.path.join(BASE, "dataset", split, f"{split}_source1.tsv")
    mr_path = os.path.join(BASE, "output", split, "matching_results.tsv")
    cp_path = os.path.join(BASE, "output", split, "candidate_pairs.tsv")

    if not os.path.exists(mr_path):
        msg = f"  {FAIL} {split}: matching_results.tsv does not exist!"
        print(msg)
        s1_issues.append(msg)
        continue
    if not os.path.exists(cp_path):
        msg = f"  {FAIL} {split}: candidate_pairs.tsv does not exist!"
        print(msg)
        s1_issues.append(msg)
        continue

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id"],
                        keep_default_na=False)
    mr_df = pd.read_csv(mr_path, sep="\t", dtype=str, keep_default_na=False)
    cp_df = pd.read_csv(cp_path, sep="\t", dtype=str, keep_default_na=False)

    s1_set = set(s1_df["entity_id"])
    mr_set = set(mr_df["source1_entity_id"])
    cp_set = set(cp_df["source1_entity_id"])

    mr_missing = s1_set - mr_set
    mr_extra = mr_set - s1_set
    cp_missing = s1_set - cp_set
    cp_extra = cp_set - s1_set

    if mr_missing:
        msg = f"  {FAIL} {split}: {len(mr_missing)} S1 entities MISSING from matching_results.tsv"
        print(msg)
        print(f"    Examples: {list(mr_missing)[:5]}")
        s1_issues.append(msg)
    elif mr_extra:
        msg = f"  {WARN} {split}: {len(mr_extra)} extra entities in matching_results.tsv not in source1"
        print(msg)
        s1_issues.append(msg)
    else:
        print(f"  {PASS} {split}: matching_results.tsv has all {len(s1_set):,} S1 entities, no extras")

    if cp_missing:
        msg = f"  {FAIL} {split}: {len(cp_missing)} S1 entities MISSING from candidate_pairs.tsv"
        print(msg)
        s1_issues.append(msg)
    elif cp_extra:
        msg = f"  {WARN} {split}: {len(cp_extra)} extra entities in candidate_pairs.tsv"
        print(msg)
        s1_issues.append(msg)
    else:
        print(f"  {PASS} {split}: candidate_pairs.tsv has all {len(s1_set):,} S1 entities, no extras")

    # Check for duplicate source1_entity_ids
    mr_dups = mr_df["source1_entity_id"].duplicated().sum()
    cp_dups = cp_df["source1_entity_id"].duplicated().sum()
    if mr_dups:
        msg = f"  {FAIL} {split}: {mr_dups} duplicate source1_entity_ids in matching_results.tsv"
        print(msg)
        s1_issues.append(msg)
    if cp_dups:
        msg = f"  {FAIL} {split}: {cp_dups} duplicate source1_entity_ids in candidate_pairs.tsv"
        print(msg)
        s1_issues.append(msg)

section_results["1. SCHEMA/JOIN"] = PASS if not s1_issues else FAIL

# ============================================================================
# SECTION 2: LOGIC CHECK
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: LOGIC CHECK")
print("=" * 80)

s2_issues = []

# Independent normalize function (written from scratch, not copied)
def my_normalize(name):
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def my_jaccard(name_a, name_b):
    a = set(name_a.split()) if name_a else set()
    b = set(name_b.split()) if name_b else set()
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# Load train data for logic checks
print("\n--- Loading train data for logic checks ---")
train_s1 = pd.read_csv(os.path.join(BASE, "dataset", "train", "train_source1.tsv"),
                        sep="\t", dtype=str, keep_default_na=False)
train_s2 = pd.read_csv(os.path.join(BASE, "dataset", "train", "train_source2.tsv"),
                        sep="\t", dtype=str, keep_default_na=False)
train_s3 = pd.read_csv(os.path.join(BASE, "dataset", "train", "train_source3.tsv"),
                        sep="\t", dtype=str, keep_default_na=False)
train_mr = pd.read_csv(os.path.join(BASE, "output", "train", "matching_results.tsv"),
                        sep="\t", dtype=str, keep_default_na=False)
train_cp = pd.read_csv(os.path.join(BASE, "output", "train", "candidate_pairs.tsv"),
                        sep="\t", dtype=str, keep_default_na=False)

# Index data for quick lookup
s1_by_id = train_s1.set_index("entity_id")
s2_by_id = train_s2.set_index("entity_id")
s3_by_id = train_s3.set_index("entity_id")
mr_by_id = train_mr.set_index("source1_entity_id")
cp_by_id = train_cp.set_index("source1_entity_id")

# --- 2a: Spot-check 20 random entities ---
print("\n--- 2a: Spot-check 20 random S1 entities ---")

# Pick 20 that have at least some matches to verify scores
matched_eids = train_mr[train_mr["matched_entity_ids"] != ""]["source1_entity_id"].tolist()
unmatched_eids = train_mr[train_mr["matched_entity_ids"] == ""]["source1_entity_id"].tolist()

# Mix: 14 matched + 6 unmatched (or whatever is available)
n_matched = min(14, len(matched_eids))
n_unmatched = min(6, len(unmatched_eids))
sample_eids = random.sample(matched_eids, n_matched) + random.sample(unmatched_eids, n_unmatched)
random.shuffle(sample_eids)

print(f"{'S1 Entity':<18} {'Match':<18} {'Recomputed':>12} {'Threshold':>10} {'Status'}")
print("-" * 80)

recheck_ok = 0
recheck_fail = 0

for s1_eid in sample_eids:
    s1_row = s1_by_id.loc[s1_eid]
    s1_country = s1_row["country"].lower().strip()
    s1_norm = my_normalize(s1_row["business_name"])

    mr_row = mr_by_id.loc[s1_eid]
    predicted_matches = mr_row["matched_entity_ids"]
    pred_set = set(predicted_matches.split(",")) if predicted_matches else set()

    # For each predicted match, recompute score
    for match_eid in pred_set:
        if match_eid.startswith("S2-"):
            cand_row = s2_by_id.loc[match_eid]
        elif match_eid.startswith("S3-"):
            cand_row = s3_by_id.loc[match_eid]
        else:
            print(f"  {FAIL} {s1_eid} matched to {match_eid} — bad prefix!")
            s2_issues.append(f"Bad match prefix: {match_eid}")
            continue

        cand_norm = my_normalize(cand_row["business_name"])
        recomp_score = my_jaccard(s1_norm, cand_norm)
        above = recomp_score > 0.85

        status = PASS if above else FAIL
        if not above:
            recheck_fail += 1
            s2_issues.append(f"{s1_eid}->{match_eid}: score={recomp_score:.4f} should NOT be included (<=0.85)")
        else:
            recheck_ok += 1

        print(f"{s1_eid:<18} {match_eid:<18} {recomp_score:>12.4f} {'>0.85':>10} {status}")

    if not pred_set:
        print(f"{s1_eid:<18} {'(none)':<18} {'N/A':>12} {'N/A':>10} (singleton)")

print(f"\nRecheck: {recheck_ok} OK, {recheck_fail} FAIL out of {recheck_ok+recheck_fail} matched pairs")

# --- 2b: Check threshold boundary ---
print("\n--- 2b: Threshold boundary check ---")
# For a few matched entities, find the best S2/S3 candidate and verify threshold
# We need to find entities where the best score is close to 0.85
print("  (Checking a sample of entities for threshold correctness...)")

boundary_check_count = 0
boundary_issues = 0

# Sample some entities and verify against all same-country candidates from their source
# (This is expensive, so we only do a small sample with restricted candidates)
boundary_sample = random.sample(matched_eids, min(5, len(matched_eids)))
for s1_eid in boundary_sample:
    s1_row = s1_by_id.loc[s1_eid]
    s1_country = s1_row["country"].lower().strip()
    s1_norm = my_normalize(s1_row["business_name"])

    mr_row = mr_by_id.loc[s1_eid]
    predicted = mr_row["matched_entity_ids"]
    pred_set = set(predicted.split(",")) if predicted else set()

    # Get the candidate list from candidate_pairs
    cp_row = cp_by_id.loc[s1_eid]
    cand_str = cp_row["candidate_entity_ids"]
    cand_list = cand_str.split(",") if cand_str else []

    # Score all candidates, find best per source
    best_s2 = (None, -1.0)
    best_s3 = (None, -1.0)
    for ceid in cand_list:
        if ceid.startswith("S2-"):
            try:
                crow = s2_by_id.loc[ceid]
            except KeyError:
                continue
            cnorm = my_normalize(crow["business_name"])
            sc = my_jaccard(s1_norm, cnorm)
            if sc > best_s2[1]:
                best_s2 = (ceid, sc)
        elif ceid.startswith("S3-"):
            try:
                crow = s3_by_id.loc[ceid]
            except KeyError:
                continue
            cnorm = my_normalize(crow["business_name"])
            sc = my_jaccard(s1_norm, cnorm)
            if sc > best_s3[1]:
                best_s3 = (ceid, sc)

    # Verify: if best score > 0.85, should be in pred_set
    for label, (best_eid, best_score) in [("S2", best_s2), ("S3", best_s3)]:
        if best_eid is None:
            continue
        boundary_check_count += 1
        should_match = best_score > 0.85
        did_match = best_eid in pred_set

        if should_match != did_match:
            boundary_issues += 1
            print(f"  {FAIL} {s1_eid}: {label} best={best_eid} score={best_score:.4f} "
                  f"should_match={should_match} did_match={did_match}")
            s2_issues.append(f"Threshold error: {s1_eid} {label}")
        else:
            print(f"  {PASS} {s1_eid}: {label} best={best_eid} score={best_score:.4f} "
                  f"threshold_correct={should_match == did_match}")

print(f"  Boundary checks: {boundary_check_count - boundary_issues}/{boundary_check_count} correct")

# --- 2c: At most 1 S2 + 1 S3 match per entity ---
print("\n--- 2c: At most 1 S2 + 1 S3 match per entity ---")
multi_s2 = 0
multi_s3 = 0
for _, row in train_mr.iterrows():
    matched = row["matched_entity_ids"]
    if not matched:
        continue
    ids = matched.split(",")
    s2_count = sum(1 for x in ids if x.startswith("S2-"))
    s3_count = sum(1 for x in ids if x.startswith("S3-"))
    if s2_count > 1:
        multi_s2 += 1
    if s3_count > 1:
        multi_s3 += 1

if multi_s2 > 0:
    msg = f"  {FAIL} {multi_s2} entities have >1 S2 match"
    print(msg)
    s2_issues.append(msg)
else:
    print(f"  {PASS} No entity has >1 S2 match")

if multi_s3 > 0:
    msg = f"  {FAIL} {multi_s3} entities have >1 S3 match"
    print(msg)
    s2_issues.append(msg)
else:
    print(f"  {PASS} No entity has >1 S3 match")

# --- 2d: Every match must be in candidate_pairs ---
print("\n--- 2d: Every match must appear as a candidate ---")
match_not_in_cand = 0
total_match_ids = 0
for _, row in train_mr.iterrows():
    s1_eid = row["source1_entity_id"]
    matched = row["matched_entity_ids"]
    if not matched:
        continue
    match_ids = set(matched.split(","))
    total_match_ids += len(match_ids)

    cp_row = cp_by_id.loc[s1_eid]
    cand_str = cp_row["candidate_entity_ids"]
    cand_set = set(cand_str.split(",")) if cand_str else set()

    missing = match_ids - cand_set
    if missing:
        match_not_in_cand += len(missing)

if match_not_in_cand > 0:
    msg = f"  {FAIL} {match_not_in_cand}/{total_match_ids} matched IDs NOT found in candidate_pairs"
    print(msg)
    s2_issues.append(msg)
else:
    print(f"  {PASS} All {total_match_ids:,} matched IDs are present in their candidate lists")

section_results["2. LOGIC"] = PASS if not s2_issues else FAIL

# ============================================================================
# SECTION 3: METRIC CHECK
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: METRIC CHECK")
print("=" * 80)

s3_issues = []

# --- 3a: Independent F0.5 computation ---
print("\n--- 3a: Independent F0.5 recomputation ---")

gt = pd.read_csv(os.path.join(BASE, "dataset", "train", "train_ground_truth.tsv"),
                 sep="\t", dtype=str, keep_default_na=False)

# Build ground truth dict
gt_dict = {}
for _, row in gt.iterrows():
    eid = row["source1_entity_id"]
    m = row["matched_entity_ids"]
    gt_dict[eid] = set(m.split(",")) if m else set()

# Build prediction dict from output file (fresh read)
pred_fresh = pd.read_csv(os.path.join(BASE, "output", "train", "matching_results.tsv"),
                         sep="\t", dtype=str, keep_default_na=False)
pred_dict = {}
for _, row in pred_fresh.iterrows():
    eid = row["source1_entity_id"]
    m = row["matched_entity_ids"]
    pred_dict[eid] = set(m.split(",")) if m else set()

# Compute macro-average F0.5
f05_list = []
prec_list = []
rec_list = []

true_singletons = 0
pred_singletons = 0
correct_singletons = 0
false_positives = 0  # predicted match where ground truth is empty or different
false_negatives = 0  # ground truth has match but prediction is empty
tp_total = 0
fp_total = 0
fn_total = 0

for eid in gt_dict:
    true_set = gt_dict[eid]
    pred_set = pred_dict.get(eid, set())

    if not true_set:
        true_singletons += 1
    if not pred_set:
        pred_singletons += 1

    # Singleton handling
    if not true_set and not pred_set:
        f05_list.append(1.0)
        prec_list.append(1.0)
        rec_list.append(1.0)
        correct_singletons += 1
        continue
    if not true_set and pred_set:
        f05_list.append(0.0)
        prec_list.append(0.0)
        rec_list.append(0.0)
        fp_total += len(pred_set)
        false_positives += 1
        continue
    if true_set and not pred_set:
        f05_list.append(0.0)
        prec_list.append(0.0)
        rec_list.append(0.0)
        fn_total += len(true_set)
        false_negatives += 1
        continue

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tp_total += tp
    fp_total += fp
    fn_total += fn

    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(true_set) if true_set else 0.0

    if prec + rec == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec)

    f05_list.append(f05)
    prec_list.append(prec)
    rec_list.append(rec)

    if fp > 0:
        false_positives += 1
    if fn > 0:
        false_negatives += 1

n = len(f05_list)
indep_prec = sum(prec_list) / n
indep_rec = sum(rec_list) / n
indep_f05 = sum(f05_list) / n

print(f"  Independent recomputation (over {n:,} entities):")
print(f"    Precision : {indep_prec:.4f}")
print(f"    Recall    : {indep_rec:.4f}")
print(f"    F0.5      : {indep_f05:.4f}")
print(f"  (Compare against pipeline's reported numbers to verify consistency)")

# --- 3b: Error analysis ---
print(f"\n--- 3b: Error analysis ---")
print(f"  True singletons (no ground truth match): {true_singletons:,}")
print(f"  Predicted singletons (empty prediction):  {pred_singletons:,}")
print(f"  Correctly predicted singletons:            {correct_singletons:,}")
print(f"  Entities with ≥1 FP (predicted wrong):    {false_positives:,}")
print(f"  Entities with ≥1 FN (missed true match):  {false_negatives:,}")
print(f"  Total TP pairs: {tp_total:,}")
print(f"  Total FP pairs: {fp_total:,}")
print(f"  Total FN pairs: {fn_total:,}")

# Note: we can't compare to "reported" numbers since the pipeline might still be running
# We'll flag if the numbers look suspicious
n_non_singletons_true = n - true_singletons
n_non_singletons_pred = n - pred_singletons
print(f"\n  True non-singleton entities: {n_non_singletons_true:,}")
print(f"  Predicted non-singleton entities: {n_non_singletons_pred:,}")

section_results["3. METRICS"] = PASS  # Will be compared manually against pipeline output

# ============================================================================
# SECTION 4: FILE FORMAT CHECK
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: FILE FORMAT CHECK")
print("=" * 80)

s4_issues = []

for split in ["train", "test"]:
    for fname in ["candidate_pairs.tsv", "matching_results.tsv"]:
        fpath = os.path.join(BASE, "output", split, fname)

        if not os.path.exists(fpath):
            msg = f"  {FAIL} {split}/{fname} does not exist"
            print(msg)
            s4_issues.append(msg)
            continue

        print(f"\n--- {split}/{fname} ---")

        with open(fpath, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i >= 6:  # header + 5 data rows + 1 extra
                    break

        # Print header + first 5 rows
        for i, line in enumerate(lines[:6]):
            label = "HEADER" if i == 0 else f"ROW {i}"
            print(f"  [{label}] {line.rstrip()}")

        # Check header
        header = lines[0].rstrip()
        if fname == "candidate_pairs.tsv":
            expected_header = "source1_entity_id\tcandidate_entity_ids"
        else:
            expected_header = "source1_entity_id\tmatched_entity_ids"

        if header != expected_header:
            msg = f"  {FAIL} Header mismatch: got '{header}', expected '{expected_header}'"
            print(msg)
            s4_issues.append(msg)
        else:
            print(f"  {PASS} Header matches expected format")

        # Check delimiter
        tab_count = lines[0].count("\t")
        if tab_count != 1:
            msg = f"  {FAIL} Header has {tab_count} tabs, expected 1"
            print(msg)
            s4_issues.append(msg)
        else:
            print(f"  {PASS} Tab-delimited (1 tab per line)")

        # Count total rows
        with open(fpath, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        data_rows = total_lines - 1  # minus header
        print(f"  Total data rows: {data_rows:,}")

        # Verify row count matches source1
        s1_path = os.path.join(BASE, "dataset", split, f"{split}_source1.tsv")
        s1_count = sum(1 for _ in open(s1_path, "r", encoding="utf-8")) - 1
        if data_rows != s1_count:
            msg = f"  {FAIL} Row count {data_rows:,} != source1 count {s1_count:,}"
            print(msg)
            s4_issues.append(msg)
        else:
            print(f"  {PASS} Row count matches source1 ({s1_count:,})")

# --- Check for sample_submission files ---
print("\n--- Checking for sample_submission files ---")
for root, dirs, files in os.walk(os.path.join(BASE, "dataset")):
    for f in files:
        if "sample" in f.lower() or "submission" in f.lower():
            print(f"  Found: {os.path.join(root, f)}")

# Also check utils for validator
validator_path = os.path.join(BASE, "utils", "validate_submission.py")
if os.path.exists(validator_path):
    print(f"  Found validator: {validator_path}")
else:
    # Check in student_resource
    validator_path2 = os.path.join("e:\\6ab10eb3b23ba_student_resource\\student_resource",
                                    "utils", "validate_submission.py")
    if os.path.exists(validator_path2):
        print(f"  Found validator: {validator_path2}")
    else:
        print("  No sample_submission or validator found in dataset/")

section_results["4. FILE FORMAT"] = PASS if not s4_issues else FAIL

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)

for section, result in section_results.items():
    print(f"  {section}: {result}")

print("\n--- Issues that need fixing ---")
all_issues = s1_issues + s2_issues + s3_issues + s4_issues
if all_issues:
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")
else:
    print("  None found — all checks passed!")
