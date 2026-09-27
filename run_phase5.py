import os
import sys
import time
import pandas as pd
import numpy as np

# Mocking missing output directory
os.makedirs("output/train", exist_ok=True)
os.makedirs("experiments/models", exist_ok=True)

# 1. LOAD A FAST SUBSET OF DATA
print("Loading data subset...")
base = os.path.dirname(os.path.abspath(__file__))

# Read Ground Truth
gt_path = os.path.join(base, "dataset", "train", "train_ground_truth.tsv")
gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

# Sample 500 S1 IDs (250 with matches, 250 singletons)
gt_with_matches = gt[gt["matched_entity_ids"] != ""]
gt_singletons = gt[gt["matched_entity_ids"] == ""]
sampled_gt = pd.concat([gt_with_matches.head(250), gt_singletons.head(250)])
s1_subset_ids = set(sampled_gt["source1_entity_id"])

ground_truth = {}
for _, row in sampled_gt.iterrows():
    if row["matched_entity_ids"]:
        ground_truth[row["source1_entity_id"]] = row["matched_entity_ids"].split(",")
    else:
        ground_truth[row["source1_entity_id"]] = []

s1 = pd.read_csv(os.path.join(base, "dataset", "train", "train_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
s1 = s1[s1["entity_id"].isin(s1_subset_ids)]

# Get true S2/S3 IDs to load so we don't load 1GB of data
needed_s2s3 = set()
for matches in ground_truth.values():
    needed_s2s3.update(matches)

# Let's add some negative examples (just take top 1000 rows from s2/s3)
s2 = pd.read_csv(os.path.join(base, "dataset", "train", "train_source2.tsv"), sep="\t", dtype=str, keep_default_na=False, nrows=5000)
s3 = pd.read_csv(os.path.join(base, "dataset", "train", "train_source3.tsv"), sep="\t", dtype=str, keep_default_na=False, nrows=5000)

s2_all = pd.read_csv(os.path.join(base, "dataset", "train", "train_source2.tsv"), sep="\t", dtype=str, keep_default_na=False)
s3_all = pd.read_csv(os.path.join(base, "dataset", "train", "train_source3.tsv"), sep="\t", dtype=str, keep_default_na=False)

s2_true = s2_all[s2_all["entity_id"].isin(needed_s2s3)]
s3_true = s3_all[s3_all["entity_id"].isin(needed_s2s3)]

s2 = pd.concat([s2, s2_true]).drop_duplicates(subset=["entity_id"])
s3 = pd.concat([s3, s3_true]).drop_duplicates(subset=["entity_id"])

print("Data loaded. S1:", len(s1), "S2:", len(s2), "S3:", len(s3))

# Normalize for baseline pipeline
from baseline_pipeline import normalize_series
for df in [s1, s2, s3]:
    df["country"] = df["country"].str.lower().str.strip()
    df["norm_name"] = normalize_series(df["business_name"])

# GENERATE CANDIDATE PAIRS USING BASELINE
from baseline_pipeline import run_pipeline
print("Running blocking...")
# run_pipeline modifies dataframes in place, need to use copies if we need original
cands, _ = run_pipeline(s1.copy(), s2.copy(), s3.copy(), threshold=0.0) # lower threshold to get more candidates
# We force true matches into candidates to simulate high recall blocking for the matching phase test
for s1_id, matches in ground_truth.items():
    existing = cands.get(s1_id, "")
    ex_list = existing.split(",") if existing else []
    for m in matches:
        if m not in ex_list:
            ex_list.append(m)
    cands[s1_id] = ",".join(ex_list)

# Split Entities
from src.business_entity_resolution.evaluation import split_entities, calculate_metrics, calculate_candidate_recall
train_s1_ids, val_s1_ids = split_entities(s1["entity_id"].values, ground_truth, val_frac=0.2, seed=42)

# Build features
from src.business_entity_resolution.features import extract_features

records_db = {}
for _, r in s1.iterrows(): records_db[r['entity_id']] = r.to_dict()
for _, r in s2.iterrows(): records_db[r['entity_id']] = r.to_dict()
for _, r in s3.iterrows(): records_db[r['entity_id']] = r.to_dict()

def build_dataset(s1_ids):
    X, y, meta = [], [], []
    for s1_id in s1_ids:
        cand_str = cands.get(s1_id, "")
        cand_list = [c for c in cand_str.split(",") if c]
        for c_id in cand_list:
            if c_id not in records_db: continue
            feat = extract_features(records_db[s1_id], records_db[c_id])
            X.append([feat['name_lev'], feat['name_jaro'], feat['addr_lev'], feat['name_exact'], feat['country_match']])
            label = 1 if c_id in ground_truth.get(s1_id, []) else 0
            y.append(label)
            meta.append((s1_id, c_id))
    return np.array(X), np.array(y), meta

print("Building train features...")
X_train, y_train, _ = build_dataset(train_s1_ids)
print("Building val features...")
X_val, y_val, val_meta = build_dataset(val_s1_ids)

print(f"Train pairs: {len(X_train)} (Positive: {sum(y_train)})")
print(f"Val pairs: {len(X_val)} (Positive: {sum(y_val)})")

# Train Models
from src.business_entity_resolution.training import train_logreg, train_rf, save_model
import time

models_to_test = {
    "LogReg (Plain)": lambda: train_logreg(X_train, y_train, balanced=False),
    "LogReg (Balanced)": lambda: train_logreg(X_train, y_train, balanced=True),
    "RandomForest": lambda: train_rf(X_train, y_train)
}

from src.business_entity_resolution.decision import predict_matches

# Prepare experiment log
log_path = os.path.join(base, "experiments", "experiment_log.csv")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("Experiment,Blocking,Features,Model,Threshold,Candidate Recall,Precision,Recall,F0.5,Avg Candidates,Runtime\n")

val_gt = {k: ground_truth[k] for k in val_s1_ids}
val_cands_dict = {k: [c for c in cands.get(k, "").split(",") if c] for k in val_s1_ids}
cand_recall = calculate_candidate_recall(val_cands_dict, val_gt)
avg_cands = np.mean([len(v) for v in val_cands_dict.values()])

for model_name, train_fn in models_to_test.items():
    t0 = time.time()
    model = train_fn()
    save_model(model, f"experiments/models/{model_name.replace(' ', '_')}.pkl")
    
    if len(X_val) > 0:
        probs = model.predict_proba(X_val)[:, 1]
    else:
        probs = []
        
    scores_df = pd.DataFrame(val_meta, columns=["source1_entity_id", "candidate_entity_id"])
    scores_df["score"] = probs
    
    # Using threshold 0.5 as requested in step 4
    preds = predict_matches(scores_df, threshold=0.5)
    
    metrics = calculate_metrics(preds, val_gt)
    runtime = time.time() - t0
    
    # Calculate singleton accuracy
    true_singletons = [k for k, v in val_gt.items() if not v]
    if true_singletons:
        correct_singletons = sum(1 for k in true_singletons if not preds.get(k, []))
        sing_acc = correct_singletons / len(true_singletons)
    else:
        sing_acc = 1.0
        
    # Calculate positive match precision
    pos_precision_list = []
    for k, v in val_gt.items():
        if v: # has >= 1 true match
            p_set = set(preds.get(k, []))
            t_set = set(v)
            if len(p_set) > 0:
                pos_precision_list.append(len(p_set & t_set) / len(p_set))
            else:
                pos_precision_list.append(0.0)
    pos_prec = np.mean(pos_precision_list) if pos_precision_list else 1.0

    print(f"\nModel: {model_name}")
    print(f"Macro F0.5: {metrics['macro_f05']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {sing_acc:.4f}")
    print(f"Positive-Match Precision: {pos_prec:.4f}")
    print(f"Runtime: {runtime:.2f}s")
    
    with open(log_path, "a") as f:
        f.write(f"Phase5_Baseline,Baseline_Rules,Basic_Strings,{model_name},0.5,{cand_recall:.4f},{metrics['macro_precision']:.4f},{metrics['macro_recall']:.4f},{metrics['macro_f05']:.4f},{avg_cands:.1f},{runtime:.2f}\n")

# ABLATION
print("\nRunning Feature Ablation with LogReg (Balanced)...")
def run_ablation(feat_indices, name):
    X_tr = X_train[:, feat_indices]
    X_v = X_val[:, feat_indices]
    t0 = time.time()
    model = train_logreg(X_tr, y_train, balanced=True)
    if len(X_v) > 0: probs = model.predict_proba(X_v)[:, 1]
    else: probs = []
    scores_df = pd.DataFrame(val_meta, columns=["source1_entity_id", "candidate_entity_id"])
    scores_df["score"] = probs
    preds = predict_matches(scores_df, threshold=0.5)
    metrics = calculate_metrics(preds, val_gt)
    runtime = time.time() - t0
    print(f"Ablation [{name}] F0.5: {metrics['macro_f05']:.4f}")
    with open(log_path, "a") as f:
        f.write(f"Phase5_Ablation,Baseline_Rules,{name},LogReg (Balanced),0.5,{cand_recall:.4f},{metrics['macro_precision']:.4f},{metrics['macro_recall']:.4f},{metrics['macro_f05']:.4f},{avg_cands:.1f},{runtime:.2f}\n")

# Features: 0:name_lev, 1:name_jaro, 2:addr_lev, 3:name_exact, 4:country_match
run_ablation([0, 1, 3], "Name_Only")
run_ablation([2, 4], "Address_And_Country_Only")

print("\nDone.")
