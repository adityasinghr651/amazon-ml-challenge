import os
import sys
import time
import gc
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def find_dataset_dir():
    candidates = [
        "dataset",
        os.path.join("student_resource", "dataset"),
        os.path.join("..", "dataset"),
        os.path.join("..", "student_resource", "dataset")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train", "train_source1.tsv")):
            return c
    raise FileNotFoundError("Could not find dataset directory.")

def profile_single_file(filepath, name):
    if not os.path.exists(filepath):
        return f"### {name}\n*File `{filepath}` not found.*\n\n", None
    
    log(f"Profiling {os.path.basename(filepath)}...")
    t0 = time.time()
    
    # Read TSV with pyarrow or standard pandas
    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)
    load_time = time.time() - t0
    log(f"  Loaded {len(df):,} rows in {load_time:.2f}s")
    
    lines = [f"### {name}"]
    lines.append(f"- **File**: `{os.path.basename(filepath)}`")
    lines.append(f"- **Total Rows**: {len(df):,}")
    lines.append(f"- **Columns**: `{', '.join(df.columns)}`")
    
    lines.append("\n**Field Completeness & Empty Values:**")
    for col in df.columns:
        empty_cnt = ((df[col] == "") | (df[col].isna())).sum()
        lines.append(f"- `{col}`: {empty_cnt:,} empty ({empty_cnt/len(df)*100:.2f}%)")
        
    lines.append("\n**Character Length Statistics:**")
    for col in ['business_name', 'business_address']:
        if col in df.columns:
            lengths = df[col].str.len().values
            non_empty = lengths[lengths > 0]
            if len(non_empty) > 0:
                lines.append(
                    f"- `{col}`: Min={non_empty.min()}, Mean={non_empty.mean():.1f}, "
                    f"Median={np.median(non_empty):.0f}, P90={np.percentile(non_empty, 90):.0f}, "
                    f"P95={np.percentile(non_empty, 95):.0f}, P99={np.percentile(non_empty, 99):.0f}, Max={non_empty.max()}"
                )
                
    lines.append("\n**Token Word Count Statistics:**")
    for col in ['business_name', 'business_address']:
        if col in df.columns:
            token_lens = df[col].str.split().str.len().values
            non_empty_t = token_lens[token_lens > 0]
            if len(non_empty_t) > 0:
                lines.append(
                    f"- `{col}` tokens: Min={non_empty_t.min()}, Mean={non_empty_t.mean():.1f}, "
                    f"Median={np.median(non_empty_t):.0f}, P95={np.percentile(non_empty_t, 95):.0f}, Max={non_empty_t.max()}"
                )

    if 'country' in df.columns:
        lines.append("\n**Country Breakdown:**")
        counts = df['country'].value_counts()
        for c_val, cnt in counts.items():
            disp_c = c_val if c_val != "" else "<EMPTY>"
            lines.append(f"- `{disp_c}`: {cnt:,} ({cnt/len(df)*100:.2f}%)")
            
    lines.append("\n")
    return "\n".join(lines), df

def analyze_gt_file(gt_path, total_s2_count, total_s3_count):
    log("Analyzing Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    
    total_s1 = len(gt_df)
    def parse_ids(val):
        if not val or val.strip() == "":
            return []
        return [x.strip() for x in val.split(",") if x.strip()]
        
    gt_df['match_list'] = gt_df['matched_entity_ids'].apply(parse_ids)
    match_lens = gt_df['match_list'].apply(len).values
    
    singletons = (match_lens == 0).sum()
    total_positive_pairs = match_lens.sum()
    
    lines = ["### Ground Truth & Match Distribution Forensics"]
    lines.append(f"- **Total Reference S1 Entities in GT**: {total_s1:,}")
    lines.append(f"- **Singletons (0 matches in S2/S3)**: {singletons:,} ({singletons/total_s1*100:.2f}%)")
    lines.append(f"- **Entities with $\\ge 1$ match**: {total_s1 - singletons:,} ({(total_s1 - singletons)/total_s1*100:.2f}%)")
    lines.append(f"- **Total Positive Match Pairs**: {total_positive_pairs:,}")
    lines.append(f"- **Matches per S1**: Mean={match_lens.mean():.2f}, Median={np.median(match_lens):.0f}, Max={match_lens.max()}")
    
    lines.append("\n**Matches per S1 Histogram:**")
    counts_hist = Counter(match_lens)
    for k in sorted(counts_hist.keys()):
        if k <= 5 or k == max(counts_hist.keys()):
            lines.append(f"- `{k}` matches: {counts_hist[k]:,} entities ({counts_hist[k]/total_s1*100:.2f}%)")
        elif k == 6:
            more_cnt = sum(cnt for m_k, cnt in counts_hist.items() if m_k >= 6)
            lines.append(f"- `6+` matches: {more_cnt:,} entities ({more_cnt/total_s1*100:.2f}%)")
            
    all_matches = [m for sublist in gt_df['match_list'] for m in sublist]
    s2_matches = [m for m in all_matches if m.startswith("S2-")]
    s3_matches = [m for m in all_matches if m.startswith("S3-")]
    
    unique_s2 = len(set(s2_matches))
    unique_s3 = len(set(s3_matches))
    
    lines.append("\n**Source 2 & Source 3 Match Coverage:**")
    lines.append(f"- Total positive links to **Source 2**: {len(s2_matches):,} (Unique S2 entities: {unique_s2:,})")
    lines.append(f"- Total positive links to **Source 3**: {len(s3_matches):,} (Unique S3 entities: {unique_s3:,})")
    
    if total_s2_count > 0:
        lines.append(f"- **S2 Recall Ceiling**: {unique_s2:,} / {total_s2_count:,} ({unique_s2/total_s2_count*100:.2f}%) matched; {total_s2_count - unique_s2:,} ({100 - (unique_s2/total_s2_count*100):.2f}%) are non-matching distractors.")
    if total_s3_count > 0:
        lines.append(f"- **S3 Recall Ceiling**: {unique_s3:,} / {total_s3_count:,} ({unique_s3/total_s3_count*100:.2f}%) matched; {total_s3_count - unique_s3:,} ({100 - (unique_s3/total_s3_count*100):.2f}%) are non-matching distractors.")
        
    lines.append("\n")
    return "\n".join(lines), gt_df

def extract_empirical_noise_samples(gt_df, s1_df, train_dir, num_samples=15):
    log("Sampling noise patterns from ground truth...")
    # Find sample matching pairs
    gt_pos = gt_df[gt_df['matched_entity_ids'] != ""].head(100)
    
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    
    # Collect sample S2 and S3 IDs to fetch
    needed_s2 = set()
    needed_s3 = set()
    pairs = []
    
    for _, row in gt_pos.iterrows():
        s1_id = row['source1_entity_id']
        m_ids = [x.strip() for x in row['matched_entity_ids'].split(",") if x.strip()]
        for m_id in m_ids:
            if m_id.startswith("S2-"):
                needed_s2.add(m_id)
            else:
                needed_s3.add(m_id)
            pairs.append((s1_id, m_id))
            if len(pairs) >= num_samples * 3:
                break
        if len(pairs) >= num_samples * 3:
            break
            
    # Read only needed rows from S2 and S3 for speed
    log("Reading sample rows from S2 and S3...")
    s2_dict = {}
    s3_dict = {}
    
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=50000, keep_default_na=False):
        sub = chunk[chunk['entity_id'].isin(needed_s2)]
        if len(sub) > 0:
            for _, r in sub.iterrows():
                s2_dict[r['entity_id']] = r.to_dict()
        if len(s2_dict) >= len(needed_s2):
            break
            
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=50000, keep_default_na=False):
        sub = chunk[chunk['entity_id'].isin(needed_s3)]
        if len(sub) > 0:
            for _, r in sub.iterrows():
                s3_dict[r['entity_id']] = r.to_dict()
        if len(s3_dict) >= len(needed_s3):
            break
            
    lines = ["### Empirical Noise & Match Variation Samples\n"]
    lines.append("| S1 Entity | S1 Business Name | S1 Address | Matched ID | Matched Name | Matched Address | Identified Noise / Variation Type |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    count = 0
    for s1_id, m_id in pairs:
        s1_info = s1_dict.get(s1_id, {})
        m_info = s2_dict.get(m_id, {}) if m_id.startswith("S2-") else s3_dict.get(m_id, {})
        if not s1_info or not m_info:
            continue
            
        s1_n = s1_info.get("business_name", "")
        s1_a = s1_info.get("business_address", "")
        m_n = m_info.get("business_name", "")
        m_a = m_info.get("business_address", "")
        
        noise = []
        if s1_n.lower().strip() == m_n.lower().strip():
            noise.append("Exact Name")
        elif any(w in s1_n.lower() or w in m_n.lower() for w in ["pvt", "ltd", "inc", "corp", "llc", "co"]):
            noise.append("Legal Suffix/Abbreviation")
        else:
            noise.append("Name Modification/Typo")
            
        if s1_a.lower().strip() == m_a.lower().strip():
            noise.append("Exact Address")
        elif s1_a == "" or m_a == "":
            noise.append("Missing Address")
        else:
            noise.append("Address Formatting / Landmark")
            
        c_s1_n = s1_n.replace("|", " ").replace("\t", " ")
        c_s1_a = s1_a.replace("|", " ").replace("\t", " ")[:35]
        c_m_n = m_n.replace("|", " ").replace("\t", " ")
        c_m_a = m_a.replace("|", " ").replace("\t", " ")[:35]
        
        lines.append(f"| `{s1_id}` | {c_s1_n} | {c_s1_a} | `{m_id}` | {c_m_n} | {c_m_a} | {', '.join(noise)} |")
        count += 1
        if count >= num_samples:
            break
            
    lines.append("\n")
    return "\n".join(lines)

def run():
    log("=" * 60)
    log("STARTING PHASE 1: SEQUENTIAL DATASET FORENSICS")
    log("=" * 60)
    
    data_dir = find_dataset_dir()
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    report = ["# Phase 1: Comprehensive Dataset Forensics Report\n"]
    report.append("> **Amazon ML Challenge 2026: Business Entity Resolution**\n")
    report.append(f"*Generated on real competition data from: `{data_dir}`*\n\n")
    
    # 1. Training Source 1
    report.append("## 1. Training Dataset Profiling\n")
    s1_train_text, s1_train_df = profile_single_file(os.path.join(train_dir, "train_source1.tsv"), "Source 1 (Train - Deduplicated Reference)")
    report.append(s1_train_text)
    
    # 2. Training Source 2
    s2_train_text, s2_train_df = profile_single_file(os.path.join(train_dir, "train_source2.tsv"), "Source 2 (Train - Noisy Input)")
    report.append(s2_train_text)
    len_s2 = len(s2_train_df) if s2_train_df is not None else 0
    del s2_train_df; gc.collect()
    
    # 3. Training Source 3
    s3_train_text, s3_train_df = profile_single_file(os.path.join(train_dir, "train_source3.tsv"), "Source 3 (Train - Noisy Input)")
    report.append(s3_train_text)
    len_s3 = len(s3_train_df) if s3_train_df is not None else 0
    del s3_train_df; gc.collect()
    
    # 4. Ground Truth
    report.append("## 2. Ground Truth & Match Structure Analysis\n")
    gt_text, gt_df = analyze_gt_file(os.path.join(train_dir, "train_ground_truth.tsv"), len_s2, len_s3)
    report.append(gt_text)
    
    # 5. Noise samples
    report.append("## 3. Empirical Noise & Pattern Analysis\n")
    noise_text = extract_empirical_noise_samples(gt_df, s1_train_df, train_dir, num_samples=15)
    report.append(noise_text)
    del s1_train_df, gt_df; gc.collect()
    
    # 6. Test Data
    report.append("## 4. Test Set Characterization\n")
    s1_test_text, s1_test_df = profile_single_file(os.path.join(test_dir, "test_source1.tsv"), "Source 1 (Test - Reference Entities to Match)")
    report.append(s1_test_text)
    del s1_test_df; gc.collect()
    
    s2_test_text, s2_test_df = profile_single_file(os.path.join(test_dir, "test_source2.tsv"), "Source 2 (Test - Noisy Input)")
    report.append(s2_test_text)
    del s2_test_df; gc.collect()
    
    s3_test_text, s3_test_df = profile_single_file(os.path.join(test_dir, "test_source3.tsv"), "Source 3 (Test - Noisy Input)")
    report.append(s3_test_text)
    del s3_test_df; gc.collect()
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/dataset_forensics_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    log(f"Phase 1 Dataset Forensics Complete! Written to `{out_path}`.")

if __name__ == "__main__":
    run()
