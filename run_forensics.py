import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os

def load_tsv(path):
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t", dtype=str)
    return None

def analyze_dataframe(df, name):
    if df is None:
        return f"### {name}\nFile not found.\n"
    
    report = [f"### {name}"]
    report.append(f"- **Row count**: {len(df)}")
    
    # Missing values
    report.append("\n**Missing Values:**")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        report.append(f"- {col}: {missing_count} ({missing_pct:.2f}%)")
    
    # Length statistics for text columns
    report.append("\n**Length Statistics:**")
    for col in ['business_name', 'business_address']:
        if col in df.columns:
            # Dropna for length calc
            lengths = df[col].dropna().apply(len)
            if len(lengths) > 0:
                report.append(f"- **{col}** - Min: {lengths.min()}, Mean: {lengths.mean():.2f}, Median: {lengths.median()}, P90: {np.percentile(lengths, 90)}, P95: {np.percentile(lengths, 95)}, P99: {np.percentile(lengths, 99)}, Max: {lengths.max()}")
            
    # Country distribution
    if 'country' in df.columns:
        report.append("\n**Country Distribution:**")
        counts = df['country'].value_counts(dropna=False)
        for country, count in counts.items():
            report.append(f"- {country}: {count} ({(count/len(df))*100:.2f}%)")
            
    return "\n".join(report) + "\n\n"

def analyze_ground_truth(gt_df, s1_df, s2_df, s3_df):
    if gt_df is None:
        return "### Ground Truth Analysis\ntrain_ground_truth.tsv not found.\n"
    
    report = ["### Ground Truth Analysis"]
    
    # Matches per S1
    gt_df['matches_list'] = gt_df['matched_entity_ids'].apply(lambda x: [i.strip() for i in str(x).split(",")] if pd.notna(x) and str(x).strip() != "" else [])
    match_counts = gt_df['matches_list'].apply(len)
    
    singletons = (match_counts == 0).sum()
    report.append(f"- **Total S1 entities in GT**: {len(gt_df)}")
    report.append(f"- **Singletons (0 matches)**: {singletons} ({(singletons/len(gt_df))*100:.2f}%)")
    report.append(f"- **Matches per S1** - Avg: {match_counts.mean():.2f}, Median: {match_counts.median()}, Max: {match_counts.max()}")
    
    total_positive_pairs = match_counts.sum()
    report.append(f"- **Total positive match pairs**: {total_positive_pairs}")
    
    # Match coverage
    all_matched_ids = set([item for sublist in gt_df['matches_list'] for item in sublist])
    if s2_df is not None and s3_df is not None:
        s2_ids = set(s2_df['entity_id'].dropna())
        s3_ids = set(s3_df['entity_id'].dropna())
        
        s2_matches = len(all_matched_ids.intersection(s2_ids))
        s3_matches = len(all_matched_ids.intersection(s3_ids))
        
        report.append(f"- **S2 match coverage**: {s2_matches} records matched ({(s2_matches/len(s2_ids))*100:.2f}% of S2)" if len(s2_ids)>0 else "- S2 match coverage: N/A")
        report.append(f"- **S3 match coverage**: {s3_matches} records matched ({(s3_matches/len(s3_ids))*100:.2f}% of S3)" if len(s3_ids)>0 else "- S3 match coverage: N/A")
        
        distractors_s2 = len(s2_ids) - s2_matches
        distractors_s3 = len(s3_ids) - s3_matches
        report.append(f"- **Negative/Distractor records**: S2={distractors_s2}, S3={distractors_s3}")
    
    return "\n".join(report) + "\n\n"

def generate_report():
    print("Loading data...")
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    
    s1_train = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
    s2_train = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    s3_train = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    gt_train = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    s1_test = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    s2_test = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    s3_test = load_tsv(os.path.join(test_dir, "test_source3.tsv"))
    
    report_lines = ["# Phase 1: Dataset Forensics Report\n"]
    
    report_lines.append("## Training Data\n")
    report_lines.append(analyze_dataframe(s1_train, "Source 1 (Train)"))
    report_lines.append(analyze_dataframe(s2_train, "Source 2 (Train)"))
    report_lines.append(analyze_dataframe(s3_train, "Source 3 (Train)"))
    report_lines.append(analyze_ground_truth(gt_train, s1_train, s2_train, s3_train))
    
    report_lines.append("## Testing Data\n")
    report_lines.append(analyze_dataframe(s1_test, "Source 1 (Test)"))
    report_lines.append(analyze_dataframe(s2_test, "Source 2 (Test)"))
    report_lines.append(analyze_dataframe(s3_test, "Source 3 (Test)"))
    
    output_path = "results/dataset_forensics_report.md"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    
    print(f"Dataset forensics report saved to {output_path}")

if __name__ == "__main__":
    generate_report()
