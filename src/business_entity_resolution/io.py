"""
io.py - High Performance & Strict TSV I/O Engine
Handles loading, streaming, and saving of TSV dataset files.
"""
import os
import pandas as pd
from typing import Optional, List, Dict, Tuple

def get_dataset_dir() -> str:
    """Finds the dataset directory, prioritizing the root dataset/ folder."""
    candidates = [
        "dataset",
        os.path.join("student_resource", "dataset"),
        os.path.join("..", "dataset"),
        os.path.join("..", "student_resource", "dataset")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train", "train_source1.tsv")):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate dataset directory containing train/train_source1.tsv.")

def load_tsv(filepath: str, usecols: Optional[List[str]] = None, dtype=str) -> pd.DataFrame:
    """
    Loads a dataset from a TSV file with strict tab separator and no default NA mangling.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, sep="\t", dtype=dtype, usecols=usecols, keep_default_na=False)

def load_train_data(dataset_dir: Optional[str] = None, usecols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all training tables: s1, s2, s3, and ground_truth."""
    d_dir = dataset_dir or get_dataset_dir()
    train_dir = os.path.join(d_dir, "train")
    
    s1 = load_tsv(os.path.join(train_dir, "train_source1.tsv"), usecols=usecols)
    s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"), usecols=usecols)
    s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"), usecols=usecols)
    gt = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))
    return s1, s2, s3, gt

def load_test_data(dataset_dir: Optional[str] = None, usecols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all test tables: s1, s2, s3."""
    d_dir = dataset_dir or get_dataset_dir()
    test_dir = os.path.join(d_dir, "test")
    
    s1 = load_tsv(os.path.join(test_dir, "test_source1.tsv"), usecols=usecols)
    s2 = load_tsv(os.path.join(test_dir, "test_source2.tsv"), usecols=usecols)
    s3 = load_tsv(os.path.join(test_dir, "test_source3.tsv"), usecols=usecols)
    return s1, s2, s3

def save_matching_results(results_df: pd.DataFrame, output_path: str = "output/matching_results.tsv"):
    """
    Saves the final entity matches.
    Columns: source1_entity_id \t matched_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, sep="\t", index=False, columns=["source1_entity_id", "matched_entity_ids"])

def save_candidate_pairs(candidates_df: pd.DataFrame, output_path: str = "output/candidate_pairs.tsv"):
    """
    Saves the candidate pairs from blocking.
    Columns: source1_entity_id \t candidate_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    candidates_df.to_csv(output_path, sep="\t", index=False, columns=["source1_entity_id", "candidate_entity_ids"])
