"""
Phase 0: io.py
Handles loading and saving of TSV files, and dataset reading logic.
"""
import pandas as pd
import os

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads a dataset from a TSV file.
    Always uses sep='\t' and reads all columns as strings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")
    return pd.read_csv(path, sep='\t', dtype=str)

def save_results(results_df: pd.DataFrame, path: str):
    """
    Saves the final matching results.
    Format exactly: source1_entity_id \t matched_entity_ids
    """
    results_df.to_csv(path, sep='\t', index=False, columns=['source1_entity_id', 'matched_entity_ids'])

def save_candidates(candidates_df: pd.DataFrame, path: str):
    """
    Saves the candidate pairs.
    Format exactly: source1_entity_id \t candidate_entity_ids
    """
    candidates_df.to_csv(path, sep='\t', index=False, columns=['source1_entity_id', 'candidate_entity_ids'])
