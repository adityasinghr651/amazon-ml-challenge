"""
Phase 6: decision.py
Threshold optimization, singleton handling, and multi-match decision logic.
"""

import pandas as pd

def predict_matches(scores_df, threshold=0.5):
    """
    Convert model probabilities to final matches based on threshold and logic.
    """
    predictions = {}
    
    for s1_id in scores_df['source1_entity_id'].unique():
        predictions[s1_id] = []
        
    matches = scores_df[scores_df['score'] > threshold]
    
    for _, row in matches.iterrows():
        predictions[row['source1_entity_id']].append(row['candidate_entity_id'])
        
    return predictions
