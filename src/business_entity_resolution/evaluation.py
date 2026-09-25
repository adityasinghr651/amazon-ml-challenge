"""
Phase 5 & 6: evaluation.py
Calculates validation metrics including Macro F0.5.
"""
import numpy as np

def calculate_metrics(predictions: dict, ground_truth: dict) -> dict:
    """
    Calculates Macro Precision, Recall, and F0.5.
    
    predictions: dict mapping source1_entity_id -> list of matched S2/S3 ids
    ground_truth: dict mapping source1_entity_id -> list of true S2/S3 ids
    """
    precisions = []
    recalls = []
    
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        
        true_set = set(true_matches)
        pred_set = set(pred_matches)
        
        # Singleton correctly predicted as singleton
        if len(true_set) == 0 and len(pred_set) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        # Singleton incorrectly predicted to have matches
        elif len(true_set) == 0 and len(pred_set) > 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        # Failed to predict any matches for a non-singleton
        elif len(pred_set) == 0 and len(true_set) > 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        # Standard Precision/Recall calculation
        intersection = true_set.intersection(pred_set)
        
        p = len(intersection) / len(pred_set)
        r = len(intersection) / len(true_set)
        
        precisions.append(p)
        recalls.append(r)
        
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    # Macro F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    if avg_precision + avg_recall == 0:
        f0_5 = 0.0
    else:
        f0_5 = (1.25 * avg_precision * avg_recall) / (0.25 * avg_precision + avg_recall)
        
    return {
        "macro_f05": f0_5,
        "macro_precision": avg_precision,
        "macro_recall": avg_recall
    }

def calculate_candidate_recall(candidates: dict, ground_truth: dict) -> float:
    """
    Calculates Candidate Recall: the percentage of true matches recovered in the candidate sets.
    
    candidates: dict mapping source1_entity_id -> list of candidate ids
    ground_truth: dict mapping source1_entity_id -> list of true ids
    """
    total_true_matches = 0
    total_recovered = 0
    
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
            
        true_set = set(true_matches)
        cand_set = set(candidates.get(s1_id, []))
        
        total_true_matches += len(true_set)
        total_recovered += len(true_set.intersection(cand_set))
        
    if total_true_matches == 0:
        return 1.0
        
    return total_recovered / total_true_matches
