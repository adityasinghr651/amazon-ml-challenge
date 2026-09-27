"""
Phase 5 & 6: evaluation.py
Calculates validation metrics including Macro F0.5.
"""
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split

def split_entities(s1_ids, ground_truth, val_frac=0.2, seed=42):
    """Splits S1 entities into train and val."""
    s1_ids = list(s1_ids)
    train_ids, val_ids = train_test_split(s1_ids, test_size=val_frac, random_state=seed)
    
    def pct_singleton(ids):
        singletons = sum(1 for x in ids if not ground_truth.get(x, []))
        return (singletons / len(ids) * 100) if len(ids) > 0 else 0.0

    print(f"Train split: {len(train_ids)} S1 entities ({pct_singleton(train_ids):.1f}% singletons)")
    print(f"Val split: {len(val_ids)} S1 entities ({pct_singleton(val_ids):.1f}% singletons)")
    
    return train_ids, val_ids

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


def decomposed_error_analysis(
    predictions: dict,
    candidates: dict,
    ground_truth: dict,
    records_db: dict = None,
) -> dict:
    """
    Decompose final errors into two categories:
      (a) Blocking recall misses — true match was NEVER in the candidate set.
      (b) Classifier/threshold mistakes — true match WAS in candidates but model got it wrong.

    Also inspects false positives and false negatives with record details.

    Parameters
    ----------
    predictions : dict S1 ID -> list of predicted match IDs.
    candidates : dict S1 ID -> list of candidate IDs (from blocking).
    ground_truth : dict S1 ID -> list of true match IDs.
    records_db : optional dict entity_id -> record dict (for FP/FN inspection).

    Returns
    -------
    Dict with decomposed error counts and example lists.
    """
    blocking_miss_entities = 0       # S1s with at least one true match not in candidates
    blocking_miss_pairs = 0          # total true-match pairs missed by blocking
    classifier_miss_entities = 0     # S1s with at least one true match in candidates but not predicted
    classifier_miss_pairs = 0        # total true-match pairs missed by classifier
    classifier_fp_entities = 0       # S1s with at least one false positive
    classifier_fp_pairs = 0          # total false positive pairs

    blocking_miss_examples = []      # (s1_id, missed_match_id)
    classifier_fn_examples = []      # (s1_id, missed_match_id)
    classifier_fp_examples = []      # (s1_id, false_match_id)

    max_examples = 20

    for s1_id, true_matches in ground_truth.items():
        true_set = set(true_matches)
        pred_set = set(predictions.get(s1_id, []))
        cand_set = set(candidates.get(s1_id, []))

        if not true_set:
            # Singleton — check for false positives only
            if pred_set:
                classifier_fp_entities += 1
                classifier_fp_pairs += len(pred_set)
                for fp_id in list(pred_set)[:3]:
                    if len(classifier_fp_examples) < max_examples:
                        classifier_fp_examples.append((s1_id, fp_id, "singleton_fp"))
            continue

        # Non-singleton
        for match_id in true_set:
            if match_id not in cand_set:
                # (a) Blocking recall miss
                blocking_miss_pairs += 1
                if len(blocking_miss_examples) < max_examples:
                    blocking_miss_examples.append((s1_id, match_id))
            elif match_id not in pred_set:
                # (b) Classifier missed it despite being in candidates
                classifier_miss_pairs += 1
                if len(classifier_fn_examples) < max_examples:
                    classifier_fn_examples.append((s1_id, match_id))

        # Entity-level counts
        blocked_misses = true_set - cand_set
        classifier_misses = (true_set & cand_set) - pred_set
        false_positives = pred_set - true_set

        if blocked_misses:
            blocking_miss_entities += 1
        if classifier_misses:
            classifier_miss_entities += 1
        if false_positives:
            classifier_fp_entities += 1
            classifier_fp_pairs += len(false_positives)
            for fp_id in list(false_positives)[:2]:
                if len(classifier_fp_examples) < max_examples:
                    classifier_fp_examples.append((s1_id, fp_id, "non_singleton_fp"))

    total_non_singleton = sum(1 for v in ground_truth.values() if v)
    total_pairs = sum(len(v) for v in ground_truth.values())

    result = {
        "summary": {
            "total_s1_entities": len(ground_truth),
            "total_non_singleton": total_non_singleton,
            "total_true_match_pairs": total_pairs,
        },
        "blocking_errors": {
            "entities_affected": blocking_miss_entities,
            "pairs_missed": blocking_miss_pairs,
            "pct_pairs_missed": blocking_miss_pairs / total_pairs * 100 if total_pairs > 0 else 0,
            "examples": blocking_miss_examples,
        },
        "classifier_errors": {
            "fn_entities_affected": classifier_miss_entities,
            "fn_pairs_missed": classifier_miss_pairs,
            "pct_fn_pairs": classifier_miss_pairs / total_pairs * 100 if total_pairs > 0 else 0,
            "fp_entities_affected": classifier_fp_entities,
            "fp_pairs": classifier_fp_pairs,
            "fn_examples": classifier_fn_examples,
            "fp_examples": classifier_fp_examples,
        },
    }

    return result


def print_error_analysis(analysis: dict):
    """Pretty-print the decomposed error analysis."""
    s = analysis["summary"]
    b = analysis["blocking_errors"]
    c = analysis["classifier_errors"]

    print("=" * 60)
    print("DECOMPOSED ERROR ANALYSIS")
    print("=" * 60)
    print(f"  Total S1 entities: {s['total_s1_entities']:,}")
    print(f"  Non-singleton:     {s['total_non_singleton']:,}")
    print(f"  True match pairs:  {s['total_true_match_pairs']:,}")

    print(f"\n  (a) BLOCKING RECALL MISSES:")
    print(f"      Entities affected: {b['entities_affected']:,}")
    print(f"      Pairs missed:      {b['pairs_missed']:,} ({b['pct_pairs_missed']:.2f}%)")
    if b["examples"]:
        print(f"      Examples:")
        for s1_id, m_id in b["examples"][:5]:
            print(f"        S1={s1_id} → missed {m_id}")

    print(f"\n  (b) CLASSIFIER/THRESHOLD MISTAKES:")
    print(f"      FN entities:  {c['fn_entities_affected']:,}")
    print(f"      FN pairs:     {c['fn_pairs_missed']:,} ({c['pct_fn_pairs']:.2f}%)")
    print(f"      FP entities:  {c['fp_entities_affected']:,}")
    print(f"      FP pairs:     {c['fp_pairs']:,}")
    if c["fn_examples"]:
        print(f"      FN Examples:")
        for s1_id, m_id in c["fn_examples"][:5]:
            print(f"        S1={s1_id} → missed {m_id} (was in candidates)")
    if c["fp_examples"]:
        print(f"      FP Examples:")
        for item in c["fp_examples"][:5]:
            print(f"        S1={item[0]} → predicted {item[1]} ({item[2]})")

    # Recommendation
    if b["pairs_missed"] > c["fn_pairs_missed"]:
        print(f"\n  → RECOMMENDATION: Focus on BLOCKING improvement.")
        print(f"     Blocking misses ({b['pairs_missed']}) > Classifier misses ({c['fn_pairs_missed']})")
    else:
        print(f"\n  → RECOMMENDATION: Focus on MODEL/THRESHOLD improvement.")
        print(f"     Classifier misses ({c['fn_pairs_missed']}) >= Blocking misses ({b['pairs_missed']})")
