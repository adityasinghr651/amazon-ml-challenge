import os
import sys
import time
import pandas as pd
import psutil

sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.io import load_train_data
from src.business_entity_resolution.normalization import process_dataframe
from src.business_entity_resolution.blocking import generate_candidates, measure_blocking_performance

def run_experiment():
    print("Loading datasets...")
    s1, s2, s3, gt = load_train_data()
    
    print("Taking a 1% sample to speed up experiment and prevent Out-Of-Memory crash...", flush=True)
    s1 = s1.sample(frac=0.01, random_state=42)
    s2 = s2.sample(frac=0.01, random_state=42)
    s3 = s3.sample(frac=0.01, random_state=42)
    
    total_s2s3 = len(s2) + len(s3)
    
    print("Normalizing data (this may take a minute)...")
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    s3_norm = process_dataframe(s3)
    
    experiments = [
        {"name": "Block A (Exact Name)", "config": {"active_blocks": ["A"], "enable_pruning": False}},
        {"name": "Block B (Address Anchors)", "config": {"active_blocks": ["B"], "enable_pruning": False}},
        {"name": "Block A+B (Name + Address)", "config": {"active_blocks": ["A", "B"], "enable_pruning": False}},
        {"name": "Block A+B+C (Rare Tokens)", "config": {"active_blocks": ["A", "B", "C"], "rare_token_percentile": 25, "enable_pruning": True, "max_candidates_before_pruning": 50, "prune_top_k": 20}},
        {"name": "Block A+B+C+D (Full Pipeline)", "config": {"active_blocks": ["A", "B", "C", "D"], "rare_token_percentile": 25, "ann_neighbors": 10, "ann_max_distance": 0.5, "enable_pruning": True, "max_candidates_before_pruning": 50, "prune_top_k": 20}}
    ]
    
    log_file = "experiments/experiment_log.csv"
    os.makedirs("experiments", exist_ok=True)
    write_header = not os.path.exists(log_file)
    
    with open(log_file, "a") as f:
        if write_header:
            f.write("Experiment,Blocking,Candidate Recall,Avg Candidates,Reduction Ratio,Runtime(s),Peak Memory(MB)\n")
            
        for exp in experiments:
            print(f"\nRunning {exp['name']}...")
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
            start_time = time.time()
            
            candidates_df = generate_candidates(s1_norm, s2_norm, s3_norm, exp["config"])
            
            runtime = time.time() - start_time
            mem_after = process.memory_info().rss / (1024 * 1024)
            peak_memory = mem_after - mem_before
            
            metrics = measure_blocking_performance(candidates_df, gt, total_s2s3)
            
            print(f"Results for {exp['name']}:")
            print(f"  Recall:          {metrics['candidate_recall']:.4f}")
            print(f"  Avg Candidates:  {metrics['avg_candidates']:.2f}")
            print(f"  Median Cands:    {metrics['median_candidates']}")
            print(f"  P95 Cands:       {metrics['p95_candidates']}")
            print(f"  Max Cands:       {metrics['max_candidates']}")
            print(f"  Reduction Ratio: {metrics['reduction_ratio']:.6f}")
            print(f"  Runtime:         {runtime:.2f}s")
            
            # Log to CSV
            f.write(f"{exp['name']},{'+'.join(exp['config']['active_blocks'])},{metrics['candidate_recall']:.4f},{metrics['avg_candidates']:.2f},{metrics['reduction_ratio']:.6f},{runtime:.2f},{peak_memory:.2f}\n")

if __name__ == "__main__":
    run_experiment()
