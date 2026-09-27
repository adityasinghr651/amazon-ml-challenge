"""
generate_blocking_notebooks.py - Generates the 8 mandatory interactive Colab/Jupyter notebooks
"""
import os
import json

NOTEBOOKS_DIR = "notebooks/blocking"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(filename, title, description, code_cells):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    f"**Amazon ML Challenge 2026: Business Entity Resolution**\n\n",
                    f"{description}\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    for header, code in code_cells:
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"### {header}\n"]
        })
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        })

    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {path}")

# Common setup code
SETUP_CODE = """import os
import sys
import pandas as pd
import numpy as np

# Adjust sys.path to root directory
if '..' not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
if '.' not in sys.path:
    sys.path.insert(0, os.getcwd())

from src.business_entity_resolution.blocking import (
    build_or_load_benchmark,
    evaluate_candidate_pairs,
    format_evaluation_summary,
    CandidateUnionEngine
)
"""

# 01_blocking_baseline.ipynb
create_notebook(
    "01_blocking_baseline.ipynb",
    "01 — Baseline Blocking Measurement",
    "Measures existing Baseline 1 (CountVectorizer Token Jaccard) and Baseline 2 (Legacy A+B+C+D).",
    [
        ("1. Environment & Setup", SETUP_CODE),
        ("2. Load Benchmark Dataset", """s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
target_pool = pd.concat([s2, s3], ignore_index=True)
print(f"Loaded: S1={len(s1):,}, Target={len(target_pool):,}, Ground Truth={len(gt):,}")
"""),
        ("3. Run and Evaluate Baseline", """from experiments.phase_b_baseline_benchmark import run_baseline_1_blocking
b1_cands, b1_time = run_baseline_1_blocking(s1, s2, s3)
metrics = evaluate_candidate_pairs(b1_cands, gt, len(target_pool))
print(format_evaluation_summary(metrics, title="Baseline 1: Token Jaccard"))
""")
    ]
)

# 02_address_blocking.ipynb
create_notebook(
    "02_address_blocking.ipynb",
    "02 — Address Anchor Blocking",
    "Evaluates structural address blocking keys (PIN, PIN+Primary Number, PIN+Token) and bucket distributions.",
    [
        ("1. Environment & Setup", SETUP_CODE),
        ("2. Load Benchmark Dataset", """s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
target_pool = pd.concat([s2, s3], ignore_index=True)
"""),
        ("3. Run Address Anchor Blocker", """from src.business_entity_resolution.blocking import AddressBlockConfig, AddressAnchorBlocker

cfg = AddressBlockConfig(use_pin_only=False, use_pin_primary_num=True, use_pin_name_token=True, max_bucket_size=500)
blk = AddressAnchorBlocker(name="Block_B", config=cfg)
blk.fit(target_pool)
cands = blk.block(s1)

metrics = evaluate_candidate_pairs(cands, gt, len(target_pool))
print(format_evaluation_summary(metrics, title="Address Anchor Blocker"))
print("Bucket Diagnostics:", blk.get_bucket_diagnostics())
""")
    ]
)

# 03_name_blocking.ipynb
create_notebook(
    "03_name_blocking.ipynb",
    "03 — Exact Name Blocking Tournament",
    "Compares raw clean name (A1), suffix-stripped core name (A2), and token-sorted name (A3).",
    [
        ("1. Environment & Setup", SETUP_CODE),
        ("2. Load Benchmark Dataset", """s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
target_pool = pd.concat([s2, s3], ignore_index=True)
"""),
        ("3. Run Multi-Representation Name Blocker", """from src.business_entity_resolution.blocking import NameBlockConfig, ExactNameBlocker

for name_mode, kwargs in [("Raw Clean", {"use_raw_clean": True, "use_core_name": False, "use_sorted_tokens": False}),
                          ("Core Name (Suffix Stripped)", {"use_raw_clean": False, "use_core_name": True, "use_sorted_tokens": False}),
                          ("Token Sorted", {"use_raw_clean": False, "use_core_name": False, "use_sorted_tokens": True}),
                          ("All Representations", {"use_raw_clean": True, "use_core_name": True, "use_sorted_tokens": True})]:
    cfg = NameBlockConfig(**kwargs)
    blk = ExactNameBlocker(name=name_mode, config=cfg)
    blk.fit(target_pool)
    cands = blk.block(s1)
    m = evaluate_candidate_pairs(cands, gt, len(target_pool))
    print(f"[{name_mode}] Pair Recall: {m['pair_recall']*100:.2f}% | Avg Cands: {m['avg_candidates']:.2f}")
""")
    ]
)

# 04_rare_token_blocking.ipynb
create_notebook(
    "04_rare_token_blocking.ipynb",
    "04 — Rare / Informative Token Blocking",
    "Builds an inverted index over informative tokens with document frequency bounds.",
    [
        ("1. Environment & Setup", SETUP_CODE),
        ("2. Load Benchmark Dataset", """s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
target_pool = pd.concat([s2, s3], ignore_index=True)
"""),
        ("3. Run Rare Token Blocker", """from src.business_entity_resolution.blocking import RareTokenConfig, RareTokenBlocker

cfg = RareTokenConfig(max_doc_freq=100, min_token_len=3, top_n_rare=2, max_bucket_size=500)
blk = RareTokenBlocker(name="Block_C", config=cfg)
blk.fit(target_pool)
cands = blk.block(s1)

metrics = evaluate_candidate_pairs(cands, gt, len(target_pool))
print(format_evaluation_summary(metrics, title="Rare Token Blocker (DF<=100, Top-2)"))
print("Token Diagnostics:", blk.get_token_diagnostics())
""")
    ]
)

# 05_char_ngram_retrieval.ipynb
create_notebook(
    "05_char_ngram_retrieval.ipynb",
    "05 — Sparse Character N-Gram Retrieval",
    "Evaluates character TF-IDF sparse retrieval across top-K values without dense memory conversion.",
    [
        ("1. Environment & Setup", SETUP_CODE),
        ("2. Load Benchmark Dataset", """s1, s2, s3, gt = build_or_load_benchmark(n_s1=5000, n_distractors=100000, random_seed=42)
target_pool = pd.concat([s2, s3], ignore_index=True)
"""),
        ("3. Run Retrieval Blocker", """from src.business_entity_resolution.blocking import RetrievalConfig, CharNgramRetrievalBlocker

cfg = RetrievalConfig(top_k=20, ngram_range=(2, 4), sim_threshold=0.30, batch_size=1000)
blk = CharNgramRetrievalBlocker(name="Block_D", config=cfg)
blk.fit(target_pool)
cands = blk.block(s1)

metrics = evaluate_candidate_pairs(cands, gt, len(target_pool))
print(format_evaluation_summary(metrics, title="Character N-Gram Retrieval (Top-K=20)"))
""")
    ]
)

# 06_blocker_ablation.ipynb
create_notebook(
    "06_blocker_ablation.ipynb",
    "06 — Blocker Ablation & Overlap Analysis",
    "Analyzes leave-one-out marginal contribution and pairwise Venn intersections.",
    [
        ("1. Load Ablation & Overlap Reports", """df_ablation = pd.read_csv("reports/blocker_ablation.tsv", sep="\\t")
df_overlap = pd.read_csv("reports/blocker_overlap.tsv", sep="\\t")
print("=== LEAVE-ONE-OUT ABLATION TABLE ===")
print(df_ablation.to_markdown(index=False))
print("\\n=== PAIRWISE OVERLAP TABLE ===")
print(df_overlap.to_markdown(index=False))
""")
    ]
)

# 07_candidate_analysis.ipynb
create_notebook(
    "07_candidate_analysis.ipynb",
    "07 — Candidate Explosion & Failure Analysis",
    "Audits candidate volume percentiles, top candidate explosion entities, and missed match failure modes.",
    [
        ("1. Inspect Failure Modes & Candidate Statistics", """df_fail = pd.read_csv("reports/blocking_failures.tsv", sep="\\t")
df_stats = pd.read_csv("reports/candidate_statistics.tsv", sep="\\t")

print("=== MISSED MATCH FAILURE BREAKDOWN ===")
print(df_fail['failure_category'].value_counts())

print("\\n=== TOP CANDIDATE EXPLOSION ENTITIES ===")
print(df_stats.head(10).to_markdown(index=False))
""")
    ]
)

# 08_final_blocking_tournament.ipynb
create_notebook(
    "08_final_blocking_tournament.ipynb",
    "08 — Final Blocker Tournament & Pareto Frontier",
    "Evaluates the complete Pareto tournament across configurations E0 through E8 and pruning strategies.",
    [
        ("1. Load Persistent Benchmark Table", """df_bench = pd.read_csv("reports/blocking_benchmark.tsv", sep="\\t")
print("=== COMPLETE BLOCKING BENCHMARK TOURNAMENT ===")
print(df_bench[['Experiment', 'Active_Blockers', 'Pair_Recall', 'Entity_Recovery', 'Avg_Candidates', 'Max_Candidates', 'Reduction_Ratio', 'Runtime_s']].to_markdown(index=False))
""")
    ]
)
print("All 8 notebooks generated successfully.")
