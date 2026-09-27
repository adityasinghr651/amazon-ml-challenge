"""
utils.py - Configuration loading, logging, and shared utility functions.

Central config contract: every script calls load_config() to get hyperparameters.
No magic numbers should appear in any script — they all live in config.yaml.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.yaml"
)

_cached_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the project YAML config. Caches result after first load.
    Falls back to the repo-root config.yaml if no path given.
    """
    global _cached_config
    if _cached_config is not None and config_path is None:
        return _cached_config

    path = config_path or _DEFAULT_CONFIG_PATH
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Create config.yaml in the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if config_path is None:
        _cached_config = cfg
    return cfg


def get_blocking_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract the blocking-specific config dict expected by
    blocking.generate_candidates().
    """
    if cfg is None:
        cfg = load_config()
    blk = cfg["blocking"]
    return {
        "active_blocks": blk["active_blocks"],
        "rare_token_percentile": blk["rare_token_percentile"],
        "ann_neighbors": blk["ann_neighbors"],
        "ann_max_distance": blk["ann_max_distance"],
        "enable_pruning": blk["enable_pruning"],
        "max_candidates_before_pruning": blk["max_candidates_before_pruning"],
        "prune_top_k": blk["prune_top_k"],
    }


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-wide logger."""
    logger = logging.getLogger("ber")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_experiment(
    cfg: Dict[str, Any],
    experiment_name: str,
    blocking_desc: str,
    features_desc: str,
    model_name: str,
    threshold: float,
    metrics: Dict[str, float],
    avg_candidates: float,
    runtime: float,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Append one row to the experiment log CSV.
    Creates the file with header if it doesn't exist.
    """
    log_path = cfg.get("experiment_log", "experiments/experiment_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "Experiment,Blocking,Features,Model,Threshold,"
                "Candidate Recall,Precision,Recall,F0.5,"
                "Avg Candidates,Runtime\n"
            )
        cand_recall = metrics.get("candidate_recall", metrics.get("cand_recall", ""))
        precision = metrics.get("macro_precision", "")
        recall = metrics.get("macro_recall", "")
        f05 = metrics.get("macro_f05", "")

        f.write(
            f"{experiment_name},{blocking_desc},{features_desc},{model_name},"
            f"{threshold:.4f},{cand_recall},{precision},{recall},{f05},"
            f"{avg_candidates:.1f},{runtime:.2f}\n"
        )
