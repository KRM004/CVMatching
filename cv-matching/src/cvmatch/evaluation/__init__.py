"""Évaluation : ground truth, métriques, ablations ML / LLM, rapport."""
from .annotation import build_pool, finalize_labels, load_cvs, load_labels
from .metrics import delta, evaluate_ranker, precision_at_k, summarize

__all__ = ["build_pool", "finalize_labels", "load_cvs", "load_labels", "delta",
           "evaluate_ranker", "precision_at_k", "summarize"]
