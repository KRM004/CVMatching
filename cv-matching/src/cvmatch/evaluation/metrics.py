"""Métriques : Precision@k, latence, deltas d'ablation, évaluation sur pool figé."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd


def _py(v):
    return v.item() if hasattr(v, "item") else v


def precision_at_k(ranked_ids: list, relevant_ids: set, k: int = 3) -> float:
    """Precision@k = nombre d'offres pertinentes parmi les k premières / k."""
    return sum(1 for j in ranked_ids[:k] if j in relevant_ids) / k


def evaluate_ranker(
    ranker,
    jobs: pd.DataFrame,
    cvs: dict[str, str],
    labels: dict[str, dict],
    preprocess: Callable[[str], str] = lambda t: t,
    k: int = 3,
) -> pd.DataFrame:
    """Classe, pour chaque CV, UNIQUEMENT les offres de son pool annoté (mêmes candidats pour
    toutes les configurations). Un seul facteur doit changer entre deux appels."""
    index_of = {jid: i for i, jid in enumerate(jobs["job_id"].tolist())}
    rows = []
    for cv_id, pool in labels.items():
        if cv_id not in cvs:
            continue
        cand_ids = [j for j in pool if j in index_of]
        relevant = {j for j, r in pool.items() if r == 1}
        t0 = time.perf_counter()
        ranked = ranker.rank(
            preprocess(cvs[cv_id]), top_k=k, candidates=[index_of[j] for j in cand_ids]
        )
        latency = time.perf_counter() - t0
        top_ids = [_py(jobs["job_id"].iloc[r.index]) for r in ranked]
        rows.append(
            {
                "cv_id": cv_id,
                "top_job_ids": "|".join(map(str, top_ids)),
                "n_candidates": len(cand_ids),
                "n_relevant": len(relevant & set(cand_ids)),
                "p_at_k": precision_at_k(top_ids, relevant, k),
                "p_at_k_max": min(k, len(relevant & set(cand_ids))) / k,
                "latency_s": latency,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str) -> dict:
    empty = len(df) == 0
    return {
        "config": name,
        "n_cv": int(len(df)),
        "mean_p_at_k": float("nan") if empty else float(df["p_at_k"].mean()),
        "mean_p_at_k_max": float("nan") if empty else float(df["p_at_k_max"].mean()),
        "median_latency_ms": float("nan") if empty else float(df["latency_s"].median() * 1000),
    }


def delta(full: float, ablated: float) -> float:
    """Impact = score version complète - score version ablatée (ex. 0,80 - 0,74 = +0,06)."""
    return round(full - ablated, 4)


def median_or_nan(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")
