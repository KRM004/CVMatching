"""Évaluation : Precision@3, latence, ablations (B0 / F1 / A1) sur un pool d'offres figé."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def precision_at_k(ranked_ids: list, relevant_ids: set, k: int = 3) -> float:
    """Precision@k = nombre d'offres pertinentes parmi les k premières / k."""
    return sum(1 for j in ranked_ids[:k] if j in relevant_ids) / k


def load_cvs(cv_dir: Path) -> dict[str, str]:
    """Charge les CV de test : *.txt (texte) et *.pdf (extraction PyMuPDF). Le nom = cv_id."""
    cvs: dict[str, str] = {}
    for p in sorted(Path(cv_dir).glob("*")):
        if p.suffix.lower() == ".txt":
            cvs[p.stem] = p.read_text(encoding="utf-8")
        elif p.suffix.lower() == ".pdf" and p.stem not in cvs:
            from .pdf_extractor import extract_pdf_text

            cvs[p.stem] = extract_pdf_text(p)
    return cvs


def load_labels(path: Path, column: str = "relevant") -> dict[str, dict]:
    """labels.csv : cv_id, job_id, relevant(0/1) -> {cv_id: {job_id: 0/1}} (pool figé par CV)."""
    df = pd.read_csv(path)
    df = df[df[column].notna()]
    out: dict[str, dict] = {}
    for cv_id, job_id, rel in zip(df["cv_id"].astype(str), df["job_id"], df[column]):
        out.setdefault(cv_id, {})[job_id.item() if hasattr(job_id, "item") else job_id] = int(rel)
    return out


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
        top_ids = [jobs["job_id"].iloc[r.index] for r in ranked]
        top_ids = [t.item() if hasattr(t, "item") else t for t in top_ids]
        rows.append(
            {
                "cv_id": cv_id,
                "top3_job_ids": "|".join(map(str, top_ids)),
                "n_candidates": len(cand_ids),
                "n_relevant": len(relevant & set(cand_ids)),
                "p_at_3": precision_at_k(top_ids, relevant, k),
                "p_at_3_max": min(k, len(relevant & set(cand_ids))) / k,
                "latency_s": latency,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str) -> dict:
    return {
        "config": name,
        "n_cv": int(len(df)),
        "mean_p_at_3": float(df["p_at_3"].mean()) if len(df) else float("nan"),
        "mean_p_at_3_max": float(df["p_at_3_max"].mean()) if len(df) else float("nan"),
        "median_latency_s": float(df["latency_s"].median()) if len(df) else float("nan"),
    }


def delta(full: float, ablated: float) -> float:
    """Impact = score version complète - score version ablatée (ex. 0,80 - 0,74 = +0,06)."""
    return round(full - ablated, 4)


def finalize_labels(pool_csv: Path, out_csv: Path, disagreements_csv: Optional[Path] = None) -> dict:
    """Fusionne les 2 annotateurs : pertinent = les deux disent 1. Rapporte accord et kappa."""
    from sklearn.metrics import cohen_kappa_score

    df = pd.read_csv(pool_csv)
    df = df[df["annotator_1"].notna() & df["annotator_2"].notna()].copy()
    if df.empty:
        raise ValueError("Aucune ligne annotée par les DEUX annotateurs.")
    a1, a2 = df["annotator_1"].astype(int), df["annotator_2"].astype(int)
    df["relevant"] = ((a1 == 1) & (a2 == 1)).astype(int)
    dis = df[a1 != a2]
    if disagreements_csv is not None:
        dis.to_csv(disagreements_csv, index=False)
    df[["cv_id", "job_id", "relevant"]].to_csv(out_csv, index=False)
    kappa = float(cohen_kappa_score(a1, a2)) if a1.nunique() > 1 or a2.nunique() > 1 else float("nan")
    return {
        "n_pairs": int(len(df)),
        "n_relevant": int(df["relevant"].sum()),
        "agreement": float((a1 == a2).mean()),
        "cohen_kappa": kappa,
        "n_disagreements": int(len(dis)),
    }


def median_or_nan(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")
