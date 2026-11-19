"""Fonction d'intégration du Membre 1 (M1.7) : recommend_jobs(resume_text, jobs, top_k=3)."""
from __future__ import annotations

import pandas as pd

from .cleaning import clean_text
from .config import get_settings
from .data import job_texts
from .pii import mask_pii
from .ranking import SbertRanker, TfidfRanker, diversify
from .skills import extract_skills

_RANKERS: dict = {}


def recommend_jobs(
    resume_text: str, jobs: pd.DataFrame, top_k: int = 3, method: str = "tfidf"
) -> dict:
    """Sortie conforme au contrat commun :
    {"resume_skills": [...], "recommendations": [{job_id, job_title, job_description,
    similarity_score}, ...]}  (max `top_k` offres, triées par score décroissant).

    `method` : "tfidf" (baseline, aucun téléchargement) ou "sbert" (calcule les embeddings
    si nécessaire ; pour un gros catalogue, préférez scripts/build_artifacts.py)."""
    key = (id(jobs), len(jobs), method)
    if key not in _RANKERS:
        texts = job_texts(jobs, cleaning=True)
        if method == "tfidf":
            _RANKERS[key] = TfidfRanker(texts)
        elif method == "sbert":
            from .ranking.sbert import load_sentence_transformer

            enc = load_sentence_transformer(get_settings().sbert_model)
            import numpy as np

            emb = np.vstack([enc(texts[i : i + 64]) for i in range(0, len(texts), 64)])
            _RANKERS[key] = SbertRanker(emb, enc)
        else:
            raise ValueError("method doit valoir 'tfidf' ou 'sbert'")
    ranker = _RANKERS[key]
    masked = mask_pii(resume_text)
    ranked = diversify(ranker.rank(clean_text(masked), top_k=max(top_k * 5, 15)),
                       jobs["job_title"].tolist(), top_k)  # fmt: skip
    recs = []
    for r in ranked[:top_k]:
        row = jobs.iloc[r.index]
        recs.append(
            {
                "job_id": row["job_id"].item() if hasattr(row["job_id"], "item") else row["job_id"],
                "job_title": row["job_title"],
                "job_description": row["job_description"],
                "similarity_score": round(r.score, 4),
            }
        )
    return {"resume_skills": extract_skills(masked), "recommendations": recs}
