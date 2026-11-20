"""API « simple » : recommend_jobs(resume_text, jobs, top_k=3) -> dict (contrat commun)."""
from __future__ import annotations

import pandas as pd

from .config import get_settings
from .explain.rules import RuleExplainer
from .ranking import build_ranker
from .service import MatchingService

_SERVICES: dict = {}


def recommend_jobs(
    resume_text: str, jobs: pd.DataFrame, top_k: int = 3, method: str = "tfidf"
) -> dict:
    """{"resume_skills": [...], "recommendations": [{job_id, job_title, job_description,
    similarity_score}, ...]} — `method` : "tfidf" (aucun téléchargement) ou "sbert"
    (embeddings pré-calculés par `cvmatch embed`)."""
    key = (id(jobs), len(jobs), method)
    if key not in _SERVICES:
        s = get_settings()
        ranker = build_ranker(jobs, s, name=method, cleaning=True)
        _SERVICES[key] = MatchingService(jobs, ranker, {"rules": RuleExplainer(s.language)})
    res = _SERVICES[key].match(resume_text, top_k=top_k, explain=False)
    keep = ("job_id", "job_title", "job_description", "similarity_score")
    return {
        "resume_skills": res.resume_skills,
        "recommendations": [r.model_dump(include=set(keep)) for r in res.recommendations],
    }
