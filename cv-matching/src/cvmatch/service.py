"""Orchestrateur : PDF/texte -> masquage PII -> (cleaning) -> ranking -> explications."""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from .artifacts import load_embeddings
from .cleaning import clean_text, raw_text
from .config import Settings, get_settings
from .data import job_texts, load_jobs
from .explain.gemini import GeminiExplainer
from .explain.rules import RuleExplainer
from .pii import mask_pii
from .ranking import Ranker, SbertRanker, TfidfRanker, diversify
from .schemas import MatchResponse, Recommendation
from .skills import extract_skills


def _none_if_empty(v: object) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


class MatchingService:
    def __init__(
        self,
        jobs: pd.DataFrame,
        ranker: Ranker,
        explainers: dict,
        default_explainer: str = "rules",
        cleaning: bool = True,
        diversify_results: bool = True,
    ):
        self.jobs = jobs.reset_index(drop=True)
        self.ranker = ranker
        self.explainers = explainers
        self.default_explainer = default_explainer if default_explainer in explainers else "rules"
        self.cleaning = cleaning
        self.diversify_results = diversify_results
        self._titles = self.jobs["job_title"].tolist()

    def preprocess(self, resume_text: str) -> str:
        """Masquage PII (toujours) puis nettoyage (si activé)."""
        masked = mask_pii(resume_text)
        return clean_text(masked) if self.cleaning else raw_text(masked)

    def match(
        self,
        resume_text: str,
        top_k: int = 3,
        explain: bool = True,
        explainer: Optional[str] = None,
    ) -> MatchResponse:
        if not resume_text or not resume_text.strip():
            raise ValueError("CV vide : aucun texte à analyser.")
        timings: dict[str, float] = {}
        warnings: list[str] = []

        t0 = time.perf_counter()
        masked = mask_pii(resume_text)  # c'est CE texte qui peut partir vers Gemini
        query = clean_text(masked) if self.cleaning else raw_text(masked)
        resume_skills = extract_skills(masked)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pool = max(top_k * 5, 15) if self.diversify_results else top_k
        ranked = self.ranker.rank(query, top_k=pool)
        if self.diversify_results:
            ranked = diversify(ranked, self._titles, top_k)
        ranked = ranked[:top_k]
        timings["ranking"] = (time.perf_counter() - t0) * 1000

        exp_name = explainer or self.default_explainer
        exp = self.explainers.get(exp_name) if explain else None
        if explain and exp is None:
            warnings.append(f"Explainer '{exp_name}' indisponible : explications désactivées.")

        recs: list[Recommendation] = []
        t0 = time.perf_counter()
        for r in ranked:
            row = self.jobs.iloc[r.index]
            job = {"job_title": row["job_title"], "job_description": row["job_description"]}
            e = exp.explain(masked, job, resume_skills) if exp else None
            if e is not None and e.source == "rules_fallback" and e.error:
                msg = f"Gemini indisponible, explication à règles utilisée ({e.error})"
                if msg not in warnings:
                    warnings.append(msg)
            recs.append(
                Recommendation(
                    job_id=row["job_id"].item() if hasattr(row["job_id"], "item") else row["job_id"],
                    job_title=row["job_title"],
                    job_description=row["job_description"],
                    similarity_score=round(r.score, 4),
                    company=_none_if_empty(row.get("company")),
                    city=_none_if_empty(row.get("city")),
                    sector=_none_if_empty(row.get("sector")),
                    explanation=e,
                )
            )
        timings["explain"] = (time.perf_counter() - t0) * 1000
        timings["total"] = sum(timings.values())
        return MatchResponse(
            resume_skills=resume_skills,
            recommendations=recs,
            ranker=self.ranker.name,
            explainer=exp_name if exp else "",
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            warnings=warnings,
        )


def build_explainers(settings: Settings) -> dict:
    explainers: dict = {"rules": RuleExplainer(settings.language)}
    explainers["gemini"] = GeminiExplainer(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        language=settings.language,
        include_job_description=settings.job_context,
        cache_dir=settings.cache_dir / "gemini",
        min_interval=settings.gemini_min_interval,
    )
    return explainers


def build_ranker(jobs: pd.DataFrame, settings: Settings, encoder=None) -> Ranker:
    if settings.ranker == "tfidf":
        return TfidfRanker(job_texts(jobs, settings.cleaning))
    if settings.ranker == "sbert":
        emb = load_embeddings(jobs, settings.sbert_model, settings.artifacts_dir, settings.cleaning)
        if encoder is not None:
            return SbertRanker(emb, encoder)
        return SbertRanker.from_model(emb, settings.sbert_model)
    raise ValueError(f"RANKER inconnu : {settings.ranker} (sbert | tfidf)")


def build_service(
    settings: Optional[Settings] = None, jobs: Optional[pd.DataFrame] = None, encoder=None
) -> MatchingService:
    settings = settings or get_settings()
    if jobs is None:
        if not settings.jobs_path.exists():
            raise FileNotFoundError(
                f"Catalogue introuvable : {settings.jobs_path}\n"
                "Lancez d'abord :  python scripts/prepare_data.py"
            )
        jobs = load_jobs(settings.jobs_path)
    return MatchingService(
        jobs,
        build_ranker(jobs, settings, encoder),
        build_explainers(settings),
        default_explainer=settings.explainer,
        cleaning=settings.cleaning,
    )
