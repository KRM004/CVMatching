"""Ablation LLM sur un Top 3 FIGÉ : L0 règles | L1 Gemini complet | A2 Gemini sans description.

Mesures automatiques (affirmations non retrouvées dans les sources, latence) + colonnes de
notation manuelle (`manual_*`) à remplir à deux, puis `cvmatch ablate-llm --summarize`."""
from __future__ import annotations

import time

import pandas as pd

from ..config import Settings
from ..explain.factory import build_explainers
from ..preprocessing import extract_skills, prepare_resume
from ..ranking.base import Ranker
from ..utils.logging import get_logger

log = get_logger(__name__)

MANUAL_COLUMNS = ["manual_evidence_ok", "manual_invented", "manual_utility_1_3"]


def llm_configs(settings: Settings, use_cache: bool = False) -> dict:
    cfg = {"L0_rules": build_explainers(settings, cache=use_cache)["rules"]}
    if settings.gemini_api_key:
        cfg["L1_gemini_full"] = build_explainers(settings, job_context=True, cache=use_cache)["gemini"]
        cfg["A2_gemini_no_job_desc"] = build_explainers(settings, job_context=False,
                                                        cache=use_cache)["gemini"]
    else:
        log.warning("GEMINI_API_KEY absente : seule l'explication à règles (L0) est évaluée.")
    return cfg


def run_llm_ablation(
    jobs: pd.DataFrame, cvs: dict[str, str], ranker: Ranker, settings: Settings,
    use_cache: bool = False, k: int = 3,
) -> pd.DataFrame:
    explainers = llm_configs(settings, use_cache)
    rows = []
    for cv_id, text in cvs.items():
        masked, query = prepare_resume(text, settings.cleaning)
        skills = extract_skills(masked)
        for r in ranker.rank(query, top_k=k):  # Top k figé, identique pour toutes les configs
            row = jobs.iloc[r.index]
            job = {"job_title": row["job_title"], "job_description": row["job_description"]}
            for name, ex in explainers.items():
                t0 = time.perf_counter()
                e = ex.explain(masked, job, skills)
                dt = time.perf_counter() - t0
                if e.source == "gemini":
                    unsupported = e.guard_removed
                else:  # règles : toute compétence citée doit être cohérente avec le CV
                    unsupported = sum(x not in skills for x in e.matching_skills) + sum(
                        x in skills for x in e.missing_skills)
                rows.append({
                    "config": name, "cv_id": cv_id, "job_id": row["job_id"],
                    "job_title": row["job_title"], "source": e.source,
                    "n_matching": len(e.matching_skills), "n_missing": len(e.missing_skills),
                    "unsupported_claims_auto": unsupported, "latency_s": round(dt, 4),
                    "explanation": e.explanation, "evidence": " || ".join(e.evidence),
                    "error": e.error or "", **{c: "" for c in MANUAL_COLUMNS},
                })
    return pd.DataFrame(rows)


def summarize_llm(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"n": ("cv_id", "count"),
           "unsupported_auto_mean": ("unsupported_claims_auto", "mean"),
           "latency_median_s": ("latency_s", "median")}
    for c in MANUAL_COLUMNS:
        if c in df and pd.to_numeric(df[c], errors="coerce").notna().any():
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg[f"{c}_mean"] = (c, "mean")
    return df.groupby("config").agg(**agg).round(3).reset_index()
