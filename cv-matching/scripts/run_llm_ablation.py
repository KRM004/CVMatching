"""Ablation LLM (Membre 2) sur un Top 3 FIGÉ par CV : L0 règles | L1 Gemini complet | A2 Gemini sans
description d'offre. Mesures automatiques (affirmations non retrouvées dans les sources, latence)
+ colonnes de notation manuelle (grille : preuves vérifiables, compétence inventée, utilité 1-3).

  python scripts/run_llm_ablation.py --ranker tfidf --n 20 --no-cache
  python scripts/run_llm_ablation.py --summarize          # après remplissage manuel du CSV
"""
import argparse
import time

import pandas as pd

from cvmatch.cleaning import clean_text, raw_text
from cvmatch.config import get_settings
from cvmatch.data import load_jobs
from cvmatch.evaluation import load_cvs
from cvmatch.explain.gemini import GeminiExplainer
from cvmatch.explain.rules import RuleExplainer
from cvmatch.pii import mask_pii
from cvmatch.service import build_ranker
from cvmatch.skills import extract_skills

MANUAL = ["manual_evidence_ok", "manual_invented", "manual_utility_1_3"]


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ranker", choices=["tfidf", "sbert"], default="tfidf")
    ap.add_argument("--n", type=int, default=20, help="nombre de CV")
    ap.add_argument("--no-cache", action="store_true", help="mesure de latence sans cache")
    ap.add_argument("--summarize", action="store_true", help="résume le CSV après notation manuelle")
    a = ap.parse_args()
    out = s.results_dir / "ablation_llm.csv"

    if a.summarize:
        df = pd.read_csv(out)
        cols = [c for c in MANUAL if df[c].notna().any()]
        agg = {"n": ("cv_id", "count"), "unsupported_auto_mean": ("unsupported_claims_auto", "mean"),
               "latency_median_s": ("latency_s", "median")}  # fmt: skip
        agg.update({f"{c}_mean": (c, "mean") for c in cols})
        print(df.groupby("config").agg(**agg).round(3).to_string())
        return

    s.ranker = a.ranker
    jobs = load_jobs(s.jobs_path)
    ranker = build_ranker(jobs, s)
    cvs = dict(list(load_cvs(s.eval_dir / "cvs").items())[: a.n])

    cache = None if a.no_cache else s.cache_dir / "gemini_ablation"
    explainers = {"L0_rules": RuleExplainer(s.language)}
    if s.gemini_api_key:
        mk = lambda ctx: GeminiExplainer(  # noqa: E731
            s.gemini_api_key, s.gemini_model, s.language, ctx, cache, s.gemini_min_interval)
        explainers["L1_gemini_full"] = mk(True)
        explainers["A2_gemini_no_job_desc"] = mk(False)
    else:
        print("⚠ GEMINI_API_KEY absente : seules les explications à règles (L0) seront évaluées.")

    rows = []
    for cv_id, text in cvs.items():
        masked = mask_pii(text)
        skills = extract_skills(masked)
        query = clean_text(masked) if s.cleaning else raw_text(masked)  # comme l'API
        top3 = ranker.rank(query, top_k=3)  # Top 3 figé, identique pour toutes les configs
        for r in top3:
            row = jobs.iloc[r.index]
            job = {"job_title": row["job_title"], "job_description": row["job_description"]}
            for name, ex in explainers.items():
                t0 = time.perf_counter()
                e = ex.explain(masked, job, skills)
                dt = time.perf_counter() - t0
                if e.source == "gemini":
                    unsupported = e.guard_removed  # éléments retirés par le garde-fou de fidélité
                else:  # règles : toute compétence citée doit être détectée dans le CV (dictionnaire)
                    unsupported = sum(1 for x in e.matching_skills if x not in skills) + sum(
                        1 for x in e.missing_skills if x in skills)
                rows.append({
                    "config": name, "cv_id": cv_id, "job_id": row["job_id"], "job_title": row["job_title"],
                    "source": e.source, "n_matching": len(e.matching_skills),
                    "n_missing": len(e.missing_skills),
                    "unsupported_claims_auto": unsupported,
                    "latency_s": round(dt, 3), "explanation": e.explanation,
                    "evidence": " || ".join(e.evidence), "error": e.error or "",
                    **{c: "" for c in MANUAL},
                })  # fmt: skip
    s.results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby("config").agg(n=("cv_id", "count"),
          unsupported_mean=("unsupported_claims_auto", "mean"),
          latency_median_s=("latency_s", "median")).round(3).to_string())  # fmt: skip
    print(f"\n✔ {out}\nRemplissez maintenant {MANUAL} (à deux), puis : python scripts/run_llm_ablation.py --summarize")
    print("Rappel : Gemini est jugé sur la fidélité aux sources, pas sur la Precision@3.")


if __name__ == "__main__":
    main()
