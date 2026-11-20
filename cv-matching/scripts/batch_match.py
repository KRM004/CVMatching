"""Matching EN LOT sur tout le corpus importé (ex. 5054 CV) : Top 3 par CV, sans LLM.

  python scripts/batch_match.py                      # ranker du .env (sbert par défaut)
  python scripts/batch_match.py --ranker tfidf       # baseline, rapide, sans modèle
  python scripts/batch_match.py --limit 500          # essai sur 500 CV

Sorties :
  results/batch_top3_<ranker>.csv     une ligne par (CV, rang) : offre, score, catégorie du CV
  results/batch_summary_<ranker>.csv  statistiques globales (latence, scores, couverture du catalogue)
  results/batch_by_category_<ranker>.csv  (si catégories) secteurs d'offres recommandés par catégorie

Gemini n'est PAS appelé ici (5054 x 3 appels dépasseraient le quota gratuit) : les explications
se démontrent sur l'échantillon annoté (run_llm_ablation.py) et dans l'application.
"""
import argparse
import time

import numpy as np
import pandas as pd

from cvmatch.cleaning import clean_text, raw_text
from cvmatch.config import get_settings
from cvmatch.data import load_jobs, read_table
from cvmatch.ranking import diversify
from cvmatch.ranking.base import top_k_from_scores
from cvmatch.service import build_ranker


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cvs", default=str(s.root / "data" / "processed" / "cvs.parquet"))
    ap.add_argument("--ranker", choices=["tfidf", "sbert"], default=s.ranker)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256, help="CV traités par lot")
    ap.add_argument("--limit", type=int, default=0, help="ne traiter que les N premiers CV")
    a = ap.parse_args()

    s.ranker = a.ranker
    jobs = load_jobs(s.jobs_path)
    corpus = read_table(a.cvs)
    if a.limit:
        corpus = corpus.head(a.limit)
    print(f"{len(corpus):,} CV | {len(jobs):,} offres | ranker : {a.ranker}")

    t0 = time.perf_counter()
    ranker = build_ranker(jobs, s)
    load_s = time.perf_counter() - t0
    prep = clean_text if s.cleaning else raw_text  # texte déjà masqué à l'import
    titles = jobs["job_title"].tolist()
    pool = max(a.k * 5, 15)

    rows, per_batch = [], []
    t_all = time.perf_counter()
    for start in range(0, len(corpus), a.batch):
        part = corpus.iloc[start : start + a.batch]
        t_b = time.perf_counter()
        scores = ranker.score_many([prep(t) for t in part["text"]])
        for (cv_id, cat), sc in zip(zip(part["cv_id"], part["category"]), scores):
            ranked = diversify(top_k_from_scores(sc, pool), titles, a.k)
            for rank, r in enumerate(ranked, start=1):
                job = jobs.iloc[r.index]
                rows.append({
                    "cv_id": cv_id, "cv_category": cat, "rank": rank, "job_id": job["job_id"],
                    "job_title": job["job_title"], "job_sector": job["sector"],
                    "score": round(float(r.score), 4),
                })  # fmt: skip
        per_batch.append((time.perf_counter() - t_b) / max(len(part), 1))
        done = min(start + a.batch, len(corpus))
        print(f"  {done:,}/{len(corpus):,} CV traités", flush=True)
    total_s = time.perf_counter() - t_all

    df = pd.DataFrame(rows)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(s.results_dir / f"batch_top3_{a.ranker}.csv", index=False)

    top1 = df[df["rank"] == 1]["score"]
    summary = {
        "ranker": a.ranker,
        "n_cv": int(corpus.shape[0]),
        "n_jobs": int(len(jobs)),
        "model_load_s": round(load_s, 2),
        "total_s": round(total_s, 2),
        "ms_per_cv": round(1000 * float(np.mean(per_batch)), 2),
        "top1_score_mean": round(float(top1.mean()), 4),
        "top1_score_median": round(float(top1.median()), 4),
        "share_cv_top1_below_0.2": round(float((top1 < 0.2).mean()), 4),
        "distinct_jobs_recommended": int(df["job_id"].nunique()),
        "catalog_coverage": round(df["job_id"].nunique() / len(jobs), 4),
    }
    pd.DataFrame([summary]).to_csv(s.results_dir / f"batch_summary_{a.ranker}.csv", index=False)
    print("\nRésumé :")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")

    if df["cv_category"].astype(str).str.len().gt(0).any():
        by_cat = (df[df["rank"] == 1].groupby(["cv_category", "job_sector"]).size()
                  .rename("n").reset_index().sort_values(["cv_category", "n"], ascending=[True, False]))  # fmt: skip
        by_cat.to_csv(s.results_dir / f"batch_by_category_{a.ranker}.csv", index=False)
        print("\nSecteur le plus recommandé (Top 1) par catégorie de CV :")
        print(by_cat.groupby("cv_category").head(1).to_string(index=False))
    print(f"\n✔ results/batch_top3_{a.ranker}.csv et batch_summary_{a.ranker}.csv")
    print("Rappel : le score est une proximité textuelle, pas une probabilité d'embauche.")


if __name__ == "__main__":
    main()
