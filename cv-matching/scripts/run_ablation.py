"""Ablation ML (Membre 1) : B0 TF-IDF  |  F1 Sentence-BERT + cleaning  |  A1 SBERT sans cleaning.

Mêmes CV, même pool d'offres annotées, même ordre : un SEUL facteur change à la fois.
Sorties : results/ablation_ml.csv (par CV) et results/ablation_ml_summary.csv (moyennes, deltas).

  python scripts/run_ablation.py                 # vraies annotations : data/eval/labels.csv
  python scripts/run_ablation.py --proxy         # TEST TECHNIQUE avec labels proxy (pas de résultats !)
  python scripts/run_ablation.py --only-tfidf    # sans modèle SBERT
"""
import argparse

import pandas as pd

from cvmatch.artifacts import load_embeddings
from cvmatch.cleaning import clean_text, raw_text
from cvmatch.config import get_settings
from cvmatch.data import job_texts, load_jobs
from cvmatch.evaluation import delta, evaluate_ranker, load_cvs, load_labels, summarize
from cvmatch.pii import mask_pii
from cvmatch.ranking import SbertRanker, TfidfRanker
from cvmatch.ranking.sbert import load_sentence_transformer


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--proxy", action="store_true", help="labels proxy (smoke test uniquement)")
    ap.add_argument("--only-tfidf", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    labels_path = s.eval_dir / ("proxy_labels.csv" if a.proxy else "labels.csv")
    tag = "_PROXY" if a.proxy else ""
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} introuvable. Voir README (étapes 5-6 : annotation).")
    if a.proxy:
        print("⚠ MODE PROXY : labels automatiques par mots-clés -> NE PAS rapporter ces chiffres.")

    jobs = load_jobs(s.jobs_path)
    cvs = load_cvs(s.eval_dir / "cvs")
    labels = load_labels(labels_path)
    print(f"{len(jobs):,} offres | {len(cvs)} CV | {len(labels)} CV annotés")

    prep_clean = lambda t: clean_text(mask_pii(t))  # noqa: E731
    prep_raw = lambda t: raw_text(mask_pii(t))  # noqa: E731

    runs = {"B0_tfidf": (TfidfRanker(job_texts(jobs, True)), prep_clean)}
    if not a.only_tfidf:
        enc = load_sentence_transformer(s.sbert_model)
        runs["F1_sbert_clean"] = (
            SbertRanker(load_embeddings(jobs, s.sbert_model, s.artifacts_dir, True), enc), prep_clean)  # fmt: skip
        runs["A1_sbert_noclean"] = (
            SbertRanker(load_embeddings(jobs, s.sbert_model, s.artifacts_dir, False), enc), prep_raw)  # fmt: skip

    per_cv, summaries = [], {}
    for name, (ranker, prep) in runs.items():
        ranker.rank("warm up", top_k=1)  # évite de compter le chargement initial dans la latence
        df = evaluate_ranker(ranker, jobs, cvs, labels, preprocess=prep, k=a.k)
        df.insert(0, "config", name)
        per_cv.append(df)
        summaries[name] = summarize(df, name)
        print(f"{name:18s} P@3 = {summaries[name]['mean_p_at_3']:.3f} "
              f"(max possible {summaries[name]['mean_p_at_3_max']:.3f}) | "
              f"latence médiane = {summaries[name]['median_latency_s'] * 1000:.1f} ms")  # fmt: skip

    s.results_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_cv).to_csv(s.results_dir / f"ablation_ml{tag}.csv", index=False)
    summ = pd.DataFrame(summaries.values())
    if "F1_sbert_clean" in summaries:
        f1 = summaries["F1_sbert_clean"]["mean_p_at_3"]
        summ["delta_vs_F1"] = [delta(f1, v["mean_p_at_3"]) for v in summaries.values()]
        print(f"\nΔ (F1 - B0)  gain de SBERT sur TF-IDF : {delta(f1, summaries['B0_tfidf']['mean_p_at_3']):+.3f}")
        print(f"Δ (F1 - A1)  contribution du cleaning  : {delta(f1, summaries['A1_sbert_noclean']['mean_p_at_3']):+.3f}")  # fmt: skip
        print("Si un delta est nul ou négatif, le composant ne montre pas de bénéfice sur CE test.")
    summ.to_csv(s.results_dir / f"ablation_ml_summary{tag}.csv", index=False)
    print(f"\n✔ results/ablation_ml{tag}.csv  et  ablation_ml_summary{tag}.csv")


if __name__ == "__main__":
    main()
