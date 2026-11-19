"""Construit le POOL de candidats à annoter (ground truth), par CV :
 top-N TF-IDF  ∪  top-N Sentence-BERT  ∪  quelques offres aléatoires (négatifs probables).

Sortie : data/eval/pool_to_annotate.csv  (à remplir par les DEUX membres : annotator_1 / annotator_2,
valeurs 1 = pertinente, 0 = non pertinente). La colonne `proxy_relevant` est un étiquetage
AUTOMATIQUE par mots-clés du titre, utile UNIQUEMENT pour tester la chaîne technique.

Attention (biais de pooling) : un pool issu des deux modèles favorise les offres qu'ils trouvent ;
les négatifs aléatoires et la double annotation limitent ce biais, pas l'éliminent. Mentionnez-le."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from cvmatch.cleaning import clean_text, fold
from cvmatch.config import get_settings
from cvmatch.data import job_texts, load_jobs
from cvmatch.evaluation import load_cvs
from cvmatch.pii import mask_pii
from cvmatch.ranking import SbertRanker, TfidfRanker
from cvmatch.artifacts import load_embeddings


def proxy(title: str, function: str, kws: list[str]) -> int:
    hay = fold(f"{title} {function}")
    return int(any(re.search(r"\b" + re.escape(fold(k)), hay) for k in kws))


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=6, help="top-N par modèle")
    ap.add_argument("--random", type=int, default=4, help="offres aléatoires par CV")
    ap.add_argument("--no-sbert", action="store_true", help="pool TF-IDF seul (sans modèle)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    jobs = load_jobs(s.jobs_path)
    cvs = load_cvs(s.eval_dir / "cvs")
    meta_path = s.eval_dir / "cv_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    tfidf = TfidfRanker(job_texts(jobs, True))
    sbert = None
    if not a.no_sbert:
        emb = load_embeddings(jobs, s.sbert_model, s.artifacts_dir, True)
        sbert = SbertRanker.from_model(emb, s.sbert_model)

    rng = np.random.default_rng(a.seed)
    rows = []
    for cv_id, text in cvs.items():
        q = clean_text(mask_pii(text))
        idx = [r.index for r in tfidf.rank(q, top_k=a.n)]
        if sbert is not None:
            idx += [r.index for r in sbert.rank(q, top_k=a.n)]
        idx += rng.choice(len(jobs), size=a.random, replace=False).tolist()
        kws = meta.get(cv_id, {}).get("target_keywords", [])
        for i in dict.fromkeys(idx):  # unique, ordre conservé
            r = jobs.iloc[i]
            rows.append(
                {
                    "cv_id": cv_id,
                    "job_id": r["job_id"],
                    "job_title": r["job_title"],
                    "company": r["company"],
                    "job_snippet": re.sub(r"\s+", " ", r["job_description"])[:300],
                    "annotator_1": "",
                    "annotator_2": "",
                    "proxy_relevant": proxy(r["job_title"], r["function"], kws) if kws else "",
                }
            )
    # ordre aléatoire par CV pour ne pas révéler le rang des modèles aux annotateurs
    df = pd.DataFrame(rows).groupby("cv_id", group_keys=False).sample(frac=1, random_state=a.seed)
    out = s.eval_dir / "pool_to_annotate.csv"
    df.to_csv(out, index=False)
    print(f"✔ {len(df)} paires (CV, offre) pour {len(cvs)} CV -> {out}")

    if df["proxy_relevant"].astype(str).str.len().gt(0).any():
        px = df[["cv_id", "job_id", "proxy_relevant"]].rename(columns={"proxy_relevant": "relevant"})
        px.to_csv(s.eval_dir / "proxy_labels.csv", index=False)
        print(f"✔ labels PROXY (test technique uniquement) -> {s.eval_dir / 'proxy_labels.csv'}")


if __name__ == "__main__":
    main()
