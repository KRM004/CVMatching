"""Pré-calcule les embeddings Sentence-BERT des offres (une seule fois, flux 'A' du pipeline).

  python scripts/build_artifacts.py                 # variante nettoyée (F1) + brute (A1)
  python scripts/build_artifacts.py --only clean
"""
import argparse

from cvmatch.artifacts import build_embeddings
from cvmatch.config import get_settings
from cvmatch.data import load_jobs


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", choices=["clean", "raw"], help="une seule variante")
    ap.add_argument("--model", default=s.sbert_model)
    a = ap.parse_args()

    jobs = load_jobs(s.jobs_path)
    print(f"{len(jobs):,} offres, modèle : {a.model}")
    for cleaning in ([True] if a.only == "clean" else [False] if a.only == "raw" else [True, False]):
        print(f"→ embeddings ({'clean' if cleaning else 'raw'})")
        out = build_embeddings(jobs, a.model, s.artifacts_dir, cleaning=cleaning)
        print(f"  ✔ {out}")


if __name__ == "__main__":
    main()
