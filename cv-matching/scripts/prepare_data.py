"""M1.1/M1.2 : télécharge le dataset Rekrute (Hugging Face), l'inspecte, le nettoie, le sauvegarde.

  python scripts/prepare_data.py                       # télécharge depuis Hugging Face
  python scripts/prepare_data.py --sectors informatique ingenieur --max-rows 8000
  python scripts/prepare_data.py --synthetic           # offres FICTIVES (test hors-ligne)
"""
import argparse
from pathlib import Path

from cvmatch.config import get_settings
from cvmatch.data import load_raw_jobs, prepare_jobs, validate_jobs, write_table


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, help="fichier local .parquet/.csv (sinon Hugging Face)")
    ap.add_argument("--synthetic", action="store_true", help="données fictives de test")
    ap.add_argument("--min-year", type=int, default=2019)
    ap.add_argument("--max-rows", type=int, default=15000)
    ap.add_argument("--sectors", nargs="*", help="mots-clés secteur/fonction/titre (ex: informatique)")
    ap.add_argument("--out", type=Path, default=s.jobs_path)
    a = ap.parse_args()

    if a.synthetic:
        from cvmatch.synthetic import make_synthetic_jobs

        raw = make_synthetic_jobs()
        print("⚠ Données SYNTHÉTIQUES : ne pas utiliser pour rapporter des résultats.")
    else:
        raw = load_raw_jobs(a.raw)
    print(f"Brut : {len(raw):,} lignes, {raw.shape[1]} colonnes")
    print("Colonnes :", list(raw.columns))
    print("Valeurs manquantes (top 8) :")
    print(raw.isna().sum().sort_values(ascending=False).head(8).to_string())

    jobs = prepare_jobs(raw, min_year=a.min_year, max_rows=a.max_rows, sectors=a.sectors)
    validate_jobs(jobs)
    write_table(jobs, a.out)
    print(f"\n✔ {len(jobs):,} offres nettoyées -> {a.out}")
    print(jobs[["job_id", "job_title", "city", "sector", "year"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
