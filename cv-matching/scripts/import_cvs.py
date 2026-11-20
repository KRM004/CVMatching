"""Importe un grand corpus de CV (ex. 5054 CV) -> data/processed/cvs.parquet (texte déjà masqué).

Sources acceptées :
  - un DOSSIER de .pdf / .txt / .docx  (sous-dossier = catégorie, ex. cvs/INFORMATIQUE/cv1.pdf)
  - un FICHIER .csv / .parquet / .xlsx (une ligne = un CV ; colonnes détectées automatiquement)

  python scripts/import_cvs.py --source data/raw/cvs/
  python scripts/import_cvs.py --source data/raw/Resume.csv --text-col Resume_str --category-col Category
  python scripts/import_cvs.py --source data/raw/Resume.csv --sample-eval 60   # + 60 CV pour annotation

`--sample-eval N` copie N CV (stratifiés par catégorie) dans data/eval/cvs/ : ce sont EUX qu'on annote
à deux (étapes 4-6 du README). Annoter 5054 CV à la main est irréaliste ; les 5054 servent au
traitement en lot (scripts/batch_match.py : couverture, latence, cohérence par catégorie).
"""
import argparse
from pathlib import Path

from cvmatch.config import get_settings
from cvmatch.cv_corpus import normalize_corpus, read_cv_folder, read_cv_table, sample_for_annotation
from cvmatch.data import write_table


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="dossier de CV ou fichier csv/parquet/xlsx")
    ap.add_argument("--text-col", help="colonne du texte (sinon détection auto)")
    ap.add_argument("--id-col", help="colonne identifiant (sinon cv_00000, cv_00001...)")
    ap.add_argument("--category-col", help="colonne catégorie/métier (optionnelle)")
    ap.add_argument("--min-chars", type=int, default=200, help="CV plus courts ignorés")
    ap.add_argument("--out", type=Path, default=s.root / "data" / "processed" / "cvs.parquet")
    ap.add_argument("--sample-eval", type=int, default=0, help="N CV copiés dans data/eval/cvs/")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.source.is_dir():
        raw, errors = read_cv_folder(a.source)
        if errors:
            print(f"⚠ {len(errors)} fichiers illisibles (PDF scannés ?), ex. : {errors[:3]}")
    else:
        raw = read_cv_table(a.source, a.text_col, a.id_col, a.category_col)
    corpus, stats = normalize_corpus(raw, min_chars=a.min_chars)
    write_table(corpus, a.out)

    print(f"✔ {stats['n_kept']:,} CV importés -> {a.out}")
    print(f"  lus : {stats['n_input']:,} | trop courts : {stats['n_too_short']:,} | "
          f"doublons : {stats['n_duplicates']:,}")  # fmt: skip
    lengths = corpus["text"].str.split().str.len()
    print(f"  longueur (mots) : médiane {lengths.median():.0f}, min {lengths.min()}, max {lengths.max()}")
    if corpus["category"].str.len().gt(0).any():
        print("  catégories (top 10) :")
        print(corpus["category"].value_counts().head(10).to_string())

    if a.sample_eval:
        out_dir = s.eval_dir / "cvs"
        out_dir.mkdir(parents=True, exist_ok=True)
        sample = sample_for_annotation(corpus, a.sample_eval, a.seed)
        for cv_id, text in zip(sample["cv_id"], sample["text"]):
            (out_dir / f"{cv_id}.txt").write_text(text, encoding="utf-8")
        print(f"✔ {len(sample)} CV copiés dans {out_dir} pour l'annotation (étapes 4-6)")


if __name__ == "__main__":
    main()
