"""Import d'un grand corpus de CV (ex. 5054 CV) : dossier de PDF/TXT ou fichier CSV/Parquet.

Sortie normalisée : DataFrame [cv_id, text, category, source] où `text` est DÉJÀ masqué (PII)
-> on ne stocke jamais d'e-mail / téléphone en clair dans data/processed/.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .cleaning import fold, normalize_spaces
from .pii import mask_pii

# Noms de colonnes fréquents dans les datasets publics de CV (Kaggle, Figshare, HF...)
TEXT_CANDIDATES = ["resume_str", "resume", "resume_text", "cv", "cv_text", "text", "content"]
ID_CANDIDATES = ["id", "cv_id", "resume_id", "candidate_id"]
CATEGORY_CANDIDATES = ["category", "categorie", "catégorie", "label", "job_category", "domain"]

CORPUS_COLUMNS = ["cv_id", "text", "category", "source"]


def _find_col(df: pd.DataFrame, wanted: Optional[str], candidates: list[str]) -> Optional[str]:
    if wanted:
        if wanted not in df.columns:
            raise KeyError(f"Colonne '{wanted}' absente. Colonnes : {list(df.columns)}")
        return wanted
    lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _read_file_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from .pdf_extractor import extract_pdf_text

        return extract_pdf_text(path, min_chars=1)
    if suffix == ".docx":
        try:
            import docx  # python-docx (optionnel)
        except ImportError as exc:
            raise ImportError("Pour les .docx : pip install python-docx") from exc
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Format non supporté : {suffix}")


def read_cv_folder(folder: Path, recursive: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Lit tous les CV d'un dossier. La catégorie = nom du sous-dossier (ex. cvs/INFORMATIQUE/x.pdf)."""
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in folder.glob(pattern) if p.suffix.lower() in {".pdf", ".txt", ".docx"})
    rows, errors = [], []
    for p in files:
        try:
            text = _read_file_text(p)
        except Exception as exc:  # noqa: BLE001 - un CV illisible ne doit pas bloquer les 5000 autres
            errors.append(f"{p.name}: {exc}")
            continue
        category = p.parent.name if p.parent != folder else ""
        rows.append({"cv_id": p.stem, "text": text, "category": category, "source": str(p)})
    return pd.DataFrame(rows, columns=CORPUS_COLUMNS), errors


def read_cv_table(
    path: Path,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    category_col: Optional[str] = None,
) -> pd.DataFrame:
    """Lit un CSV / Parquet / Excel de CV (une ligne = un CV)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")  # détecte , ou ; ou \t
    tcol = _find_col(df, text_col, TEXT_CANDIDATES)
    if tcol is None:
        raise KeyError(f"Colonne texte introuvable, utilisez --text-col. Colonnes : {list(df.columns)}")
    icol = _find_col(df, id_col, ID_CANDIDATES)
    ccol = _find_col(df, category_col, CATEGORY_CANDIDATES)
    return pd.DataFrame(
        {
            "cv_id": df[icol].astype(str) if icol else [f"cv_{i:05d}" for i in range(len(df))],
            "text": df[tcol],
            "category": df[ccol].fillna("").astype(str) if ccol else "",
            "source": path.name,
        }
    )[CORPUS_COLUMNS]


def normalize_corpus(df: pd.DataFrame, min_chars: int = 200) -> tuple[pd.DataFrame, dict]:
    """Masque les PII, retire les CV vides/trop courts et les doublons exacts. Retourne (df, stats)."""
    stats = {"n_input": int(len(df))}
    out = df.copy()
    out["text"] = out["text"].map(lambda t: t if isinstance(t, str) else "")
    out["text"] = out["text"].map(mask_pii).map(lambda t: re.sub(r"[ \t]+", " ", t).strip())
    out = out[out["text"].str.len() >= min_chars]
    stats["n_too_short"] = stats["n_input"] - int(len(out))
    sig = out["text"].map(lambda t: hashlib.md5(normalize_spaces(fold(t)).encode()).hexdigest())
    dup = sig.duplicated()
    stats["n_duplicates"] = int(dup.sum())
    out = out[~dup]
    # identifiants uniques (deux fichiers "cv.pdf" dans deux sous-dossiers, par ex.)
    if not out["cv_id"].is_unique:
        out["cv_id"] = out["cv_id"] + "_" + out.groupby("cv_id").cumcount().astype(str)
    out["cv_id"] = out["cv_id"].astype(str)
    stats["n_kept"] = int(len(out))
    return out.reset_index(drop=True), stats


def sample_for_annotation(
    corpus: pd.DataFrame, n: int, seed: int = 42
) -> pd.DataFrame:
    """Échantillon stratifié par catégorie (si disponible) pour l'annotation manuelle."""
    n = min(n, len(corpus))
    if corpus["category"].astype(str).str.len().gt(0).any():
        frac = n / len(corpus)
        sample = pd.concat(
            g.sample(n=max(1, round(len(g) * frac)), random_state=seed)
            for _, g in corpus.groupby("category")
        )
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=seed)
        return sample.reset_index(drop=True)
    return corpus.sample(n=n, random_state=seed).reset_index(drop=True)
