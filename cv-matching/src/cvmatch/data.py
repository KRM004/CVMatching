"""Chargement / nettoyage du catalogue d'offres (Rekrute Maroc, Hugging Face)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .cleaning import clean_text, fold

RAW_REPO = "adlbh/rekrute-2005-2022"

# Colonnes du dataset (fiche HF). Les noms exacts doivent être vérifiés après téléchargement.
COL_ID = "job_offer_id"
COL_TITLE = "job_title"
COL_CITY = "city"
COL_SECTOR = "sector"
COL_FUNCTION = "function"
COL_COMPANY = "Entreprise"
COL_EDU = "Niveau d'étude et formation"
COL_EXP = "Expérience requise"
COL_YEAR = "pub_date_year"
DESCRIPTION_COLS = [
    "Poste",
    "Profil recherché",
    "Qualités requises pour réussir dans ce rôle",
    "soft_skills",
]

CLEAN_COLUMNS = [
    "job_id", "job_title", "job_description", "company", "city", "sector",
    "function", "education", "experience", "year",
]  # fmt: skip


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    raise ValueError(f"Format non supporté : {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_raw_jobs(path: Optional[Path] = None) -> pd.DataFrame:
    """Charge le brut depuis un fichier local, sinon depuis Hugging Face (`datasets`)."""
    if path:
        return read_table(Path(path))
    from datasets import load_dataset

    return load_dataset(RAW_REPO, split="train").to_pandas()


def _s(v: object) -> str:
    return v.strip() if isinstance(v, str) else ""


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Colonne si elle existe, sinon série vide (le schéma réel peut différer de la fiche)."""
    return df[name] if name in df.columns else pd.Series("", index=df.index)


def build_description(row: pd.Series) -> str:
    parts = [_s(row.get(c)) for c in DESCRIPTION_COLS]
    return "\n".join(p for p in parts if p)


def prepare_jobs(
    raw: pd.DataFrame,
    min_year: Optional[int] = 2019,
    max_rows: Optional[int] = 15000,
    sectors: Optional[Iterable[str]] = None,
    min_desc_chars: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Nettoie le brut : ID stable, titre + description non vides, sans doublons exacts."""
    df = raw.copy()
    if COL_YEAR in df.columns and min_year:
        df = df[pd.to_numeric(df[COL_YEAR], errors="coerce").fillna(0) >= min_year]
    if sectors:
        keys = [fold(s) for s in sectors]
        hay = (_col(df, COL_SECTOR).fillna("").astype(str) + " " +
               _col(df, COL_FUNCTION).fillna("").astype(str) + " " +
               _col(df, COL_TITLE).fillna("").astype(str)).map(fold)  # fmt: skip
        df = df[hay.map(lambda h: any(k in h for k in keys))]

    out = pd.DataFrame(
        {
            "job_id": df[COL_ID],
            "job_title": _col(df, COL_TITLE).map(_s),
            "job_description": df.apply(build_description, axis=1),
            "company": _col(df, COL_COMPANY).map(_s),
            "city": _col(df, COL_CITY).map(_s),
            "sector": _col(df, COL_SECTOR).map(_s),
            "function": _col(df, COL_FUNCTION).map(_s),
            "education": _col(df, COL_EDU).map(_s),
            "experience": _col(df, COL_EXP).map(_s),
            "year": pd.to_numeric(_col(df, COL_YEAR), errors="coerce"),
        }
    )
    # description : HTML/espaces nettoyés dès l'import (le texte "raw" garde le reste intact)
    out = out[out["job_id"].notna() & (out["job_title"] != "")]
    out = out[out["job_description"].str.len() >= min_desc_chars]
    # doublons exacts (même titre + même description normalisée)
    sig = (out["job_title"] + "||" + out["job_description"]).map(
        lambda t: hashlib.md5(re.sub(r"\s+", " ", fold(t)).encode("utf-8")).hexdigest()
    )
    out = out[~sig.duplicated()]
    out = out.drop_duplicates(subset="job_id")
    if max_rows and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    out = out.sort_values("job_id").reset_index(drop=True)
    out["year"] = out["year"].fillna(0).astype(int)
    return out[CLEAN_COLUMNS]


def validate_jobs(jobs: pd.DataFrame) -> None:
    """Test de validation M1.2 : aucune offre sans ID ni description."""
    assert jobs["job_id"].notna().all(), "offre sans ID"
    assert jobs["job_id"].is_unique, "ID dupliqué"
    assert (jobs["job_description"].str.len() > 0).all(), "offre sans description"


def job_texts(jobs: pd.DataFrame, cleaning: bool = True) -> list[str]:
    """Texte indexé pour chaque offre : titre + description (nettoyés ou bruts)."""
    f = clean_text if cleaning else (lambda x: x if isinstance(x, str) else "")
    return [f(f"{t}. {d}") for t, d in zip(jobs["job_title"], jobs["job_description"])]


def load_jobs(path: Path) -> pd.DataFrame:
    jobs = read_table(path)
    validate_jobs(jobs)
    return jobs.reset_index(drop=True)
