"""Ground truth : CV de test, pool d'offres à annoter, labels proxy, fusion des annotateurs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..preprocessing.cleaning import fold
from ..preprocessing.pdf import extract_pdf_text
from ..utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------- CV de test
def load_cvs(cv_dir: Path) -> dict[str, str]:
    """Charge *.txt (prioritaire) et *.pdf (PyMuPDF). Nom du fichier = cv_id."""
    cvs: dict[str, str] = {}
    files = sorted(Path(cv_dir).glob("*"))
    for p in files:
        if p.suffix.lower() == ".txt":
            cvs[p.stem] = p.read_text(encoding="utf-8")
    for p in files:
        if p.suffix.lower() == ".pdf" and p.stem not in cvs:
            cvs[p.stem] = extract_pdf_text(p)
    return cvs


def text_to_pdf(text: str, path: Path) -> bool:
    """Écrit un PDF numérique simple (pour tester l'extraction). False si PyMuPDF absent."""
    try:
        import pymupdf as fitz
    except ImportError:
        return False
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(40, 40, page.rect.width - 40, page.rect.height - 40)
    for size in (10, 9, 8):
        if page.insert_textbox(rect, text, fontsize=size, fontname="helv") >= 0:
            break
    doc.save(str(path))
    doc.close()
    return True


def write_demo_cvs(out_dir: Path, with_pdf: bool = True) -> dict[str, dict]:
    """20 CV FICTIFS (.txt + .pdf) ; retourne la méta {cv_id: {family, target_keywords}}."""
    from ..data.synthetic import make_demo_cvs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for cv_id, cv in make_demo_cvs().items():
        (out_dir / f"{cv_id}.txt").write_text(cv["text"], encoding="utf-8")
        if with_pdf:
            text_to_pdf(cv["text"], out_dir / f"{cv_id}.pdf")
        meta[cv_id] = {"target_keywords": cv["target_keywords"], "family": cv["family"]}
    return meta


# ---------------------------------------------------------------- pool à annoter
def proxy_relevance(title: str, function: str, keywords: Sequence[str]) -> int:
    """Label AUTOMATIQUE par mots-clés : smoke test uniquement, jamais pour le rapport."""
    hay = fold(f"{title} {function}")
    return int(any(re.search(r"\b" + re.escape(fold(k)), hay) for k in keywords))


def build_pool(
    jobs: pd.DataFrame,
    queries: dict[str, str],
    rankers: Sequence,
    top_n: int = 6,
    n_random: int = 4,
    seed: int = 42,
    meta: Optional[dict] = None,
) -> pd.DataFrame:
    """Par CV : ∪ top-N de chaque ranker ∪ offres aléatoires, mélangés (rang masqué)."""
    rng = np.random.default_rng(seed)
    meta = meta or {}
    rows = []
    for cv_id, q in queries.items():
        idx: list[int] = []
        for r in rankers:
            idx += [x.index for x in r.rank(q, top_k=top_n)]
        idx += rng.choice(len(jobs), size=min(n_random, len(jobs)), replace=False).tolist()
        kws = meta.get(cv_id, {}).get("target_keywords", [])
        for i in dict.fromkeys(idx):  # unique, ordre conservé
            row = jobs.iloc[i]
            rows.append(
                {
                    "cv_id": cv_id,
                    "job_id": row["job_id"],
                    "job_title": row["job_title"],
                    "company": row.get("company", ""),
                    "job_snippet": re.sub(r"\s+", " ", row["job_description"])[:300],
                    "annotator_1": "",
                    "annotator_2": "",
                    "proxy_relevant": proxy_relevance(row["job_title"], row.get("function", ""), kws)
                    if kws else "",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("cv_id", group_keys=False).sample(frac=1, random_state=seed)


# ---------------------------------------------------------------- labels
def load_labels(path: Path, column: str = "relevant") -> dict[str, dict]:
    """labels.csv : cv_id, job_id, relevant(0/1) -> {cv_id: {job_id: 0/1}}."""
    df = pd.read_csv(path)
    df = df[df[column].notna() & (df[column].astype(str).str.strip() != "")]
    out: dict[str, dict] = {}
    for cv_id, job_id, rel in zip(df["cv_id"].astype(str), df["job_id"], df[column]):
        out.setdefault(cv_id, {})[job_id.item() if hasattr(job_id, "item") else job_id] = int(rel)
    return out


def has_annotations(pool_csv: Path) -> bool:
    if not Path(pool_csv).exists():
        return False
    df = pd.read_csv(pool_csv, dtype=str)
    if not {"annotator_1", "annotator_2"} <= set(df.columns):
        return False
    return bool((df["annotator_1"].notna() & df["annotator_2"].notna()).any())


def finalize_labels(
    pool_csv: Path, out_csv: Path, disagreements_csv: Optional[Path] = None
) -> dict:
    """Fusion des 2 annotateurs : pertinent = les deux disent 1. Rapporte accord et kappa."""
    from sklearn.metrics import cohen_kappa_score

    df = pd.read_csv(pool_csv)
    df = df[df["annotator_1"].notna() & df["annotator_2"].notna()].copy()
    if df.empty:
        raise ValueError("Aucune ligne annotée par les DEUX annotateurs.")
    a1, a2 = df["annotator_1"].astype(int), df["annotator_2"].astype(int)
    df["relevant"] = ((a1 == 1) & (a2 == 1)).astype(int)
    dis = df[a1 != a2]
    if disagreements_csv is not None:
        dis.to_csv(disagreements_csv, index=False)
    df[["cv_id", "job_id", "relevant"]].to_csv(out_csv, index=False)
    kappa = float(cohen_kappa_score(a1, a2)) if a1.nunique() > 1 or a2.nunique() > 1 else float("nan")
    return {
        "n_pairs": int(len(df)),
        "n_relevant": int(df["relevant"].sum()),
        "agreement": float((a1 == a2).mean()),
        "cohen_kappa": kappa,
        "n_disagreements": int(len(dis)),
    }


def write_proxy_labels(pool: pd.DataFrame, out_csv: Path) -> int:
    px = pool[pool["proxy_relevant"].astype(str).str.len() > 0]
    px = px[["cv_id", "job_id", "proxy_relevant"]].rename(columns={"proxy_relevant": "relevant"})
    px.to_csv(out_csv, index=False)
    return len(px)
