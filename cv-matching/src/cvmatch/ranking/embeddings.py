"""Artefacts hors-ligne : embeddings Sentence-BERT des offres (calculés une seule fois)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.catalog import job_texts
from ..utils.logging import get_logger

log = get_logger(__name__)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def embeddings_path(artifacts_dir: Path, model_name: str, cleaning: bool) -> Path:
    return Path(artifacts_dir) / f"emb_{_slug(model_name)}_{'clean' if cleaning else 'raw'}.npy"


def build_embeddings(
    jobs: pd.DataFrame,
    model_name: str,
    artifacts_dir: Path,
    cleaning: bool = True,
    encoder=None,
    batch_size: int = 64,
) -> Path:
    """Encode toutes les offres et sauvegarde `emb_*.npy` + méta JSON (modèle, nb, ids)."""
    if encoder is None:
        from .ranking.sbert import load_sentence_transformer

        encoder = load_sentence_transformer(model_name)
    texts = job_texts(jobs, cleaning)
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encoder(texts[i : i + batch_size]), dtype=np.float32))
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 20 == 0 or done == len(texts):
            log.info("  %d/%d offres encodées (%s)", done, len(texts), "clean" if cleaning else "raw")
    emb = np.vstack(chunks)
    out = embeddings_path(artifacts_dir, model_name, cleaning)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    meta = {
        "model": model_name,
        "cleaning": cleaning,
        "n_jobs": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "first_job_id": str(jobs["job_id"].iloc[0]),
        "last_job_id": str(jobs["job_id"].iloc[-1]),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_embeddings(
    jobs: pd.DataFrame, model_name: str, artifacts_dir: Path, cleaning: bool = True
) -> np.ndarray:
    path = embeddings_path(artifacts_dir, model_name, cleaning)
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings introuvables : {path}\n"
            "Lancez d'abord :  cvmatch embed"
        )
    emb = np.load(path)
    if emb.shape[0] != len(jobs):
        raise ValueError(
            f"Embeddings ({emb.shape[0]}) et offres ({len(jobs)}) désynchronisés : "
            "relancez :  cvmatch embed --force"
        )
    return emb
