"""Construction d'un ranker à partir de la configuration (point d'entrée unique)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import Settings
from ..data.catalog import job_texts
from .base import Ranker
from .embeddings import load_embeddings
from .sbert import Encoder, SbertRanker, load_sentence_transformer
from .tfidf import TfidfRanker

RANKERS = ("tfidf", "sbert")


def build_ranker(
    jobs: pd.DataFrame,
    settings: Settings,
    encoder: Optional[Encoder] = None,
    name: Optional[str] = None,
    cleaning: Optional[bool] = None,
) -> Ranker:
    name = (name or settings.ranker).lower()
    cleaning = settings.cleaning if cleaning is None else cleaning
    if name == "tfidf":
        return TfidfRanker(job_texts(jobs, cleaning))
    if name == "sbert":
        emb = load_embeddings(jobs, settings.sbert_model, settings.artifacts_dir, cleaning)
        return SbertRanker(emb, encoder or load_sentence_transformer(settings.sbert_model))
    raise ValueError(f"RANKER inconnu : {name} ({' | '.join(RANKERS)})")
