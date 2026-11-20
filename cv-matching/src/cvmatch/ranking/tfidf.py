"""Baseline B0 : TF-IDF + similarité cosinus."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..cleaning import TOKEN_PATTERN
from .base import Ranked, top_k_from_scores


class TfidfRanker:
    name = "tfidf"

    def __init__(self, job_texts: Sequence[str], min_df: int = 2, max_df: float = 0.95):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=min_df if len(job_texts) >= 50 else 1,
            max_df=max_df if len(job_texts) >= 50 else 1.0,
        )
        self.matrix = self.vectorizer.fit_transform(list(job_texts))  # lignes L2-normalisées

    def scores(self, resume_text: str):
        q = self.vectorizer.transform([resume_text])
        return (self.matrix @ q.T).toarray().ravel()  # cosinus (vecteurs normalisés)

    def score_many(self, resume_texts: Sequence[str]) -> np.ndarray:
        """Scores de plusieurs CV d'un coup : matrice (n_cv, n_offres)."""
        q = self.vectorizer.transform(list(resume_texts))
        return (q @ self.matrix.T).toarray().astype(np.float32)

    def rank(
        self, resume_text: str, top_k: int = 3, candidates: Optional[Sequence[int]] = None
    ) -> list[Ranked]:
        return top_k_from_scores(self.scores(resume_text), top_k, candidates)
