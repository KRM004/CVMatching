"""Ranker non-LLM : Sentence-BERT (embeddings d'offres pré-calculés) + similarité cosinus."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .base import Ranked, top_k_from_scores

Encoder = Callable[[list[str]], np.ndarray]

#: Nom spécial : encodeur déterministe sans téléchargement (tests / pipeline hors-ligne).
HASHING_MODEL = "hashing"


def chunk_words(text: str, size: int = 110, max_chunks: int = 8) -> list[str]:
    """Découpe un long CV en blocs (le modèle tronque à ~128 tokens), moyennés ensuite."""
    words = text.split()
    if not words:
        return [""]
    chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
    return chunks[:max_chunks]


def hashing_encoder(n_features: int = 512) -> Encoder:
    """Encodeur factice (HashingVectorizer) : valide la chaîne SBERT sans réseau.
    Ne JAMAIS l'utiliser pour rapporter des résultats."""
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")

    def encode(texts: list[str]) -> np.ndarray:
        return hv.transform(list(texts)).toarray().astype(np.float32)

    return encode


def load_sentence_transformer(model_name: str, max_seq_length: int = 256) -> Encoder:
    if model_name == HASHING_MODEL:
        return hashing_encoder()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False),
            dtype=np.float32,
        )

    return encode


class SbertRanker:
    name = "sbert"

    def __init__(self, embeddings: np.ndarray, encoder: Encoder):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = (embeddings / np.clip(norms, 1e-9, None)).astype(np.float32)
        self.encoder = encoder

    @classmethod
    def from_model(cls, embeddings: np.ndarray, model_name: str) -> "SbertRanker":
        return cls(embeddings, load_sentence_transformer(model_name))

    def encode_resume(self, resume_text: str) -> np.ndarray:
        vecs = np.asarray(self.encoder(chunk_words(resume_text)), dtype=np.float32)
        v = vecs.mean(axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def rank(
        self, resume_text: str, top_k: int = 3, candidates: Optional[Sequence[int]] = None
    ) -> list[Ranked]:
        scores = self.embeddings @ self.encode_resume(resume_text)
        return top_k_from_scores(scores, top_k, candidates)
