"""Interface commune des rankers (non-LLM) : TF-IDF (baseline) et Sentence-BERT."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from ..cleaning import fold


@dataclass
class Ranked:
    index: int  # position de l'offre dans le DataFrame des offres
    score: float  # similarité cosinus (proximité textuelle/sémantique)


class Ranker(Protocol):
    name: str

    def rank(
        self, resume_text: str, top_k: int = 3, candidates: Optional[Sequence[int]] = None
    ) -> list[Ranked]: ...

    def score_many(self, resume_texts: Sequence[str]) -> np.ndarray: ...  # (n_cv, n_offres)


def top_k_from_scores(
    scores: np.ndarray, top_k: int, candidates: Optional[Sequence[int]] = None
) -> list[Ranked]:
    """Top-k par score décroissant ; `candidates` restreint le classement à un sous-ensemble fixe
    (utilisé par l'évaluation : même pool d'offres pour toutes les configurations)."""
    if candidates is not None:
        cand = np.asarray(list(candidates), dtype=int)
        if cand.size == 0:
            return []
        sub = scores[cand]
        order = np.argsort(-sub, kind="stable")[:top_k]
        return [Ranked(int(cand[i]), float(sub[i])) for i in order]
    k = min(top_k, scores.shape[0])
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return [Ranked(int(i), float(scores[i])) for i in order]


_TITLE_NOISE = re.compile(r"\((?:h|m|f|h/f|f/h|m/f|f/m)\)|\d+|[-–|/].*$", re.IGNORECASE)
_SENIORITY = re.compile(
    r"\b(?:junior|senior|confirme|experimente|debutant|stagiaire|stage|sr|jr|h/f|f/h)\b"
)


def title_key(title: str) -> str:
    t = _TITLE_NOISE.sub(" ", fold(title or ""))
    return re.sub(r"\s+", " ", _SENIORITY.sub(" ", t)).strip()


def diversify(ranked: list[Ranked], titles: Sequence[str], k: int) -> list[Ranked]:
    """Évite un Top-k de quasi-doublons (même intitulé publié plusieurs fois)."""
    seen: set[str] = set()
    picked: list[Ranked] = []
    rest: list[Ranked] = []
    for r in ranked:
        key = title_key(titles[r.index])
        if key in seen:
            rest.append(r)
        else:
            seen.add(key)
            picked.append(r)
    return (picked + rest)[:k]
