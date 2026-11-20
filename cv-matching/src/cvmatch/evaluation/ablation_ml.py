"""Ablation ML : B0 TF-IDF | F1 Sentence-BERT + cleaning | A1 SBERT sans cleaning.

Mêmes CV, même pool d'offres annotées : un SEUL facteur change à la fois."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import Settings
from ..preprocessing import prepare_resume
from ..ranking import build_ranker, load_sentence_transformer
from ..utils.logging import get_logger
from .metrics import delta, evaluate_ranker, summarize

log = get_logger(__name__)


def ml_configs(settings: Settings, with_sbert: bool = True) -> dict:
    """{nom: (ranker_kwargs, cleaning)} — la table d'ablation, en un seul endroit."""
    runs = {"B0_tfidf": ({"name": "tfidf", "cleaning": True}, True)}
    if with_sbert:
        runs["F1_sbert_clean"] = ({"name": "sbert", "cleaning": True}, True)
        runs["A1_sbert_noclean"] = ({"name": "sbert", "cleaning": False}, False)
    return runs


def run_ml_ablation(
    jobs: pd.DataFrame,
    cvs: dict[str, str],
    labels: dict[str, dict],
    settings: Settings,
    with_sbert: bool = True,
    k: int = 3,
    encoder=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (résultats par CV, résumé avec deltas vs F1)."""
    if with_sbert and encoder is None:
        encoder = load_sentence_transformer(settings.sbert_model)
    per_cv, summaries = [], {}
    for name, (kw, cleaning) in ml_configs(settings, with_sbert).items():
        ranker = build_ranker(jobs, settings, encoder=encoder, **kw)
        ranker.rank("warm up", top_k=1)  # le chargement initial n'entre pas dans la latence
        df = evaluate_ranker(ranker, jobs, cvs, labels,
                             preprocess=lambda t, c=cleaning: prepare_resume(t, c)[1], k=k)
        df.insert(0, "config", name)
        per_cv.append(df)
        summaries[name] = summarize(df, name)
        log.info("%-18s P@%d = %.3f (max %.3f) | latence médiane %.1f ms", name, k,
                 summaries[name]["mean_p_at_k"], summaries[name]["mean_p_at_k_max"],
                 summaries[name]["median_latency_ms"])
    summary = pd.DataFrame(summaries.values())
    ref: Optional[float] = summaries.get("F1_sbert_clean", {}).get("mean_p_at_k")
    if ref is not None:
        summary["delta_vs_F1"] = [delta(ref, v["mean_p_at_k"]) for v in summaries.values()]
    return pd.concat(per_cv, ignore_index=True), summary
