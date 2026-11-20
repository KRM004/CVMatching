"""Rapport Markdown final (results/report.md) assemblé à partir des sorties du pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(aucune donnée)_"
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _read(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if Path(path).exists() else None


def build_report(results_dir: Path, run_info: dict) -> Path:
    results_dir = Path(results_dir)
    tag = "_PROXY" if run_info.get("labels") == "proxy" else ""
    ml = _read(results_dir / f"ablation_ml_summary{tag}.csv")
    llm = _read(results_dir / "ablation_llm_summary.csv")
    warn = []
    if run_info.get("labels") == "proxy":
        warn.append("Labels **PROXY** (mots-clés automatiques) : chiffres = test technique, "
                    "**à ne pas rapporter**.")
    if run_info.get("synthetic"):
        warn.append("Catalogue **SYNTHÉTIQUE** : chiffres sans valeur scientifique.")
    if run_info.get("sbert_model") == "hashing":
        warn.append("Encodeur **hashing** (factice) à la place de Sentence-BERT.")
    steps = run_info.get("steps", {})
    md = [
        "# Rapport du pipeline CV Matching",
        f"_Généré le {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        *[f"> ⚠ {w}" for w in warn],
        "",
        "## Exécution",
        _table(pd.DataFrame([{"étape": k, "statut": v.get("status"), "durée_s": v.get("seconds")}
                             for k, v in steps.items() if v.get("status") != "running"])),
        "",
        "## Ablation ML (Precision@k, pool figé)",
        _table(ml),
        "",
        "`delta_vs_F1 = P@k(F1) − P@k(config)` : F1−B0 = apport de SBERT, F1−A1 = apport du cleaning.",
        "",
        "## Ablation LLM (fidélité aux sources)",
        _table(llm),
        "",
        "Gemini est jugé sur la fidélité (affirmations non retrouvées dans le CV/l'offre), "
        "pas sur la Precision@k.",
    ]
    out = results_dir / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out
