"""Configuration d'exécution (variables d'environnement / `.env`).

Deux niveaux de configuration :
- `Settings` (ce fichier)      : comportement du service (ranker, explainer, modèles, chemins).
- `PipelineConfig` (pipeline/) : paramètres des étapes hors-ligne (configs/pipeline.yaml).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def find_root() -> Path:
    """Racine du projet : $CVMATCH_HOME, sinon premier parent contenant pyproject.toml."""
    env = os.getenv("CVMATCH_HOME")
    if env:
        return Path(env).resolve()
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return cur


ROOT = find_root()
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).split("#")[0].strip()


def _flag(name: str, default: str) -> bool:
    return _env(name, default).lower() in {"1", "true", "on", "yes"}


def _path(name: str, default: Path) -> Path:
    v = _env(name, "")
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


@dataclass
class Settings:
    # --- comportement (un seul facteur change par ablation) ---
    ranker: str = field(default_factory=lambda: _env("RANKER", "sbert").lower())
    cleaning: bool = field(default_factory=lambda: _flag("CLEANING", "on"))
    explainer: str = field(default_factory=lambda: _env("EXPLAINER", "gemini").lower())
    job_context: bool = field(default_factory=lambda: _flag("JOB_CONTEXT", "on"))
    language: str = field(default_factory=lambda: _env("LANGUAGE", "fr").lower())
    top_k: int = 3
    # --- modèles ---
    sbert_model: str = field(
        default_factory=lambda: _env("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    )
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_min_interval: float = field(
        default_factory=lambda: float(_env("GEMINI_MIN_INTERVAL", "1.0"))
    )
    # --- arborescence ---
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", ROOT / "data"))
    jobs_path: Path = field(
        default_factory=lambda: _path("JOBS_PATH", ROOT / "data" / "processed" / "jobs_clean.parquet")
    )
    artifacts_dir: Path = field(default_factory=lambda: _path("ARTIFACTS_DIR", ROOT / "artifacts"))
    cache_dir: Path = field(default_factory=lambda: _path("CACHE_DIR", ROOT / "data" / "cache"))
    eval_dir: Path = field(default_factory=lambda: _path("EVAL_DIR", ROOT / "data" / "eval"))
    results_dir: Path = field(default_factory=lambda: _path("RESULTS_DIR", ROOT / "results"))

    @property
    def cvs_dir(self) -> Path:
        return self.eval_dir / "cvs"


def get_settings() -> Settings:
    """Relit l'environnement à chaque appel (pratique pour les tests et les ablations)."""
    return Settings()
