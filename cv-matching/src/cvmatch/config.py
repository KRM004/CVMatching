"""Configuration centralisée (variables d'environnement / .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def find_root() -> Path:
    """Racine du projet : $CVMATCH_HOME, sinon dossier parent contenant pyproject.toml."""
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


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "on", "yes"}


def _path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


@dataclass
class Settings:
    ranker: str = field(default_factory=lambda: os.getenv("RANKER", "sbert").strip().lower())
    cleaning: bool = field(default_factory=lambda: _flag("CLEANING", "on"))
    explainer: str = field(default_factory=lambda: os.getenv("EXPLAINER", "gemini").strip().lower())
    job_context: bool = field(default_factory=lambda: _flag("JOB_CONTEXT", "on"))
    language: str = field(default_factory=lambda: os.getenv("LANGUAGE", "fr").strip().lower())
    sbert_model: str = field(
        default_factory=lambda: os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_min_interval: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_MIN_INTERVAL", "1.0"))
    )
    top_k: int = 3
    root: Path = ROOT
    jobs_path: Path = field(
        default_factory=lambda: _path("JOBS_PATH", ROOT / "data" / "processed" / "jobs_clean.parquet")
    )
    artifacts_dir: Path = field(default_factory=lambda: _path("ARTIFACTS_DIR", ROOT / "artifacts"))
    cache_dir: Path = field(default_factory=lambda: _path("CACHE_DIR", ROOT / "data" / "cache"))
    eval_dir: Path = field(default_factory=lambda: ROOT / "data" / "eval")
    results_dir: Path = field(default_factory=lambda: ROOT / "results")


def get_settings() -> Settings:
    """Relit l'environnement à chaque appel (pratique pour les tests et les ablations)."""
    return Settings()
