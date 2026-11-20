"""Journalisation homogène pour la CLI et le pipeline."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_cvmatch", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        h._cvmatch = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(level.upper())
    for noisy in ("httpx", "urllib3", "sentence_transformers", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
