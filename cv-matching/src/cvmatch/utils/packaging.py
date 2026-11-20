"""Archive .zip propre du projet (code + config + docs, sans données ni secrets)."""
from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
                ".ipynb_checkpoints", ".mypy_cache", "dist", "build", ".hf_cache"}
EXCLUDE_FILES = [".env", "*.pyc", "*.egg-info", ".DS_Store"]
# dossiers générés : seuls les .gitkeep sont gardés (structure conservée)
GENERATED = ["data/raw", "data/processed", "data/cache", "data/eval", "artifacts", "results"]


def _skip(rel: Path, with_results: bool) -> bool:
    if any(part in EXCLUDE_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return True
    if any(fnmatch.fnmatch(rel.name, pat) for pat in EXCLUDE_FILES):
        return True
    posix = rel.as_posix()
    for g in GENERATED:
        if posix.startswith(g + "/") and rel.name != ".gitkeep":
            if with_results and g == "results":
                return False
            return True
    return False


def make_zip(root: Path, out: Path, with_results: bool = False) -> tuple[Path, int]:
    root, out = Path(root).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = root.name
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p == out:
                continue
            rel = p.relative_to(root)
            if _skip(rel, with_results):
                continue
            zf.write(p, f"{prefix}/{rel.as_posix()}")
            n += 1
    return out, n
