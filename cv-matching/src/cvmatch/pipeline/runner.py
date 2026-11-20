"""Orchestrateur : exécute les étapes dans l'ordre, saute celles déjà faites, trace tout."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, Sequence

from ..config import Settings, get_settings
from ..utils.io import write_json
from ..utils.logging import get_logger
from .config import PipelineConfig
from .steps import STEP_NAMES, STEPS, Context

log = get_logger("cvmatch.pipeline")


def select_steps(only: Optional[Sequence[str]] = None, start: Optional[str] = None,
                 stop: Optional[str] = None) -> list[str]:
    for n in [*(only or []), start, stop]:
        if n and n not in STEP_NAMES:
            raise ValueError(f"Étape inconnue : {n} (choix : {', '.join(STEP_NAMES)})")
    if only:
        return [n for n in STEP_NAMES if n in only]
    i = STEP_NAMES.index(start) if start else 0
    j = STEP_NAMES.index(stop) + 1 if stop else len(STEP_NAMES)
    return STEP_NAMES[i:j]


def run_pipeline(
    cfg: Optional[PipelineConfig] = None,
    settings: Optional[Settings] = None,
    only: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    stop: Optional[str] = None,
    force: bool = False,
) -> Context:
    ctx = Context(settings or get_settings(), cfg or PipelineConfig(), force=force)
    names = select_steps(only, start, stop)
    ctx.state["steps"] = {}
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": ctx.cfg.to_dict(),
        "settings": {"sbert_model": ctx.settings.sbert_model, "language": ctx.settings.language,
                     "gemini_configured": bool(ctx.settings.gemini_api_key)},
        "steps": ctx.state["steps"],
    }
    log.info("Pipeline : %s", " → ".join(names))
    for i, step in enumerate(s for s in STEPS if s.name in names):
        rec: dict = {"status": "running"}
        ctx.state["steps"][step.name] = rec
        if step.is_done(ctx) and not force:
            rec.update(status="cached", seconds=0.0)
            log.info("[%d/%d] %-10s déjà fait (sorties présentes ; --force pour refaire)",
                     i + 1, len(names), step.name)
            continue
        log.info("[%d/%d] %-10s %s", i + 1, len(names), step.name, step.help)
        t0 = time.perf_counter()
        try:
            rec["result"] = step.run(ctx)
            rec["status"] = "ok"
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            write_json(manifest, ctx.settings.results_dir / "pipeline_run.json")
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest, ctx.settings.results_dir / "pipeline_run.json")
    log.info("Pipeline terminé ✔  (manifest : %s)", ctx.settings.results_dir / "pipeline_run.json")
    return ctx
