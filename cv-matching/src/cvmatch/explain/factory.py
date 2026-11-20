"""Construction des explainers à partir de la configuration."""
from __future__ import annotations

from typing import Optional

from ..config import Settings
from .gemini import GeminiExplainer
from .rules import RuleExplainer

EXPLAINERS = ("rules", "gemini")


def build_explainers(settings: Settings, job_context: Optional[bool] = None,
                     cache: bool = True) -> dict:
    """{"rules": L0, "gemini": L1 (ou A2 si job_context=False)}. Sans clé, Gemini se replie
    proprement sur les règles (source = "rules_fallback")."""
    ctx = settings.job_context if job_context is None else job_context
    return {
        "rules": RuleExplainer(settings.language),
        "gemini": GeminiExplainer(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            language=settings.language,
            include_job_description=ctx,
            cache_dir=(settings.cache_dir / "gemini") if cache else None,
            min_interval=settings.gemini_min_interval,
        ),
    }
