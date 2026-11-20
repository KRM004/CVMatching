"""Étape « explication » : règles (L0) et Gemini avec garde-fou de fidélité (L1 / A2)."""
from .factory import EXPLAINERS, build_explainers
from .gemini import GeminiExplainer
from .rules import RuleExplainer

__all__ = ["RuleExplainer", "GeminiExplainer", "build_explainers", "EXPLAINERS"]
