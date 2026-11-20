"""Prétraitement du texte : extraction PDF, masquage PII, nettoyage, compétences."""
from .cleaning import clean_text, fold, raw_text
from .pii import mask_pii
from .skills import extract_skills, find_evidence


def prepare_resume(text: str, cleaning: bool = True) -> tuple[str, str]:
    """CV brut -> (texte masqué [envoyable à un LLM], requête de classement)."""
    masked = mask_pii(text)
    return masked, clean_text(masked) if cleaning else raw_text(masked)


__all__ = ["clean_text", "raw_text", "fold", "mask_pii", "extract_skills", "find_evidence",
           "prepare_resume"]
