"""Contrat de données partagé (Membre 1 -> Membre 2 -> API -> UI)."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

JobId = Union[int, str]


class Explanation(BaseModel):
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: list[str] = Field(default_factory=list)
    source: Literal["rules", "gemini", "rules_fallback"] = "rules"
    # Contrôle de fidélité : éléments retirés car non retrouvés dans le CV / l'offre
    unverified_skills: list[str] = Field(default_factory=list)
    guard_removed: int = 0
    error: Optional[str] = None


class Recommendation(BaseModel):
    job_id: JobId
    job_title: str
    job_description: str
    similarity_score: float  # proximité textuelle/sémantique, PAS une probabilité d'embauche
    company: Optional[str] = None
    city: Optional[str] = None
    sector: Optional[str] = None
    explanation: Optional[Explanation] = None


class MatchResponse(BaseModel):
    resume_skills: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    ranker: str = ""
    explainer: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MatchTextRequest(BaseModel):
    resume_text: str
    top_k: int = 3
    explain: bool = True
    explainer: Optional[str] = None  # "rules" | "gemini" | None (= défaut serveur)
