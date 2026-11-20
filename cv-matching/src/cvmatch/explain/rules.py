"""L0 : explication à règles (sans LLM) fondée sur le dictionnaire de compétences."""
from __future__ import annotations

from typing import Optional

from ..preprocessing.skills import extract_skills, find_evidence
from ..schemas import Explanation

_TXT = {
    "fr": {
        "match": "Votre profil correspond à « {title} » grâce à : {skills}.",
        "missing": " Compétences demandées à développer : {skills}.",
        "none": "Peu de compétences explicites en commun avec « {title} » : "
        "la proximité est surtout textuelle/sémantique.",
    },
    "en": {
        "match": 'Your profile matches "{title}" through: {skills}.',
        "missing": " Skills requested that you could develop: {skills}.",
        "none": 'Few explicit skills in common with "{title}": the match is mostly textual/semantic.',
    },
}


class RuleExplainer:
    name = "rules"

    def __init__(self, language: str = "fr", max_missing: int = 6):
        self.lang = language if language in _TXT else "fr"
        self.max_missing = max_missing

    def explain(
        self, resume_text: str, job: dict, resume_skills: Optional[list[str]] = None
    ) -> Explanation:
        title = job.get("job_title", "")
        job_text = f"{title}. {job.get('job_description', '')}"
        cv_sk = resume_skills if resume_skills is not None else extract_skills(resume_text)
        job_sk = extract_skills(job_text)
        matching = [s for s in job_sk if s in cv_sk]
        missing = [s for s in job_sk if s not in cv_sk][: self.max_missing]

        t = _TXT[self.lang]
        if matching:
            text = t["match"].format(title=title, skills=", ".join(matching))
            if missing:
                text += t["missing"].format(skills=", ".join(missing))
        else:
            text = t["none"].format(title=title)

        evidence = []
        for s in matching[:3]:
            cv_ev, job_ev = find_evidence(resume_text, s), find_evidence(job_text, s)
            if cv_ev or job_ev:
                evidence.append(f"[{s}] CV: « {cv_ev} » | Offre: « {job_ev} »")
        return Explanation(
            matching_skills=matching,
            missing_skills=missing,
            explanation=text,
            evidence=evidence,
            source="rules",
        )
