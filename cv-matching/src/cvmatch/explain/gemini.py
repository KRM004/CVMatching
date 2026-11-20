"""L1 / A2 : explications Gemini, sortie JSON, cache, retry, garde-fou de fidélité."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from ..preprocessing.cleaning import fold, normalize_spaces
from ..schemas import Explanation
from ..utils.cache import JsonCache, make_key
from .prompts import PROMPT_VERSION, build_prompt
from .rules import RuleExplainer


class GeminiUnavailable(RuntimeError):
    pass


def _contains(haystack_folded: str, needle: str) -> bool:
    n = normalize_spaces(fold(needle))
    return bool(n) and n in haystack_folded


def _parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Réponse Gemini sans JSON")
    return json.loads(t[start : end + 1])


def _str_list(v: object, limit: int = 12) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:limit]


def verify_against_sources(
    data: dict, cv_text: str, job_text: Optional[str]
) -> tuple[list[str], list[str], list[str], list[str], int]:
    """Garde-fou : retire ce qui n'est pas retrouvé dans les sources.
    Retourne (matching, missing, evidence, unverified, nb_retirés)."""
    cv_f = normalize_spaces(fold(cv_text))
    job_f = normalize_spaces(fold(job_text)) if job_text else None
    matching, unverified, removed = [], [], 0
    for s in _str_list(data.get("matching_skills")):
        in_cv = _contains(cv_f, s)
        in_job = True if job_f is None else _contains(job_f, s)
        if in_cv and in_job:
            matching.append(s)
        else:
            unverified.append(s)
            removed += 1
    missing = []
    for s in _str_list(data.get("missing_skills")):
        if _contains(cv_f, s):  # contradiction : la compétence "manquante" est dans le CV
            unverified.append(s)
            removed += 1
        else:
            missing.append(s)
    evidence = []
    for e in _str_list(data.get("evidence"), limit=5):
        core = re.sub(r"^\[[^\]]*\]\s*", "", e).strip(" «»\"'…")
        if _contains(cv_f, core) or (job_f is not None and _contains(job_f, core)):
            evidence.append(e)
        else:
            removed += 1
    return matching, missing, evidence, unverified, removed


class GeminiExplainer:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        language: str = "fr",
        include_job_description: bool = True,
        cache_dir: Optional[Path] = None,
        min_interval: float = 1.0,
        client=None,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.include_job = include_job_description
        self.cache = JsonCache(cache_dir) if cache_dir else None
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._client = client
        self._last_call = 0.0
        self.fallback = RuleExplainer(language)

    # ---- appel API -------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise GeminiUnavailable("GEMINI_API_KEY absente (voir .env.example)")
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _config():
        cfg = {"temperature": 0.0, "response_mime_type": "application/json"}
        try:
            from google.genai import types

            return types.GenerateContentConfig(**cfg)
        except ImportError:  # client injecté (tests) : un dict de config est accepté
            return cfg

    def _call(self, prompt: str) -> str:
        client = self._get_client()
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.time()
                resp = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._config(),
                )
                if not getattr(resp, "text", None):
                    raise ValueError("Réponse Gemini vide")
                return resp.text
            except Exception as exc:  # quota / indisponibilité : on retente avec backoff
                last_exc = exc
                msg = str(exc)
                retryable = any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE"))
                if not retryable or attempt == self.max_retries - 1:
                    break
                time.sleep(2**attempt * 2)
        raise GeminiUnavailable(f"Appel Gemini échoué : {last_exc}")

    # ---- API publique ----------------------------------------------------
    def explain(
        self, resume_text: str, job: dict, resume_skills: Optional[list[str]] = None
    ) -> Explanation:
        title = job.get("job_title", "")
        desc = job.get("job_description", "")
        job_text = f"{title}. {desc}" if self.include_job else None
        prompt = build_prompt(resume_text, title, desc, self.include_job, self.language)
        key = make_key(PROMPT_VERSION, self.model, str(self.include_job), self.language, prompt)

        try:
            data = self.cache.get(key) if self.cache else None
            if data is None:
                data = _parse_json(self._call(prompt))
                if self.cache:
                    self.cache.set(key, data)
        except Exception as exc:  # pas de clé, quota, JSON invalide... : on dégrade proprement
            fb = self.fallback.explain(resume_text, job, resume_skills)
            fb.source = "rules_fallback"
            fb.error = str(exc)
            return fb

        matching, missing, evidence, unverified, removed = verify_against_sources(
            data, resume_text, job_text
        )
        return Explanation(
            matching_skills=matching,
            missing_skills=missing,
            explanation=str(data.get("explanation", "")).strip(),
            evidence=evidence,
            source="gemini",
            unverified_skills=unverified,
            guard_removed=removed,
        )
