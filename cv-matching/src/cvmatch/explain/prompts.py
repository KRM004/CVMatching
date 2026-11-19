"""Prompt Gemini anti-invention (versionné : toute modification invalide le cache)."""

PROMPT_VERSION = "v1"

_TEMPLATE = """Tu es un assistant d'orientation professionnelle. Tu compares UN CV et UNE offre d'emploi.

RÈGLES STRICTES
- Utilise UNIQUEMENT des informations écrites dans le CV et l'offre ci-dessous.
- matching_skills : compétences présentes dans le CV ET demandées par l'offre.
- missing_skills : compétences demandées par l'offre mais ABSENTES du CV.
- N'invente aucune compétence, expérience, diplôme, employeur ou durée.
- evidence : 1 à 3 courts extraits COPIÉS MOT POUR MOT du CV ou de l'offre.
- explanation : 2 à 3 phrases, en __LANG__, sans promesse d'embauche (le score n'est pas une probabilité).
- Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour ni balises Markdown :
{"matching_skills": ["..."], "missing_skills": ["..."], "explanation": "...", "evidence": ["..."]}

CV (données personnelles masquées) :
<<<
__CV__
>>>

OFFRE — titre : __TITLE__
__JOB_BLOCK__
"""


def build_prompt(cv_text: str, title: str, description: str, include_job: bool, language: str) -> str:
    lang = {"fr": "français", "en": "anglais"}.get(language, "français")
    if include_job:
        block = f"Description :\n<<<\n{description}\n>>>"
    else:
        block = "(Description de l'offre non fournie : n'utilise que le titre.)"
    return (
        _TEMPLATE.replace("__LANG__", lang)
        .replace("__CV__", cv_text)
        .replace("__TITLE__", title)
        .replace("__JOB_BLOCK__", block)
    )
