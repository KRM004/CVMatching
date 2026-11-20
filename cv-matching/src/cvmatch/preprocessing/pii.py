"""Masquage des données personnelles avant tout log / appel à un service externe (Gemini)."""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Téléphones : marocain international (+212 / 00212), marocain local (05-08...), autres +CC
_PHONES = [
    re.compile(r"(?<![\w+])(?:\+|00)\s?212[\s.\-]?\(?0?\)?[\s.\-]?[5-8](?:[\s.\-]?\d){8}(?!\d)"),
    re.compile(r"(?<![\w+])0[5-8](?:[\s.\-]?\d){8}(?!\d)"),
    re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.\-]?\d){7,12}(?!\d)"),
]
_URL_PERSO = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com|facebook\.com|instagram\.com)/\S+",
    re.IGNORECASE,
)
_ADDR_LINE = re.compile(r"^\s*(?:adresse|address|domicile)\s*:.*$", re.IGNORECASE | re.MULTILINE)


def mask_pii(text: str) -> str:
    if not text:
        return ""
    t = _EMAIL.sub("[EMAIL]", text)
    t = _URL_PERSO.sub("[URL]", t)
    t = _ADDR_LINE.sub("Adresse : [ADRESSE]", t)
    for pat in _PHONES:
        t = pat.sub("[TELEPHONE]", t)
    return t
