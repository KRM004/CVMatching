"""Nettoyage de texte qui PRÉSERVE les termes techniques (C++, C#, .NET, Node.js, SQL...)."""
from __future__ import annotations

import html
import re
import unicodedata

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_BULLETS = re.compile(r"[•●▪■□◦·►➢✓✔★☆→]")
# On garde lettres (y compris arabes/accentuées), chiffres et ponctuation utile aux termes techniques.
_NOISE = re.compile(r"[^\w\s.,;:'’/&%()+#-]")
_SPACES = re.compile(r"\s+")

# Tokenisation TF-IDF qui conserve c++, c#, f#, .net, node.js, asp.net, etc.
TOKEN_PATTERN = r"(?u)[^\W_]+(?:\+\+|#)|\.net\b|[^\W_]+(?:\.(?:js|net))?"


def clean_text(text: object) -> str:
    """Nettoyage 'complet' : HTML, URL, puces, caractères parasites, espaces."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _BULLETS.sub(" ", t)
    t = _NOISE.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def raw_text(text: object) -> str:
    """Version 'sans cleaning' (ablation A1) : le texte tel quel, sans aucune transformation."""
    return text if isinstance(text, str) else ""


def fold(text: str) -> str:
    """Minuscules + suppression des accents (pour comparer des chaînes)."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def normalize_spaces(text: str) -> str:
    return _SPACES.sub(" ", text).strip()
