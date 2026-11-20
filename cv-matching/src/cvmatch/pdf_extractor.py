"""Extraction du texte d'un CV PDF (PyMuPDF)."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union


class PdfTextError(Exception):
    """PDF illisible, vide ou scanné (aucun texte extractible)."""


def extract_pdf_text(file: Union[str, Path, bytes, BinaryIO], min_chars: int = 50) -> str:
    import pymupdf as fitz  # PyMuPDF (l'ancien nom `fitz` est déprécié)

    try:
        if isinstance(file, (str, Path)):
            doc = fitz.open(str(file))
        else:
            data = bytes(file) if isinstance(file, (bytes, bytearray)) else file.read()
            doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # fichier corrompu / pas un PDF
        raise PdfTextError(f"PDF illisible : {exc}") from exc

    try:
        text = "\n".join(page.get_text("text") for page in doc).strip()
    finally:
        doc.close()

    if len(text) < min_chars:
        raise PdfTextError(
            "Aucun texte extractible (PDF scanné ou vide). Utilisez un PDF numérique."
        )
    return text
