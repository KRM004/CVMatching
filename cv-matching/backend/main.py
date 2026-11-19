"""API FastAPI : POST /match (PDF), POST /match_text (texte), GET /health. Docs : /docs"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from cvmatch.config import get_settings
from cvmatch.pdf_extractor import PdfTextError, extract_pdf_text
from cvmatch.schemas import MatchResponse, MatchTextRequest
from cvmatch.service import build_service

MAX_PDF_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    try:
        app.state.service = build_service(app.state.settings)
        app.state.error = None
    except Exception as exc:  # artefacts manquants, etc. : l'API démarre, /health explique
        app.state.service = None
        app.state.error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="CV Matching API",
    description="Upload d'un CV PDF -> Top 3 offres (TF-IDF / Sentence-BERT) + explications Gemini.",
    version="0.1.0",
    lifespan=lifespan,
)


def _service():
    svc = getattr(app.state, "service", None)
    if svc is None:
        raise HTTPException(503, detail=f"Service indisponible : {app.state.error}")
    return svc


def _validate_explainer(name: Optional[str]) -> Optional[str]:
    if name not in (None, "", "rules", "gemini"):
        raise HTTPException(422, detail="explainer doit valoir 'rules' ou 'gemini'")
    return name or None


def _run(text: str, top_k: int, explain: bool, explainer: Optional[str]) -> MatchResponse:
    svc = _service()
    if not 1 <= top_k <= 10:
        raise HTTPException(422, detail="top_k doit être compris entre 1 et 10")
    try:
        return svc.match(text, top_k=top_k, explain=explain, explainer=_validate_explainer(explainer))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc


@app.get("/health")
def health():
    svc = getattr(app.state, "service", None)
    s = app.state.settings
    return {
        "status": "ok" if svc is not None else "degraded",
        "ranker": s.ranker,
        "n_jobs": len(svc.jobs) if svc is not None else 0,
        "gemini_configured": bool(s.gemini_api_key),
        "error": app.state.error,
    }


@app.post("/match", response_model=MatchResponse)
def match_pdf(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    explain: bool = Form(True),
    explainer: Optional[str] = Form(None),
):
    data = file.file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, detail="PDF trop volumineux (max 5 Mo)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, detail="Le fichier n'est pas un PDF valide")
    try:
        text = extract_pdf_text(data)
    except PdfTextError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return _run(text, top_k, explain, explainer)


@app.post("/match_text", response_model=MatchResponse)
def match_text(req: MatchTextRequest):
    return _run(req.resume_text, req.top_k, req.explain, req.explainer)
