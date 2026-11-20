import pytest

fitz = pytest.importorskip("pymupdf")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from cvmatch.data.synthetic import make_demo_cvs  # noqa: E402
from cvmatch.preprocessing.pdf import PdfTextError, extract_pdf_text  # noqa: E402
from cvmatch.utils.io import write_table  # noqa: E402

CV_TEXT = make_demo_cvs()["cv_01"]["text"]


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_textbox(fitz.Rect(40, 40, 550, 800), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_text_and_scanned_pdf():
    assert "Python" in extract_pdf_text(make_pdf(CV_TEXT))
    with pytest.raises(PdfTextError):
        extract_pdf_text(make_pdf(""))  # page sans texte = PDF "scanné"
    with pytest.raises(PdfTextError):
        extract_pdf_text(b"pas un pdf")


@pytest.fixture
def client(jobs, tmp_path, monkeypatch):
    path = tmp_path / "jobs.csv"
    write_table(jobs, path)
    for k, v in {"JOBS_PATH": str(path), "RANKER": "tfidf", "EXPLAINER": "rules",
                 "GEMINI_API_KEY": "", "CACHE_DIR": str(tmp_path / "cache")}.items():  # fmt: skip
        monkeypatch.setenv(k, v)
    from cvmatch.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_and_match_pdf(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["n_jobs"] > 0
    r = client.post("/match", files={"file": ("cv.pdf", make_pdf(CV_TEXT), "application/pdf")},
                    data={"top_k": 3, "explain": "true", "explainer": "rules"})  # fmt: skip
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 3 and body["resume_skills"]
    assert body["recommendations"][0]["explanation"]["source"] == "rules"


def test_match_errors(client):
    bad = client.post("/match", files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert bad.status_code == 400
    scanned = client.post("/match", files={"file": ("x.pdf", make_pdf(""), "application/pdf")})
    assert scanned.status_code == 422
    assert client.post("/match_text", json={"resume_text": "   "}).status_code == 422
    assert client.post("/match_text", json={"resume_text": CV_TEXT, "explainer": "foo"}).status_code == 422
