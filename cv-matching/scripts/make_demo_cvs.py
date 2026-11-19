"""Génère 20 CV FICTIFS (txt + pdf) dans data/eval/cvs/ pour tester le pipeline.

Pour la vraie évaluation, remplacez-les (ou complétez-les) par des CV anonymisés / consentis,
par exemple un échantillon du dataset Figshare (emploi.ma) converti en .txt/.pdf."""
import json

from cvmatch.config import get_settings
from cvmatch.synthetic import make_demo_cvs


def to_pdf(text: str, path) -> bool:
    try:
        import fitz
    except ImportError:
        return False
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(40, 40, page.rect.width - 40, page.rect.height - 40)
    for size in (10, 9, 8):
        if page.insert_textbox(rect, text, fontsize=size, fontname="helv") >= 0:
            break
    doc.save(str(path))
    doc.close()
    return True


def main() -> None:
    s = get_settings()
    out = s.eval_dir / "cvs"
    out.mkdir(parents=True, exist_ok=True)
    meta, n_pdf = {}, 0
    for cv_id, cv in make_demo_cvs().items():
        (out / f"{cv_id}.txt").write_text(cv["text"], encoding="utf-8")
        n_pdf += to_pdf(cv["text"], out / f"{cv_id}.pdf")
        meta[cv_id] = {"target_keywords": cv["target_keywords"], "family": cv["family"]}
    (s.eval_dir / "cv_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✔ {len(meta)} CV fictifs (.txt) et {n_pdf} PDF -> {out}")


if __name__ == "__main__":
    main()
