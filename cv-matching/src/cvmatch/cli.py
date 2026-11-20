"""Interface en ligne de commande unique : `cvmatch <commande>` (ou `python -m cvmatch`).

  cvmatch pipeline                    # tout le pipeline (configs/pipeline.yaml)
  cvmatch pipeline --offline          # démo 100 % hors-ligne (offres fictives + encodeur hashing)
  cvmatch pipeline --from pool        # reprend à partir d'une étape
  cvmatch prepare | embed | cvs | pool | labels | ablate-ml | ablate-llm | report
  cvmatch match data/eval/cvs/cv_01.pdf --explainer rules
  cvmatch serve                       # API FastAPI (http://localhost:8000/docs)
  cvmatch info                        # configuration + état des fichiers
  cvmatch package                     # dist/cv-matching.zip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .pipeline import STEP_NAMES, STEPS, PipelineConfig, run_pipeline
from .utils.logging import setup_logging


def _pipeline_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None,
                   help="fichier YAML (défaut : configs/pipeline.yaml)")
    p.add_argument("--synthetic", action="store_true", help="catalogue d'offres FICTIF")
    p.add_argument("--offline", action="store_true",
                   help="--synthetic + encodeur 'hashing' : aucun accès réseau")
    p.add_argument("--tfidf-only", action="store_true", help="désactive Sentence-BERT")
    p.add_argument("--max-rows", type=int, help="surcharge data.max_rows")
    p.add_argument("--sectors", nargs="*", help="surcharge data.sectors (ex. informatique)")
    p.add_argument("--raw", type=Path, help="fichier brut local (.parquet/.csv)")
    p.add_argument("--force", action="store_true", help="refait les étapes déjà faites")


def _load(a: argparse.Namespace):
    s = get_settings()
    cfg_path = a.config or (s.root / "configs" / "pipeline.yaml")
    cfg = PipelineConfig.load(cfg_path)
    if a.synthetic or a.offline:
        cfg.data.source = "synthetic"
    if a.offline:
        s.sbert_model = "hashing"
    if a.tfidf_only:
        cfg.embeddings.enabled = False
    if a.max_rows is not None:
        cfg.data.max_rows = a.max_rows
    if a.sectors is not None:
        cfg.data.sectors = a.sectors
    if a.raw:
        cfg.data.source, cfg.data.raw_path = "local", str(a.raw.resolve())
    return s, cfg


def cmd_pipeline(a):
    s, cfg = _load(a)
    run_pipeline(cfg, s, only=a.only, start=a.start, stop=a.stop, force=a.force)


def cmd_step(a):
    if a.cmd == "ablate-llm" and getattr(a, "summarize", False):
        import pandas as pd

        from .evaluation.ablation_llm import summarize_llm

        s = get_settings()
        df = summarize_llm(pd.read_csv(s.results_dir / "ablation_llm.csv"))
        df.to_csv(s.results_dir / "ablation_llm_summary.csv", index=False)
        print(df.to_string(index=False))
        return
    s, cfg = _load(a)
    run_pipeline(cfg, s, only=[a.cmd], force=a.force)


def cmd_match(a):
    from .preprocessing.pdf import extract_pdf_text
    from .service import build_service

    s = get_settings()
    if a.ranker:
        s.ranker = a.ranker
    text = extract_pdf_text(a.file) if a.file.suffix.lower() == ".pdf" else a.file.read_text("utf-8")
    res = build_service(s).match(text, top_k=a.top_k, explainer=a.explainer)
    if a.json:
        print(res.model_dump_json(indent=2))
        return
    print(f"Compétences : {', '.join(res.resume_skills) or '—'}")
    for i, r in enumerate(res.recommendations, 1):
        print(f"\n{i}. [{r.similarity_score:.3f}] {r.job_title} — {r.company or ''} {r.city or ''}")
        if r.explanation:
            print(f"   {r.explanation.explanation}")
    for w in res.warnings:
        print(f"⚠ {w}")


def cmd_serve(a):
    import uvicorn

    uvicorn.run("cvmatch.api.app:app", host=a.host, port=a.port, reload=a.reload)


def cmd_info(a):
    from dataclasses import asdict

    s = get_settings()
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(s).items()}
    d["gemini_api_key"] = "***" if s.gemini_api_key else ""
    print(json.dumps(d, indent=2, ensure_ascii=False))
    for label, p in [("catalogue", s.jobs_path), ("CV de test", s.cvs_dir),
                     ("pool", s.eval_dir / "pool_to_annotate.csv"),
                     ("labels", s.eval_dir / "labels.csv"), ("rapport", s.results_dir / "report.md")]:
        print(f"{'✔' if p.exists() else '✘'} {label:12s} {p}")


def cmd_package(a):
    from .utils.packaging import make_zip

    s = get_settings()
    out, n = make_zip(s.root, a.out or s.root / "dist" / f"{s.root.name}.zip", a.with_results)
    print(f"✔ {out}  ({n} fichiers, {out.stat().st_size / 1024:.0f} Ko)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cvmatch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pipeline", help="exécute tout (ou une partie) du pipeline")
    _pipeline_options(p)
    p.add_argument("--only", nargs="+", choices=STEP_NAMES, help="étapes à exécuter")
    p.add_argument("--from", dest="start", choices=STEP_NAMES, help="première étape")
    p.add_argument("--to", dest="stop", choices=STEP_NAMES, help="dernière étape")
    p.set_defaults(func=cmd_pipeline)

    for step in STEPS:
        p = sub.add_parser(step.name, help=step.help)
        _pipeline_options(p)
        if step.name == "ablate-llm":
            p.add_argument("--summarize", action="store_true",
                           help="résume ablation_llm.csv après notation manuelle")
        p.set_defaults(func=cmd_step)

    p = sub.add_parser("match", help="Top-k offres pour un CV (.pdf ou .txt)")
    p.add_argument("file", type=Path)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--ranker", choices=["tfidf", "sbert"])
    p.add_argument("--explainer", choices=["rules", "gemini"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("serve", help="lance l'API FastAPI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    sub.add_parser("info", help="configuration et état des fichiers").set_defaults(func=cmd_info)

    p = sub.add_parser("package", help="crée l'archive zip du projet")
    p.add_argument("--out", type=Path)
    p.add_argument("--with-results", action="store_true", help="inclut results/")
    p.set_defaults(func=cmd_package)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    setup_logging(a.log_level)
    try:
        a.func(a)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✘ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
