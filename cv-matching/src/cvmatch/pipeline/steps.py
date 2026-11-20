"""Les étapes du pipeline hors-ligne. Chaque étape lit/écrit des fichiers bien identifiés :

  prepare    -> data/processed/jobs_clean.parquet
  embed      -> artifacts/emb_<modèle>_{clean,raw}.npy (+ .json)
  cvs        -> data/eval/cvs/*.txt|pdf (+ data/eval/cv_meta.json)
  pool       -> data/eval/pool_to_annotate.csv (+ proxy_labels.csv)
  labels     -> data/eval/labels.csv (si annotations) sinon labels proxy
  ablate-ml  -> results/ablation_ml[_PROXY].csv + ablation_ml_summary[_PROXY].csv
  ablate-llm -> results/ablation_llm.csv + ablation_llm_summary.csv
  report     -> results/report.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..utils.io import read_json, write_json, write_table
from ..utils.logging import get_logger
from .config import PipelineConfig

log = get_logger("cvmatch.pipeline")


@dataclass
class Context:
    settings: Settings
    cfg: PipelineConfig
    force: bool = False
    state: dict = field(default_factory=dict)

    # chemins dérivés (une seule source de vérité)
    @property
    def eval_dir(self) -> Path:
        return self.settings.eval_dir

    @property
    def pool_csv(self) -> Path:
        return self.eval_dir / "pool_to_annotate.csv"

    @property
    def labels_csv(self) -> Path:
        return self.eval_dir / "labels.csv"

    @property
    def proxy_csv(self) -> Path:
        return self.eval_dir / "proxy_labels.csv"

    @property
    def use_sbert(self) -> bool:
        return self.cfg.embeddings.enabled

    def jobs(self):
        if "jobs" not in self.state:
            from ..data.catalog import load_jobs

            self.state["jobs"] = load_jobs(self.settings.jobs_path)
        return self.state["jobs"]

    def cvs(self) -> dict[str, str]:
        if "cvs" not in self.state:
            from ..evaluation.annotation import load_cvs

            self.state["cvs"] = load_cvs(self.settings.cvs_dir)
        return self.state["cvs"]

    def encoder(self):
        if "encoder" not in self.state:
            from ..ranking import load_sentence_transformer

            log.info("Chargement de l'encodeur : %s", self.settings.sbert_model)
            self.state["encoder"] = load_sentence_transformer(self.settings.sbert_model)
        return self.state["encoder"]


@dataclass
class Step:
    name: str
    help: str
    run: Callable[[Context], dict]
    outputs: Callable[[Context], list[Path]] = lambda ctx: []
    cacheable: bool = False  # True : sautée si toutes ses sorties existent (sauf --force)

    def is_done(self, ctx: Context) -> bool:
        outs = self.outputs(ctx)
        return self.cacheable and bool(outs) and all(p.exists() for p in outs)


# ------------------------------------------------------------------ 1. données
def _prepare(ctx: Context) -> dict:
    from ..data.catalog import load_raw_jobs, prepare_jobs, validate_jobs

    d = ctx.cfg.data
    if d.source == "synthetic":
        from ..data.synthetic import make_synthetic_jobs

        log.warning("Données SYNTHÉTIQUES : ne pas utiliser pour rapporter des résultats.")
        raw = make_synthetic_jobs()
    elif d.source == "local":
        if not d.raw_path:
            raise ValueError("data.source = local exige data.raw_path")
        raw = load_raw_jobs(ctx.settings.root / d.raw_path)
    elif d.source == "huggingface":
        raw = load_raw_jobs(None)
    else:
        raise ValueError(f"data.source inconnu : {d.source}")
    log.info("Brut : %s lignes × %s colonnes", f"{len(raw):,}", raw.shape[1])
    jobs = prepare_jobs(raw, min_year=d.min_year, max_rows=d.max_rows, sectors=d.sectors or None)
    validate_jobs(jobs)
    write_table(jobs, ctx.settings.jobs_path)
    ctx.state["jobs"] = jobs.reset_index(drop=True)
    log.info("%s offres nettoyées -> %s", f"{len(jobs):,}", ctx.settings.jobs_path)
    return {"n_raw": len(raw), "n_jobs": len(jobs)}


# ------------------------------------------------------------------ 2. embeddings
def _variants(ctx: Context) -> list[bool]:
    return [v == "clean" for v in ctx.cfg.embeddings.variants]


def _embed_outputs(ctx: Context) -> list[Path]:
    from ..ranking.embeddings import embeddings_path

    if not ctx.use_sbert:
        return []
    s = ctx.settings
    return [embeddings_path(s.artifacts_dir, s.sbert_model, c) for c in _variants(ctx)]


def _embed(ctx: Context) -> dict:
    from ..ranking.embeddings import build_embeddings

    if not ctx.use_sbert:
        log.info("embeddings.enabled = false : étape ignorée (TF-IDF seul).")
        return {"skipped": True}
    out = {}
    for cleaning in _variants(ctx):
        p = build_embeddings(ctx.jobs(), ctx.settings.sbert_model, ctx.settings.artifacts_dir,
                             cleaning=cleaning, encoder=ctx.encoder(),
                             batch_size=ctx.cfg.embeddings.batch_size)
        out["clean" if cleaning else "raw"] = str(p)
    return out


# ------------------------------------------------------------------ 3. CV de test
def _cvs(ctx: Context) -> dict:
    from ..evaluation.annotation import write_demo_cvs

    cv_dir = ctx.settings.cvs_dir
    existing = [p for p in cv_dir.glob("*") if p.suffix.lower() in {".txt", ".pdf"}]
    if existing and not ctx.force:
        log.info("%d CV déjà présents dans %s : conservés.", len(existing), cv_dir)
    elif ctx.cfg.eval.demo_cvs:
        meta = write_demo_cvs(cv_dir)
        write_json(meta, ctx.eval_dir / "cv_meta.json")
        log.info("%d CV FICTIFS écrits -> %s", len(meta), cv_dir)
    else:
        raise FileNotFoundError(f"Aucun CV dans {cv_dir} (eval.demo_cvs = false).")
    ctx.state.pop("cvs", None)
    return {"n_cvs": len(ctx.cvs())}


# ------------------------------------------------------------------ 4. pool à annoter
def _pool(ctx: Context) -> dict:
    from ..evaluation.annotation import build_pool, has_annotations, write_proxy_labels
    from ..preprocessing import prepare_resume
    from ..ranking import build_ranker

    if has_annotations(ctx.pool_csv):
        log.warning("Le pool contient déjà des annotations : il n'est PAS régénéré.")
        return {"kept_existing": True}
    e = ctx.cfg.eval
    jobs = ctx.jobs()
    rankers = [build_ranker(jobs, ctx.settings, name="tfidf", cleaning=True)]
    if ctx.use_sbert:
        rankers.append(build_ranker(jobs, ctx.settings, encoder=ctx.encoder(), name="sbert",
                                    cleaning=True))
    queries = {cv: prepare_resume(t, True)[1] for cv, t in ctx.cvs().items()}
    meta = read_json(ctx.eval_dir / "cv_meta.json", {}) or {}
    pool = build_pool(jobs, queries, rankers, e.pool_top_n, e.pool_random, e.seed, meta)
    write_table(pool, ctx.pool_csv)
    n_proxy = write_proxy_labels(pool, ctx.proxy_csv) if len(pool) else 0
    log.info("%d paires (CV, offre) -> %s | %d labels proxy", len(pool), ctx.pool_csv, n_proxy)
    return {"n_pairs": len(pool), "n_proxy_labels": n_proxy}


# ------------------------------------------------------------------ 5. labels
def _labels(ctx: Context) -> dict:
    from ..evaluation.annotation import finalize_labels, has_annotations

    mode = ctx.cfg.eval.labels
    annotated = has_annotations(ctx.pool_csv)
    if mode == "annotated" or (mode == "auto" and annotated):
        if not annotated:
            raise ValueError(f"eval.labels = annotated mais {ctx.pool_csv} n'est pas annoté.")
        stats = finalize_labels(ctx.pool_csv, ctx.labels_csv, ctx.eval_dir / "disagreements.csv")
        ctx.state["labels_mode"] = "annotated"
        log.info("labels.csv (2 annotateurs) : %s", stats)
        return {"mode": "annotated", **stats}
    if not ctx.proxy_csv.exists():
        raise FileNotFoundError(f"{ctx.proxy_csv} introuvable (lancez l'étape pool).")
    ctx.state["labels_mode"] = "proxy"
    log.warning("Labels PROXY (mots-clés) : test technique UNIQUEMENT, pas de résultats à rapporter.")
    return {"mode": "proxy"}


def _labels_path(ctx: Context) -> tuple[Path, str]:
    mode = ctx.state.get("labels_mode")
    if mode is None:
        _labels(ctx)
        mode = ctx.state["labels_mode"]
    return (ctx.labels_csv, "") if mode == "annotated" else (ctx.proxy_csv, "_PROXY")


# ------------------------------------------------------------------ 6. ablation ML
def _ablate_ml(ctx: Context) -> dict:
    from ..evaluation.ablation_ml import run_ml_ablation
    from ..evaluation.annotation import load_labels

    path, tag = _labels_path(ctx)
    per_cv, summary = run_ml_ablation(
        ctx.jobs(), ctx.cvs(), load_labels(path), ctx.settings, with_sbert=ctx.use_sbert,
        k=ctx.cfg.eval.k, encoder=ctx.encoder() if ctx.use_sbert else None)
    r = ctx.settings.results_dir
    write_table(per_cv, r / f"ablation_ml{tag}.csv")
    write_table(summary, r / f"ablation_ml_summary{tag}.csv")
    return {"summary": summary.round(4).to_dict(orient="records")}


# ------------------------------------------------------------------ 7. ablation LLM
def _ablate_llm(ctx: Context) -> dict:
    from ..evaluation.ablation_llm import run_llm_ablation, summarize_llm
    from ..ranking import build_ranker

    l = ctx.cfg.llm  # noqa: E741
    if not l.enabled:
        return {"skipped": True}
    name = l.ranker if (l.ranker != "sbert" or ctx.use_sbert) else "tfidf"
    enc = ctx.encoder() if name == "sbert" else None
    ranker = build_ranker(ctx.jobs(), ctx.settings, encoder=enc, name=name)
    cvs = dict(list(ctx.cvs().items())[: l.n_cvs])
    df = run_llm_ablation(ctx.jobs(), cvs, ranker, ctx.settings, l.use_cache, ctx.cfg.eval.k)
    r = ctx.settings.results_dir
    write_table(df, r / "ablation_llm.csv")
    summary = summarize_llm(df)
    write_table(summary, r / "ablation_llm_summary.csv")
    return {"summary": summary.to_dict(orient="records")}


# ------------------------------------------------------------------ 8. rapport
def _report(ctx: Context) -> dict:
    from ..evaluation.report import build_report

    info = {
        "labels": ctx.state.get("labels_mode"),
        "synthetic": ctx.cfg.data.source == "synthetic",
        "sbert_model": ctx.settings.sbert_model if ctx.use_sbert else None,
        "steps": ctx.state.get("steps", {}),
    }
    return {"path": str(build_report(ctx.settings.results_dir, info))}


STEPS: list[Step] = [
    Step("prepare", "Télécharge / charge et nettoie le catalogue d'offres", _prepare,
         lambda c: [c.settings.jobs_path], cacheable=True),
    Step("embed", "Calcule les embeddings Sentence-BERT des offres", _embed, _embed_outputs,
         cacheable=True),
    Step("cvs", "Prépare les CV de test (20 CV fictifs si dossier vide)", _cvs),
    Step("pool", "Construit le pool (CV, offre) à annoter + labels proxy", _pool,
         lambda c: [c.pool_csv], cacheable=True),
    Step("labels", "Fusionne les annotations (ou bascule en labels proxy)", _labels),
    Step("ablate-ml", "Ablation B0 / F1 / A1 : Precision@k + latence", _ablate_ml),
    Step("ablate-llm", "Ablation L0 / L1 / A2 : fidélité + latence", _ablate_llm),
    Step("report", "Rapport Markdown (results/report.md)", _report),
]
STEP_NAMES = [s.name for s in STEPS]
