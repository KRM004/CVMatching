"""Pipeline complet hors-ligne (données fictives + encodeur hashing) et commandes CLI."""
import zipfile

import pandas as pd
import pytest

from cvmatch.cli import main
from cvmatch.pipeline import PipelineConfig, run_pipeline, select_steps

pytest.importorskip("pymupdf")


@pytest.fixture
def cfg() -> PipelineConfig:
    c = PipelineConfig()
    c.data.source = "synthetic"
    c.data.max_rows = None
    c.llm.n_cvs = 5
    return c


def test_select_steps():
    assert select_steps(start="labels", stop="ablate-ml") == ["labels", "ablate-ml"]
    assert select_steps(only=["report", "prepare"]) == ["prepare", "report"]
    with pytest.raises(ValueError):
        select_steps(only=["nope"])


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"data": {"sourse": "x"}})


def test_full_offline_pipeline(settings, cfg):
    ctx = run_pipeline(cfg, settings)
    steps = ctx.state["steps"]
    assert all(v["status"] == "ok" for v in steps.values()), steps
    r = settings.results_dir
    summary = pd.read_csv(r / "ablation_ml_summary_PROXY.csv")
    assert set(summary["config"]) == {"B0_tfidf", "F1_sbert_clean", "A1_sbert_noclean"}
    assert summary["mean_p_at_k"].between(0, 1).all()
    assert (r / "ablation_llm.csv").exists() and (r / "pipeline_run.json").exists()
    report = (r / "report.md").read_text(encoding="utf-8")
    assert "PROXY" in report and "B0_tfidf" in report

    # 2e passage : les étapes coûteuses sont sautées, le pool annoté n'est jamais écrasé
    pool = pd.read_csv(settings.eval_dir / "pool_to_annotate.csv")
    pool["annotator_1"] = pool["proxy_relevant"]
    pool["annotator_2"] = pool["proxy_relevant"]
    pool.to_csv(settings.eval_dir / "pool_to_annotate.csv", index=False)
    ctx2 = run_pipeline(cfg, settings, force=False)
    assert ctx2.state["steps"]["prepare"]["status"] == "cached"
    assert ctx2.state["labels_mode"] == "annotated"
    assert (settings.eval_dir / "labels.csv").exists()
    assert (r / "ablation_ml_summary.csv").exists()


def test_tfidf_only_pipeline(settings, cfg):
    cfg.embeddings.enabled = False
    ctx = run_pipeline(cfg, settings)
    summary = pd.read_csv(settings.results_dir / "ablation_ml_summary_PROXY.csv")
    assert list(summary["config"]) == ["B0_tfidf"]
    assert ctx.state["steps"]["embed"]["result"] == {"skipped": True}


def test_cli_match_and_package(settings, cfg, tmp_path, monkeypatch, capsys):
    run_pipeline(cfg, settings, stop="cvs")
    for k, v in {"CVMATCH_HOME": str(settings.root), "JOBS_PATH": str(settings.jobs_path),
                 "RANKER": "tfidf", "EXPLAINER": "rules", "GEMINI_API_KEY": ""}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("cvmatch.config.ROOT", settings.root)
    assert main(["match", str(settings.cvs_dir / "cv_01.pdf"), "--explainer", "rules"]) == 0
    assert "1. [" in capsys.readouterr().out

    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1")
    from cvmatch.utils.packaging import make_zip

    out, n = make_zip(tmp_path, tmp_path / "dist" / "p.zip")
    names = zipfile.ZipFile(out).namelist()
    prefix = tmp_path.name
    assert f"{prefix}/src/a.py" in names
    assert not any(x.endswith(".env") or "/data/processed/" in x for x in names)
