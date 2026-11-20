import pytest

from conftest import hash_encoder
from cvmatch.data import job_texts, validate_jobs
from cvmatch.data.synthetic import make_demo_cvs
from cvmatch.ranking import SbertRanker, TfidfRanker, diversify
from cvmatch.ranking.base import Ranked
from cvmatch.ranking.embeddings import build_embeddings, embeddings_path, load_embeddings

CVS = make_demo_cvs()


def test_prepare_jobs_dedup_and_validation(raw_jobs, jobs):
    assert len(jobs) == len(raw_jobs) - 5  # 5 doublons exacts retirés
    validate_jobs(jobs)
    assert {"job_id", "job_title", "job_description"} <= set(jobs.columns)


def test_tfidf_finds_right_family(jobs):
    ranker = TfidfRanker(job_texts(jobs, True))
    top = ranker.rank(CVS["cv_01"]["text"], top_k=3)  # Data Analyst
    titles = [jobs.iloc[r.index]["job_title"].lower() for r in top]
    assert len(top) == 3 and top[0].score >= top[1].score >= top[2].score
    assert any("data" in t or "donn" in t for t in titles)


def test_candidate_restriction(jobs):
    ranker = TfidfRanker(job_texts(jobs, True))
    pool = [3, 40, 90, 150]
    top = ranker.rank(CVS["cv_04"]["text"], top_k=3, candidates=pool)
    assert {r.index for r in top} <= set(pool)


def test_sbert_ranker_and_artifacts(jobs, tmp_path):
    out = build_embeddings(jobs, "fake-model", tmp_path, cleaning=True, encoder=hash_encoder, batch_size=64)
    assert out == embeddings_path(tmp_path, "fake-model", True) and out.exists()
    emb = load_embeddings(jobs, "fake-model", tmp_path, True)
    ranker = SbertRanker(emb, hash_encoder)
    top = ranker.rank(CVS["cv_08"]["text"], top_k=3)  # DevOps
    assert len(top) == 3 and all(-1.01 <= r.score <= 1.01 for r in top)
    assert "devops" in " ".join(jobs.iloc[r.index]["job_title"].lower() for r in top)
    with pytest.raises(ValueError):
        load_embeddings(jobs.iloc[:10], "fake-model", tmp_path, True)
    with pytest.raises(FileNotFoundError):
        load_embeddings(jobs, "absent-model", tmp_path, True)


def test_diversify_avoids_duplicate_titles():
    titles = ["Data Analyst", "Data Analyst (H/F)", "Data Scientist", "Data Analyst"]
    ranked = [Ranked(i, 1 - i * 0.1) for i in range(4)]
    out = diversify(ranked, titles, 2)
    assert [titles[r.index] for r in out] == ["Data Analyst", "Data Scientist"]
