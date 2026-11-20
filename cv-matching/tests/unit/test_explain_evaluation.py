import json

import pytest

from cvmatch.data import job_texts
from cvmatch.data.synthetic import make_demo_cvs
from cvmatch.evaluation import delta, evaluate_ranker, finalize_labels, precision_at_k
from cvmatch.explain.gemini import GeminiExplainer, verify_against_sources
from cvmatch.explain.rules import RuleExplainer
from cvmatch.ranking import TfidfRanker
from cvmatch.service import build_service
from cvmatch.utils.io import write_table

CVS = make_demo_cvs()
JOB = {
    "job_title": "Data Analyst",
    "job_description": "Analyser des données avec Python et SQL. Maîtrise de Power BI et Tableau software.",
}
CV = "Analyste. Je maîtrise Python et SQL. Excel au quotidien."


class FakeClient:
    def __init__(self, payload):
        self.calls, self.payload = 0, payload
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        return type("R", (), {"text": json.dumps(self.payload)})()


def test_rules_explainer():
    e = RuleExplainer("fr").explain(CV, JOB)
    assert set(e.matching_skills) == {"python", "sql"}
    assert "power bi" in e.missing_skills
    assert e.source == "rules" and e.evidence


def test_gemini_guard_removes_invented_claims(tmp_path):
    payload = {
        "matching_skills": ["Python", "SQL", "Kubernetes"],  # Kubernetes : inventé
        "missing_skills": ["Power BI", "Excel"],  # Excel est dans le CV : contradiction
        "explanation": "Votre profil correspond.",
        "evidence": ["Je maîtrise Python et SQL", "5 ans chez Google"],  # 2e : inventée
    }
    client = FakeClient(payload)
    ex = GeminiExplainer("k", cache_dir=tmp_path, client=client, min_interval=0)
    e = ex.explain(CV, JOB)
    assert e.source == "gemini"
    assert e.matching_skills == ["Python", "SQL"]
    assert e.missing_skills == ["Power BI"]
    assert e.evidence == ["Je maîtrise Python et SQL"]
    assert e.guard_removed == 3 and "Kubernetes" in e.unverified_skills
    ex.explain(CV, JOB)  # 2e appel identique : servi par le cache
    assert client.calls == 1


def test_gemini_without_key_falls_back_to_rules():
    e = GeminiExplainer("", min_interval=0).explain(CV, JOB)
    assert e.source == "rules_fallback" and "GEMINI_API_KEY" in (e.error or "")
    assert e.matching_skills


def test_verify_without_job_context():
    m, miss, ev, unv, removed = verify_against_sources(
        {"matching_skills": ["Python"], "missing_skills": ["Power BI"], "evidence": []}, CV, None)
    assert m == ["Python"] and miss == ["Power BI"] and removed == 0


def test_service_end_to_end_and_pii_not_sent(jobs, settings):
    seen = []

    class Spy(RuleExplainer):
        def explain(self, resume_text, job, resume_skills=None):
            seen.append(resume_text)
            return super().explain(resume_text, job, resume_skills)

    write_table(jobs, settings.jobs_path)
    svc = build_service(settings)
    svc.explainers["rules"] = Spy("fr")
    res = svc.match(CVS["cv_02"]["text"], explain=True, explainer="rules")
    assert len(res.recommendations) == 3 and res.ranker == "tfidf"
    scores = [r.similarity_score for r in res.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert res.recommendations[0].explanation is not None
    assert set(res.timings_ms) >= {"preprocess", "ranking", "explain", "total"}
    assert seen and all("@example.com" not in t and "+212" not in t for t in seen)
    # gemini demandé sans clé : repli sur les règles + avertissement, jamais de crash
    res2 = svc.match(CVS["cv_02"]["text"], explainer="gemini")
    assert res2.recommendations[0].explanation.source == "rules_fallback" and res2.warnings


def test_service_rejects_empty_resume(jobs, settings):
    write_table(jobs, settings.jobs_path)
    with pytest.raises(ValueError):
        build_service(settings).match("   ")


def test_evaluation_metrics(jobs, tmp_path):
    assert precision_at_k([1, 2, 3], {1, 3, 9}) == pytest.approx(2 / 3)
    assert delta(0.80, 0.74) == pytest.approx(0.06)
    ids = jobs["job_id"].tolist()
    labels = {"cv_01": {ids[0]: 1, ids[1]: 0, ids[2]: 0, ids[3]: 1}}
    ranker = TfidfRanker(job_texts(jobs, True))
    df = evaluate_ranker(ranker, jobs, {"cv_01": CVS["cv_01"]["text"]}, labels)
    assert len(df) == 1 and 0 <= df["p_at_k"][0] <= df["p_at_k_max"][0] <= 1
    assert df["latency_s"][0] > 0


def test_finalize_labels(tmp_path):
    import pandas as pd

    pool = tmp_path / "pool.csv"
    pd.DataFrame({"cv_id": ["a"] * 4, "job_id": [1, 2, 3, 4],
                  "annotator_1": [1, 1, 0, 0], "annotator_2": [1, 0, 0, 1]}).to_csv(pool, index=False)  # fmt: skip
    stats = finalize_labels(pool, tmp_path / "labels.csv", tmp_path / "dis.csv")
    labels = pd.read_csv(tmp_path / "labels.csv")
    assert labels["relevant"].tolist() == [1, 0, 0, 0]
    assert stats["n_disagreements"] == 2 and stats["agreement"] == 0.5


def test_recommend_jobs_contract(jobs):
    from cvmatch.matching import recommend_jobs

    out = recommend_jobs(CVS["cv_04"]["text"], jobs, top_k=3)
    assert set(out) == {"resume_skills", "recommendations"}
    assert len(out["recommendations"]) == 3
    assert set(out["recommendations"][0]) == {"job_id", "job_title", "job_description",
                                              "similarity_score"}
