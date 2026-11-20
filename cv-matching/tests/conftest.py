import pandas as pd
import pytest

from cvmatch.config import Settings
from cvmatch.data.catalog import prepare_jobs
from cvmatch.data.synthetic import make_synthetic_jobs
from cvmatch.ranking.sbert import hashing_encoder

hash_encoder = hashing_encoder()  # encodeur factice déterministe (pas de téléchargement)


@pytest.fixture(scope="session")
def raw_jobs() -> pd.DataFrame:
    return make_synthetic_jobs(n_per_family=12, seed=1, n_exact_duplicates=5)


@pytest.fixture(scope="session")
def jobs(raw_jobs) -> pd.DataFrame:
    return prepare_jobs(raw_jobs, min_year=2019, max_rows=None)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings isolés : tout est écrit dans tmp_path, aucun réseau."""
    s = Settings()
    s.ranker, s.explainer, s.cleaning = "tfidf", "rules", True
    s.gemini_api_key = ""
    s.sbert_model = "hashing"
    s.root = tmp_path
    s.data_dir = tmp_path / "data"
    s.jobs_path = tmp_path / "data" / "processed" / "jobs_clean.parquet"
    s.artifacts_dir = tmp_path / "artifacts"
    s.cache_dir = tmp_path / "data" / "cache"
    s.eval_dir = tmp_path / "data" / "eval"
    s.results_dir = tmp_path / "results"
    return s
