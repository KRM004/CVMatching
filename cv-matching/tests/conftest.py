import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from cvmatch.config import Settings
from cvmatch.data import prepare_jobs
from cvmatch.synthetic import make_synthetic_jobs


def hash_encoder(texts):
    """Encodeur factice déterministe (pas de téléchargement) pour tester la chaîne SBERT."""
    hv = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")
    return hv.transform(list(texts)).toarray().astype(np.float32)


@pytest.fixture(scope="session")
def raw_jobs() -> pd.DataFrame:
    return make_synthetic_jobs(n_per_family=12, seed=1, n_exact_duplicates=5)


@pytest.fixture(scope="session")
def jobs(raw_jobs) -> pd.DataFrame:
    return prepare_jobs(raw_jobs, min_year=2019, max_rows=None)


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.ranker, s.explainer, s.cleaning = "tfidf", "rules", True
    s.gemini_api_key = ""
    s.artifacts_dir = tmp_path / "artifacts"
    s.cache_dir = tmp_path / "cache"
    s.jobs_path = tmp_path / "jobs.csv"
    return s
