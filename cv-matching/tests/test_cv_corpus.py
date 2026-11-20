import numpy as np
import pandas as pd

from cvmatch.cv_corpus import normalize_corpus, read_cv_folder, read_cv_table, sample_for_annotation
from cvmatch.data import job_texts
from cvmatch.ranking import SbertRanker, TfidfRanker
from cvmatch.synthetic import make_demo_cvs
from conftest import hash_encoder

CVS = make_demo_cvs()
TEXTS = [c["text"] for c in CVS.values()]


def test_read_table_autodetects_columns(tmp_path):
    df = pd.DataFrame({"ID": range(4), "Resume_str": TEXTS[:4], "Category": ["IT", "IT", "RH", "RH"]})
    path = tmp_path / "Resume.csv"
    df.to_csv(path, index=False)
    out = read_cv_table(path)
    assert list(out.columns) == ["cv_id", "text", "category", "source"]
    assert out["category"].tolist() == ["IT", "IT", "RH", "RH"]


def test_normalize_masks_pii_and_dedups():
    raw = pd.DataFrame({
        "cv_id": ["a", "b", "c", "d"],
        "text": [TEXTS[0], TEXTS[0], "trop court", TEXTS[1]],
        "category": "", "source": "x",
    })  # fmt: skip
    out, stats = normalize_corpus(raw)
    assert stats == {"n_input": 4, "n_too_short": 1, "n_duplicates": 1, "n_kept": 2}
    assert not out["text"].str.contains("@").any()  # e-mails masqués dès l'import


def test_read_folder_uses_subfolder_as_category(tmp_path):
    (tmp_path / "IT").mkdir()
    (tmp_path / "IT" / "cv1.txt").write_text(TEXTS[0], encoding="utf-8")
    (tmp_path / "cv2.txt").write_text(TEXTS[1], encoding="utf-8")
    df, errors = read_cv_folder(tmp_path)
    assert not errors and dict(zip(df["cv_id"], df["category"])) == {"cv1": "IT", "cv2": ""}


def test_sample_is_stratified():
    corpus = pd.DataFrame({"cv_id": [str(i) for i in range(100)], "text": "x",
                           "category": ["A"] * 80 + ["B"] * 20, "source": ""})  # fmt: skip
    sample = sample_for_annotation(corpus, 10)
    assert len(sample) == 10 and set(sample["category"]) == {"A", "B"}


def test_score_many_matches_rank(jobs):
    tfidf = TfidfRanker(job_texts(jobs, True))
    sbert = SbertRanker(hash_encoder(job_texts(jobs, True)), hash_encoder)
    for ranker in (tfidf, sbert):
        scores = ranker.score_many(TEXTS[:3])
        assert scores.shape == (3, len(jobs))
        for i in range(3):
            top = ranker.rank(TEXTS[i], top_k=3)
            assert np.isclose(scores[i].max(), top[0].score, atol=1e-5)
