"""Paramètres des étapes hors-ligne (configs/pipeline.yaml), surchargeables en CLI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    source: str = "huggingface"        # huggingface | synthetic | local
    raw_path: Optional[str] = None     # requis si source = local (.parquet / .csv)
    min_year: int = 2019
    max_rows: Optional[int] = 15000
    sectors: list[str] = field(default_factory=list)


@dataclass
class EmbeddingsConfig:
    enabled: bool = True               # False = pipeline TF-IDF uniquement
    variants: list[str] = field(default_factory=lambda: ["clean", "raw"])
    batch_size: int = 64


@dataclass
class EvalConfig:
    demo_cvs: bool = True              # génère 20 CV fictifs si data/eval/cvs/ est vide
    pool_top_n: int = 6
    pool_random: int = 4
    seed: int = 42
    k: int = 3
    labels: str = "auto"               # auto | annotated | proxy


@dataclass
class LlmConfig:
    enabled: bool = True
    ranker: str = "tfidf"
    n_cvs: int = 20
    use_cache: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        parts = {}
        for f in fields(cls):
            sub_cls = f.default_factory  # type: ignore[misc]
            known = {x.name for x in fields(sub_cls)}
            given = d.get(f.name) or {}
            unknown = set(given) - known
            if unknown:
                raise ValueError(f"Clés inconnues dans '{f.name}' : {sorted(unknown)}")
            parts[f.name] = sub_cls(**given)
        return cls(**parts)

    @classmethod
    def load(cls, path: Optional[Path]) -> "PipelineConfig":
        if path is None or not Path(path).exists():
            return cls()
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict:
        return asdict(self)
