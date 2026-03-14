from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    strip_html: bool = True
    lowercase: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: Path
    model_filename: str = "model.joblib"

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_filename


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    artifact: ArtifactConfig
    preprocess: PreprocessConfig = PreprocessConfig()
    model_name: str = "baseline"
    seed: int = 42
    max_samples: int | None = None
    test_size: float = 0.2


@dataclass(frozen=True)
class EvalConfig:
    data_path: Path
    artifact_path: Path
    preprocess: PreprocessConfig = PreprocessConfig()
    metrics: tuple[str, ...] = ("accuracy", "f1")
    max_samples: int | None = None
    split_seed: int | None = None
    test_size: float | None = None
    use_artifact_meta: bool = True
    # Performance metrics settings
    measure_latency: bool = True
    latency_warmup: int = 10
    latency_n_iterations: int = 100
