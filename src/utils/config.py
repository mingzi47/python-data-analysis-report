from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    random_seed: int = 42
    sample_size: int | None = 500_000  # None = 全量数据
    test_size: float = 0.2
    cv_folds: int = 3
    cv_iter: int = 20
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("outputs/figures")
    model_dir: Path = Path("outputs/models")
