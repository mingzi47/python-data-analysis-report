import pytest
from pathlib import Path


def test_config_default_values():
    from src.utils.config import Config

    config = Config()

    assert config.random_seed == 42
    assert config.sample_size == 500_000
    assert config.test_size == 0.2
    assert config.cv_folds == 3
    assert config.cv_iter == 20
    assert config.output_dir == Path("outputs")
    assert config.figure_dir == Path("outputs/figures")
    assert config.model_dir == Path("outputs/models")


def test_config_custom_values():
    from src.utils.config import Config

    config = Config(random_seed=123, sample_size=1000, test_size=0.3, cv_folds=5)

    assert config.random_seed == 123
    assert config.sample_size == 1000
    assert config.test_size == 0.3
    assert config.cv_folds == 5
    assert config.cv_iter == 20  # default unchanged
