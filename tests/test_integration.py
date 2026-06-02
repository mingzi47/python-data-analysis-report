"""Integration tests for the full analysis pipeline.

Verifies that the main pipeline stages (load_and_clean → run_eda → run_modeling)
work together without crashing, data flows consistently between stages, and output
artifacts are generated.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest

from src.utils.config import Config


# ---------------------------------------------------------------------------
# Fixtures — generate enough data for GroupShuffleSplit, CV, and early-stopping
# ---------------------------------------------------------------------------

def _make_recommendations(rng: np.random.RandomState, n_users: int, n_games: int) -> pd.DataFrame:
    """Deterministic recommendation records. Each user reviews ~4-8 distinct games."""
    recs = []
    for user_id in range(1, n_users + 1):
        n_recs = rng.randint(4, 9)
        games_for_user = rng.choice(range(100, 100 + n_games), size=n_recs, replace=False)
        for app_id in games_for_user:
            recs.append({
                "app_id": int(app_id),
                "user_id": user_id,
                "is_recommended": int(rng.random() > 0.35),  # ~65% positive
                "hours": float(round(rng.uniform(0.1, 800.0), 1)),
                "date": "2021-06-15",
                "helpful": int(rng.poisson(2)),
                "funny": int(rng.poisson(0.5)),
            })
    return pd.DataFrame(recs)


@pytest.fixture(scope="function")
def integration_data_dir(tmp_path):
    """Self-contained dataset: 15 games, 25 users, ~150 recommendations."""
    seed = 42
    n_games = 15
    n_users = 25

    # --- games.csv ---
    games = pd.DataFrame({
        "app_id": list(range(100, 100 + n_games)),
        "title": [f"Game_{i}" for i in range(100, 100 + n_games)],
        "date_release": ["2020-01-15"] * n_games,
        "rating": [0.85] * n_games,
        "positive_ratio": [85] * n_games,
        "price_original": [9.99] * n_games,
        "price_final": [9.99] * n_games,
        "win": [1] * n_games,
        "mac": [0] * n_games,
        "linux": [0] * n_games,
        "steam_deck": [0] * n_games,
        "discount": [0.0] * n_games,
        "user_reviews": [1000] * n_games,
    })
    games.to_csv(tmp_path / "games.csv", index=False)

    # --- users.csv ---
    users = pd.DataFrame({
        "user_id": list(range(1, n_users + 1)),
        "products": [35 + i % 20 for i in range(n_users)],
        "reviews": [8 + i % 12 for i in range(n_users)],
    })
    users.to_csv(tmp_path / "users.csv", index=False)

    # --- recommendations.csv ---
    recs = _make_recommendations(np.random.RandomState(seed), n_users, n_games)
    recs.to_csv(tmp_path / "recommendations.csv", index=False)

    # --- games_metadata.json ---
    tags_pool = ["Action", "Indie", "Strategy", "RPG", "Simulation", "Adventure"]
    with open(tmp_path / "games_metadata.json", "w") as f:
        for app_id in range(100, 100 + n_games):
            json.dump({
                "app_id": app_id,
                "description": f"Description for game {app_id}.",
                "tags": [tags_pool[app_id % len(tags_pool)], tags_pool[(app_id + 1) % len(tags_pool)]],
                "genres": [tags_pool[app_id % len(tags_pool)]],
                "type": "game",
                "early_access": 0,
            }, f)
            f.write("\n")

    return tmp_path


@pytest.fixture(scope="function")
def integration_config(tmp_path):
    """Config with all paths rooted in tmp_path (no real kagglehub download)."""
    config = Config(
        sample_size=None,          # use all rows in the fixture
        data_dir=tmp_path,
        output_dir=tmp_path / "outputs",
    )
    # main.py:252-253 creates these before running stages; we do the same here
    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline_runs_without_error(
        self, tmp_path, integration_config, integration_data_dir, monkeypatch,
    ):
        """The complete pipeline (load → clean → EDA → feature engineering →
        model training → evaluation → visualization) should complete without
        raising an exception."""
        from main import load_and_clean, run_eda, run_modeling

        # Stub kagglehub download — return our pre-built data directory
        monkeypatch.setattr(
            "main.download_dataset",
            lambda output_dir: integration_data_dir,
        )

        # Run pipeline stages
        games_df, users_df, recs_df = load_and_clean(integration_config)
        run_eda(games_df, users_df, recs_df, integration_config)
        run_modeling(games_df, users_df, recs_df, integration_config)

    def test_data_flow_shapes_consistent(
        self, tmp_path, integration_config, integration_data_dir, monkeypatch,
    ):
        """Data flow invariants:
        - Train / test user sets are disjoint (no leakage)
        - Feature matrices have matching row counts with their labels
        - Train and test have identical feature columns
        - Labels are binary {0, 1}
        """
        from main import load_and_clean
        from src.models.trainer import split_recommendations
        from src.features.builder import fit_interaction_aggregates, build_features

        monkeypatch.setattr(
            "main.download_dataset",
            lambda output_dir: integration_data_dir,
        )

        games_df, users_df, recs_df = load_and_clean(integration_config)

        # --- Stage 5: split → aggregates → features ---
        train_recs, test_recs = split_recommendations(recs_df, test_size=0.2, random_state=42)

        train_users = set(train_recs["user_id"])
        test_users = set(test_recs["user_id"])
        assert train_users.isdisjoint(test_users), "Train/test user sets must be disjoint"

        game_aggs, user_aggs = fit_interaction_aggregates(train_recs)

        X_train, y_train, groups_train = build_features(train_recs, games_df, users_df)
        X_test, y_test, groups_test = build_features(
            test_recs, games_df, users_df, game_aggs=game_aggs, user_aggs=user_aggs,
        )

        assert len(X_train) == len(y_train), "X_train and y_train row counts must match"
        assert len(X_test) == len(y_test), "X_test and y_test row counts must match"
        assert len(y_train) == len(groups_train), "y_train and groups_train row counts must match"
        assert len(y_test) == len(groups_test), "y_test and groups_test row counts must match"

        assert X_train.shape[1] == X_test.shape[1], (
            f"Train ({X_train.shape[1]}) and test ({X_test.shape[1]}) "
            f"must have the same number of feature columns"
        )

        assert set(X_train.columns) == set(X_test.columns), (
            "Train and test feature column names must be identical"
        )

        assert y_train.isin([0, 1]).all(), "y_train must be binary"
        assert y_test.isin([0, 1]).all(), "y_test must be binary"

    def test_output_files_generated(
        self, tmp_path, integration_config, integration_data_dir, monkeypatch,
    ):
        """Pipeline should produce EDA figures and a model comparison CSV."""
        from main import load_and_clean, run_eda, run_modeling

        monkeypatch.setattr(
            "main.download_dataset",
            lambda output_dir: integration_data_dir,
        )

        games_df, users_df, recs_df = load_and_clean(integration_config)
        run_eda(games_df, users_df, recs_df, integration_config)
        run_modeling(games_df, users_df, recs_df, integration_config)

        figure_dir = integration_config.figure_dir
        model_dir = integration_config.model_dir

        # EDA + model figures
        assert figure_dir.exists(), f"Figure directory missing: {figure_dir}"
        figures = sorted(figure_dir.glob("*.png"))
        assert len(figures) >= 5, (
            f"Expected ≥5 figures, got {len(figures)}: {[f.name for f in figures]}"
        )

        # Model comparison table
        assert model_dir.exists(), f"Model directory missing: {model_dir}"
        comparison_csv = model_dir / "comparison.csv"
        assert comparison_csv.exists(), f"Missing: {comparison_csv}"
