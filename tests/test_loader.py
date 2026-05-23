import pytest
import pandas as pd
from pathlib import Path


class TestLoadGames:
    def test_loads_csv_and_returns_dataframe(self, sample_games_csv):
        from src.data.loader import load_games

        df = load_games(sample_games_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_returns_all_expected_columns(self, sample_games_csv):
        from src.data.loader import load_games

        df = load_games(sample_games_csv)

        expected_cols = {"app_id", "title", "date_release", "rating",
                         "positive_ratio", "user_score", "price_original",
                         "price_final", "discount", "owners", "steam_deck"}
        assert set(df.columns) == expected_cols

    def test_nrows_limits_output(self, sample_games_csv):
        from src.data.loader import load_games

        df = load_games(sample_games_csv, nrows=2)

        assert len(df) == 2


class TestLoadUsers:
    def test_loads_csv_and_returns_dataframe(self, sample_users_csv):
        from src.data.loader import load_users

        df = load_users(sample_users_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_match_expected(self, sample_users_csv):
        from src.data.loader import load_users

        df = load_users(sample_users_csv)

        assert list(df.columns) == ["user_id", "products", "reviews"]


class TestLoadRecommendations:
    def test_loads_with_nrows_sampling(self, sample_recommendations_csv):
        from src.data.loader import load_recommendations

        df = load_recommendations(sample_recommendations_csv, nrows=3)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_match_expected(self, sample_recommendations_csv):
        from src.data.loader import load_recommendations

        df = load_recommendations(sample_recommendations_csv)

        expected_cols = {"app_id", "user_id", "is_recommended", "hours",
                         "date", "helpful", "funny"}
        assert set(df.columns) == expected_cols


class TestLoadMetadata:
    def test_loads_jsonl_and_returns_dataframe(self, sample_metadata_json):
        from src.data.loader import load_metadata

        df = load_metadata(sample_metadata_json)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_contains_expected_columns(self, sample_metadata_json):
        from src.data.loader import load_metadata

        df = load_metadata(sample_metadata_json)

        for col in ["app_id", "tags", "genres", "type", "early_access"]:
            assert col in df.columns

    def test_tags_and_genres_are_lists(self, sample_metadata_json):
        from src.data.loader import load_metadata

        df = load_metadata(sample_metadata_json)

        assert isinstance(df["tags"].iloc[0], list)
        assert isinstance(df["genres"].iloc[0], list)


class TestDownloadDataset:
    def test_returns_path_object(self, monkeypatch):
        import kagglehub

        def mock_download(dataset):
            return "/fake/path/to/dataset"

        monkeypatch.setattr(kagglehub, "dataset_download", mock_download)

        from src.data.loader import download_dataset
        path = download_dataset()

        assert isinstance(path, Path)
