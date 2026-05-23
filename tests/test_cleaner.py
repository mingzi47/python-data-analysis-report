import pytest
import pandas as pd
import numpy as np


def make_games_df(data=None):
    """Helper to create a games DataFrame with realistic values."""
    defaults = {
        "app_id": [730, 570, 440],
        "title": ["CS:GO", "Dota 2", "TF2"],
        "date_release": ["2012-08-21", "2013-07-09", "2007-10-10"],
        "rating": [0.87, 0.82, 0.93],
        "positive_ratio": [0.85, 0.80, 0.92],
        "user_score": [8, 7, 9],
        "price_original": [14.99, 0.0, 19.99],
        "price_final": [14.99, 0.0, 19.99],
        "discount": [0.0, 0.0, 0.0],
        "owners": ["1000000-2000000", "500000-1000000", "500000-1000000"],
        "steam_deck": [1, 0, 1],
    }
    if data:
        defaults.update(data)
    return pd.DataFrame(defaults)


def make_users_df(data=None):
    defaults = {
        "user_id": [1, 2, 3],
        "products": [42, 15, 0],
        "reviews": [10, 5, 0],
    }
    if data:
        defaults.update(data)
    return pd.DataFrame(defaults)


def make_recs_df(data=None):
    defaults = {
        "app_id": [730, 570, 440, 730, 570],
        "user_id": [1, 1, 2, 3, 3],
        "is_recommended": [1, 1, 0, 1, 0],
        "hours": [1234.5, 567.8, 0.0, 42.0, 999.9],
        "date": ["2018-03-15", "2019-06-01", "2020-01-10", "2021-11-20", "2022-05-05"],
        "helpful": [12, 3, 0, 1, 0],
        "funny": [1, 0, 0, 2, 0],
    }
    if data:
        defaults.update(data)
    return pd.DataFrame(defaults)


class TestCleanGames:
    def test_removes_rows_with_null_title(self):
        from src.data.cleaner import clean_games

        df = make_games_df({"title": ["CS:GO", None, "TF2"]})
        result = clean_games(df)

        assert result["title"].notna().all()
        assert len(result) == 2

    def test_fills_null_rating_with_median(self):
        from src.data.cleaner import clean_games

        df = make_games_df({"rating": [0.87, np.nan, 0.93]})
        result = clean_games(df)

        assert result["rating"].notna().all()
        assert result.loc[1, "rating"] == pytest.approx(0.90, abs=0.01)

    def test_adds_release_year_and_month(self):
        from src.data.cleaner import clean_games

        df = make_games_df()
        result = clean_games(df)

        assert "release_year" in result.columns
        assert "release_month" in result.columns
        assert result.loc[0, "release_year"] == 2012
        assert result.loc[0, "release_month"] == 8

    def test_deduplicates_on_app_id(self):
        from src.data.cleaner import clean_games

        df = make_games_df({
            "app_id": [730, 730, 440],
            "title": ["CS:GO", "CS:GO dup", "TF2"],
        })
        result = clean_games(df)

        assert len(result) == 2
        assert result["app_id"].is_unique

    def test_detects_rating_out_of_range(self, capsys):
        from src.data.cleaner import clean_games

        df = make_games_df({"rating": [1.5, 0.82, -0.1]})
        clean_games(df)

        captured = capsys.readouterr()
        assert "rating" in captured.out.lower() or "评分" in captured.out


class TestCleanUsers:
    def test_fills_null_products_with_zero(self):
        from src.data.cleaner import clean_users

        df = make_users_df({"products": [42, np.nan, 0]})
        result = clean_users(df)

        assert result["products"].notna().all()
        assert result.loc[1, "products"] == 0

    def test_fills_null_reviews_with_zero(self):
        from src.data.cleaner import clean_users

        df = make_users_df({"reviews": [10, np.nan, np.nan]})
        result = clean_users(df)

        assert result["reviews"].notna().all()
        assert result.loc[1, "reviews"] == 0


class TestCleanRecommendations:
    def test_removes_rows_with_null_is_recommended(self):
        from src.data.cleaner import clean_recommendations

        df = make_recs_df({"is_recommended": [1, np.nan, 0, 1, np.nan]})
        result = clean_recommendations(df)

        assert result["is_recommended"].notna().all()
        assert len(result) == 3

    def test_converts_is_recommended_to_int(self):
        from src.data.cleaner import clean_recommendations

        df = make_recs_df({"is_recommended": [1.0, 1.0, 0.0, 1.0, 0.0]})
        result = clean_recommendations(df)

        assert result["is_recommended"].dtype == int

    def test_deduplicates_keeping_latest_date(self):
        from src.data.cleaner import clean_recommendations

        df = make_recs_df({
            "user_id": [1, 1, 2, 3, 3],
            "app_id": [730, 730, 440, 730, 570],
            "date": ["2018-03-15", "2019-06-01", "2020-01-10", "2021-11-20", "2022-05-05"],
        })
        result = clean_recommendations(df)

        dup_count = result.duplicated(subset=["user_id", "app_id"]).sum()
        assert dup_count == 0
        # For user 1, app 730, should keep the 2019 record
        kept = result[(result["user_id"] == 1) & (result["app_id"] == 730)]
        assert kept.iloc[0]["date"] == pd.Timestamp("2019-06-01")


class TestMergeMetadata:
    def test_left_merges_tags_and_genres(self):
        from src.data.cleaner import merge_metadata

        games = make_games_df()
        metadata = pd.DataFrame({
            "app_id": [730, 570, 440],
            "tags": [["FPS"], ["MOBA"], ["Classic"]],
            "genres": [["Action"], ["Strategy"], ["Action"]],
            "type": ["game", "game", "game"],
            "early_access": [0, 0, 0],
        })
        result = merge_metadata(games, metadata)

        assert "tags" in result.columns
        assert "genres" in result.columns
        assert result.loc[0, "tags"] == ["FPS"]

    def test_unmatched_games_get_none_tags(self):
        from src.data.cleaner import merge_metadata

        games = make_games_df()
        metadata = pd.DataFrame({
            "app_id": [730],
            "tags": [["FPS"]],
            "genres": [["Action"]],
            "type": ["game"],
            "early_access": [0],
        })
        result = merge_metadata(games, metadata)

        assert len(result) == 3
        assert result.loc[1, "tags"] is None or (isinstance(result.loc[1, "tags"], float) and np.isnan(result.loc[1, "tags"]))
