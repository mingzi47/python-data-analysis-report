import pytest
import pandas as pd
import numpy as np


def make_games_df():
    return pd.DataFrame({
        "app_id": [730, 570, 440],
        "title": ["CS:GO", "Dota 2", "TF2"],
        "date_release": pd.to_datetime(["2012-08-21", "2013-07-09", "2007-10-10"]),
        "release_year": [2012, 2013, 2007],
        "release_month": [8, 7, 10],
        "rating": [0.87, 0.82, 0.93],
        "price_final": [14.99, 0.0, 19.99],
        "tags": [["FPS", "Multiplayer"], ["MOBA", "Strategy"], ["FPS", "Classic"]],
        "genres": [["Action", "Free to Play"], ["Strategy", "Free to Play"], ["Action"]],
        "type": ["game", "game", "game"],
        "early_access": [0, 0, 0],
    }).set_index("app_id")


def make_users_df():
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "products": [42, 15, 0],
        "reviews": [10, 5, 0],
    }).set_index("user_id")


def make_recs_df():
    return pd.DataFrame({
        "app_id": [730, 570, 440, 730, 570],
        "user_id": [1, 1, 2, 3, 3],
        "is_recommended": [1, 1, 0, 1, 0],
        "hours": [1234.5, 567.8, 0.0, 42.0, 999.9],
    })


class TestGameFeatures:
    def test_extracts_numerical_features(self):
        from src.features.builder import _build_game_features

        games = make_games_df()
        result = _build_game_features(games)

        for col in ["price", "is_free", "rating", "years_since_release", "num_tags", "num_genres"]:
            assert col in result.columns
        assert result.loc[570, "is_free"] == 1
        assert result.loc[730, "is_free"] == 0
        assert result.loc[730, "num_tags"] == 2

    def test_index_is_app_id(self):
        from src.features.builder import _build_game_features

        games = make_games_df()
        result = _build_game_features(games)

        assert result.index.name == "app_id"
        assert 730 in result.index


class TestUserFeatures:
    def test_extracts_features_with_ratio(self):
        from src.features.builder import _build_user_features

        users = make_users_df()
        result = _build_user_features(users)

        for col in ["user_products_count", "user_reviews_count", "review_ratio"]:
            assert col in result.columns
        assert result.loc[1, "user_products_count"] == 42
        assert result.loc[1, "review_ratio"] == pytest.approx(10 / 42, abs=0.01)

    def test_avoids_division_by_zero(self):
        from src.features.builder import _build_user_features

        users = make_users_df()
        result = _build_user_features(users)

        assert result.loc[3, "review_ratio"] == 0.0


class TestInteractionFeatures:
    def test_computes_game_aggregates(self):
        from src.features.builder import _build_interaction_features

        recs = make_recs_df()
        result = _build_interaction_features(recs)

        assert "game_review_count" in result.columns
        assert "game_recommend_rate" in result.columns
        assert "game_avg_hours" in result.columns
        assert "user_review_count" in result.columns
        assert "user_recommend_rate" in result.columns
        assert "user_avg_hours" in result.columns

    def test_game_recommend_rate_in_range(self):
        from src.features.builder import _build_interaction_features

        recs = make_recs_df()
        result = _build_interaction_features(recs)

        assert result["game_recommend_rate"].between(0, 1).all()
        assert result["user_recommend_rate"].between(0, 1).all()


class TestBuildFeatures:
    def test_returns_X_y_groups(self):
        from src.features.builder import build_features

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(groups, pd.Series)
        assert len(X) == len(recs)
        assert len(y) == len(recs)
        assert y.isin([0, 1]).all()

    def test_y_matches_is_recommended(self):
        from src.features.builder import build_features

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)

        assert list(y) == [1, 1, 0, 1, 0]

    def test_groups_are_user_ids(self):
        from src.features.builder import build_features

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)

        assert list(groups) == [1, 1, 2, 3, 3]


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from src.features.builder import build_preprocessor
        from sklearn.compose import ColumnTransformer

        ct = build_preprocessor()

        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform_produces_no_nan(self):
        from src.features.builder import build_features, build_preprocessor

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)
        ct = build_preprocessor()
        X_scaled = ct.fit_transform(X)

        if hasattr(X_scaled, "toarray"):
            X_scaled = X_scaled.toarray()
        assert not np.isnan(X_scaled).any()
