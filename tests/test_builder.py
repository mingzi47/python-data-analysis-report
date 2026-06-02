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

    def test_handles_missing_release_date(self):
        """没有 release_year 和 date_release 时不应崩溃。"""
        from src.features.builder import _build_game_features

        games = pd.DataFrame({
            "app_id": [730, 570],
            "title": ["Game A", "Game B"],
            "rating": [0.9, 0.8],
            "price_final": [0.0, 9.99],
            "tags": [["Action"], ["RPG"]],
        }).set_index("app_id")

        result = _build_game_features(games)
        assert "years_since_release" in result.columns
        # 所有游戏都应使用当前年份（即 years_since_release ≈ 0）
        assert (result["years_since_release"] >= 0).all()

    def test_num_genres_is_zero_when_no_genres(self):
        """没有 genres 列时，num_genres 应为 0 而非回退到 num_tags。"""
        from src.features.builder import _build_game_features

        games = pd.DataFrame({
            "app_id": [730, 570],
            "title": ["Game A", "Game B"],
            "rating": [0.9, 0.8],
            "price_final": [0.0, 9.99],
            "tags": [["Action", "FPS"], ["RPG"]],
            "release_year": [2020, 2019],
        }).set_index("app_id")

        result = _build_game_features(games)
        assert result.loc[730, "num_tags"] == 2
        assert result.loc[730, "num_genres"] == 0
        # 两列不应冗余相同
        assert not (result["num_tags"] == result["num_genres"]).all()

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


class TestFitInteractionAggregates:
    def test_returns_game_and_user_aggs(self):
        from src.features.builder import fit_interaction_aggregates

        recs = make_recs_df()
        game_aggs, user_aggs = fit_interaction_aggregates(recs)

        assert "game_review_count" in game_aggs.columns
        assert "game_recommend_rate" in game_aggs.columns
        assert "game_avg_hours" in game_aggs.columns
        assert "user_review_count" in user_aggs.columns
        assert "user_recommend_rate" in user_aggs.columns
        assert "user_avg_hours" in user_aggs.columns

    def test_aggregates_are_in_valid_range(self):
        from src.features.builder import fit_interaction_aggregates

        recs = make_recs_df()
        game_aggs, user_aggs = fit_interaction_aggregates(recs)

        assert game_aggs["game_recommend_rate"].between(0, 1).all()
        assert user_aggs["user_recommend_rate"].between(0, 1).all()
        assert (game_aggs["game_review_count"] > 0).all()


class TestInteractionFeaturesWithPrecomputedAggs:
    def test_uses_provided_aggregates(self):
        from src.features.builder import _build_interaction_features, fit_interaction_aggregates

        recs = make_recs_df()
        game_aggs, user_aggs = fit_interaction_aggregates(recs)

        # Modify an aggregate to verify it's used (not recomputed)
        game_aggs_modified = game_aggs.copy()
        game_aggs_modified["game_recommend_rate"] = 0.999

        result = _build_interaction_features(recs, game_aggs=game_aggs_modified, user_aggs=user_aggs)
        # The first row's app_id=730 should have game_recommend_rate=0.999
        assert (result.loc[result["app_id"] == 730, "game_recommend_rate"] == 0.999).all()

    def test_no_data_leakage_in_test_set(self):
        """游戏同时出现在训练集和测试集时，测试集的聚合特征必须仅用训练集计算。"""
        from src.features.builder import fit_interaction_aggregates, _build_interaction_features

        # 训练集: app_id=730 有2条记录，推荐率=1.0 (2/2)
        train_recs = pd.DataFrame({
            "app_id": [730, 730, 570, 570, 440],
            "user_id": [1, 2, 1, 3, 2],
            "is_recommended": [1, 1, 1, 0, 0],
            "hours": [100.0, 200.0, 50.0, 30.0, 10.0],
        })
        # 测试集: app_id=730 有1条记录，推荐=0
        test_recs = pd.DataFrame({
            "app_id": [730],
            "user_id": [4],
            "is_recommended": [0],
            "hours": [5.0],
        })

        game_aggs, user_aggs = fit_interaction_aggregates(train_recs)
        result = _build_interaction_features(test_recs, game_aggs=game_aggs, user_aggs=user_aggs)

        # 测试集中 app_id=730 的 game_recommend_rate 应该来自训练集(1.0)，而非包含自身后的值
        assert result.loc[0, "game_recommend_rate"] == 1.0
        # 冷启动用户的聚合特征应填充默认值
        assert result.loc[0, "user_recommend_rate"] == 0.5
        assert result.loc[0, "user_review_count"] == 0


class TestBuildFeaturesWithPrecomputedAggs:
    def test_build_features_accepts_aggs(self):
        from src.features.builder import build_features, fit_interaction_aggregates

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        game_aggs, user_aggs = fit_interaction_aggregates(recs)
        X, y, groups = build_features(recs, games, users,
                                      game_aggs=game_aggs, user_aggs=user_aggs)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(recs)
        assert "game_recommend_rate" in X.columns
        assert "user_recommend_rate" in X.columns


class TestBuildPreprocessor:
    def test_returns_scaler(self):
        from src.features.builder import build_preprocessor
        from sklearn.preprocessing import StandardScaler

        ct = build_preprocessor()

        assert isinstance(ct, StandardScaler)

    def test_fit_transform_produces_no_nan(self):
        from src.features.builder import build_features, build_preprocessor

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)
        scaler = build_preprocessor()
        X_scaled = scaler.fit_transform(X)

        if hasattr(X_scaled, "toarray"):
            X_scaled = X_scaled.toarray()
        assert not np.isnan(X_scaled).any()

    def test_scaling_normalizes_features(self):
        """缩放后非常量列应接近均值为0、标准差为1。"""
        from src.features.builder import build_features, build_preprocessor

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)
        scaler = build_preprocessor()
        X_scaled = scaler.fit_transform(X)

        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0)
        # 过滤常量列（原始方差为 0 的列缩放后 std 保持为 0）
        non_constant = stds > 0
        # 非常量列的均值应接近 0
        assert np.allclose(means[non_constant], 0, atol=1e-7)
        # 非常量列的标准差应接近 1
        assert np.allclose(stds[non_constant], 1, atol=1e-7)

    def test_transform_uses_fit_statistics(self):
        """transform 应使用 fit 时的统计量，而非重新计算。"""
        from src.features.builder import build_features, build_preprocessor

        recs = make_recs_df()
        games = make_games_df()
        users = make_users_df()

        X, y, groups = build_features(recs, games, users)
        scaler = build_preprocessor()

        # Fit on first half, transform on second half
        half = len(X) // 2
        X_train = X.iloc[:half]
        X_test = X.iloc[half:]

        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 测试集的缩放后均值不应为 0（因为使用的是训练集统计量）
        test_means = np.mean(X_test_scaled, axis=0)
        assert not np.allclose(test_means, 0, atol=1e-7)
