import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestPlotPriceDistribution:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_price_distribution

        df = pd.DataFrame({
            "app_id": [1, 2, 3, 4, 5],
            "title": ["Game A", "Game B", "Game C", "Game D", "Game E"],
            "release_year": [2015, 2018, 2020, 2019, 2017],
            "rating": [0.85, 0.72, 0.91, 0.68, 0.80],
            "price_final": [0.0, 9.99, 29.99, 0.0, 14.99],
            "tags": [["FPS"], ["RPG"], ["Action"], ["Free"], ["Indie"]],
            "genres": [["Action"], ["RPG"], ["Action"], ["Free to Play"], ["Indie"]],
        })
        save_path = tmp_path / "price_distribution.png"
        plot_price_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_free_games(self, tmp_path):
        from src.visualization.eda_plots import plot_price_distribution

        df = pd.DataFrame({
            "app_id": [1, 2, 3],
            "title": ["Free A", "Free B", "Paid"],
            "release_year": [2020, 2021, 2019],
            "rating": [0.8, 0.9, 0.7],
            "price_final": [0.0, 0.0, 19.99],
            "tags": [["a"], ["b"], ["c"]],
            "genres": [["a"], ["b"], ["c"]],
        })
        save_path = tmp_path / "price_dist.png"
        plot_price_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotRatingDistribution:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_rating_distribution

        df = pd.DataFrame({
            "app_id": list(range(10)),
            "title": [f"Game {i}" for i in range(10)],
            "release_year": np.repeat([2018, 2019], 5),
            "rating": np.linspace(0.3, 0.99, 10),
            "price_final": np.random.uniform(0, 60, 10),
            "tags": [["A"] for _ in range(10)],
            "genres": [["Action"] for _ in range(10)],
        })
        save_path = tmp_path / "rating_distribution.png"
        plot_rating_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotGenreBar:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_genre_bar

        df = pd.DataFrame({
            "app_id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "release_year": [2018, 2019, 2020, 2021],
            "rating": [0.8, 0.9, 0.7, 0.85],
            "price_final": [9.99, 0.0, 19.99, 4.99],
            "genres": [
                ["Action", "FPS"],
                ["Action", "Adventure"],
                ["RPG"],
                ["Strategy", "Simulation"],
            ],
        })
        save_path = tmp_path / "genre_bar.png"
        plot_genre_bar(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_empty_genres(self, tmp_path):
        from src.visualization.eda_plots import plot_genre_bar

        df = pd.DataFrame({
            "app_id": [1, 2],
            "title": ["A", "B"],
            "release_year": [2020, 2021],
            "rating": [0.8, 0.9],
            "price_final": [0.0, 9.99],
            "genres": [[], ["Action"]],
        })
        save_path = tmp_path / "genre_bar_empty.png"
        plot_genre_bar(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotReleaseTimeline:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_release_timeline

        df = pd.DataFrame({
            "app_id": list(range(10)),
            "title": [f"Game {i}" for i in range(10)],
            "release_year": [2015, 2015, 2016, 2017, 2017, 2017, 2018, 2019, 2019, 2020],
            "rating": np.linspace(0.5, 0.95, 10),
            "price_final": np.random.uniform(0, 50, 10),
        })
        save_path = tmp_path / "release_timeline.png"
        plot_release_timeline(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_single_year(self, tmp_path):
        from src.visualization.eda_plots import plot_release_timeline

        df = pd.DataFrame({
            "app_id": [1, 2, 3],
            "title": ["A", "B", "C"],
            "release_year": [2020, 2020, 2020],
            "rating": [0.7, 0.8, 0.9],
            "price_final": [0.0, 9.99, 19.99],
        })
        save_path = tmp_path / "release_timeline_single.png"
        plot_release_timeline(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotUserActivity:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_user_activity

        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6],
            "products": [0, 15, 42, 100, 200, 5],
            "reviews": [0, 5, 10, 50, 80, 2],
        })
        save_path = tmp_path / "user_activity.png"
        plot_user_activity(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_zeros(self, tmp_path):
        from src.visualization.eda_plots import plot_user_activity

        df = pd.DataFrame({
            "user_id": [1, 2],
            "products": [0, 0],
            "reviews": [0, 0],
        })
        save_path = tmp_path / "user_activity_zeros.png"
        plot_user_activity(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotLongTail:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_long_tail

        df = pd.DataFrame({
            "app_id": [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4],
            "user_id": [101, 102, 103, 201, 202, 301, 401, 402, 403, 404, 405],
            "is_recommended": [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
            "hours": [100, 200, 50, 300, 10, 500, 10, 20, 30, 40, 50],
        })
        save_path = tmp_path / "long_tail.png"
        plot_long_tail(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotCorrelationHeatmap:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_correlation_heatmap

        np.random.seed(42)
        df = pd.DataFrame({
            "rating": np.random.uniform(0, 1, 50),
            "price_final": np.random.uniform(0, 60, 50),
            "positive_ratio": np.random.uniform(0, 1, 50),
            "user_score": np.random.randint(1, 10, 50),
            "steam_deck": np.random.choice([0, 1], 50),
        })
        save_path = tmp_path / "correlation_heatmap.png"
        plot_correlation_heatmap(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_non_numeric_columns(self, tmp_path):
        from src.visualization.eda_plots import plot_correlation_heatmap

        np.random.seed(42)
        df = pd.DataFrame({
            "app_id": [1, 2, 3, 4, 5],
            "title": ["A", "B", "C", "D", "E"],
            "rating": [0.8, 0.7, 0.9, 0.6, 0.85],
            "price_final": [9.99, 0.0, 29.99, 14.99, 0.0],
            "genres": [["Action"], ["RPG"], ["Action"], ["Strategy"], ["Indie"]],
        })
        save_path = tmp_path / "correlation_non_numeric.png"
        plot_correlation_heatmap(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotUserActivityDistribution:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_user_activity_distribution

        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6],
            "products": [0, 15, 42, 100, 200, 5],
            "reviews": [0, 5, 10, 50, 80, 2],
        })
        save_path = tmp_path / "user_activity_distribution.png"
        plot_user_activity_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_zeros(self, tmp_path):
        from src.visualization.eda_plots import plot_user_activity_distribution

        df = pd.DataFrame({
            "user_id": [1, 2],
            "products": [0, 0],
            "reviews": [0, 0],
        })
        save_path = tmp_path / "user_activity_dist_zeros.png"
        plot_user_activity_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotUserRecommendRateDistribution:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_user_recommend_rate_distribution

        df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            "app_id": [10, 20, 30, 10, 20, 10, 20, 30, 40, 50],
            "is_recommended": [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
            "hours": [100, 50, 200, 300, 10, 500, 50, 100, 30, 20],
        })
        save_path = tmp_path / "recommend_rate_dist.png"
        plot_user_recommend_rate_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_single_user(self, tmp_path):
        from src.visualization.eda_plots import plot_user_recommend_rate_distribution

        df = pd.DataFrame({
            "user_id": [1, 1, 1],
            "app_id": [10, 20, 30],
            "is_recommended": [1, 0, 1],
            "hours": [100, 50, 200],
        })
        save_path = tmp_path / "recommend_rate_single.png"
        plot_user_recommend_rate_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_extra_columns(self, tmp_path):
        from src.visualization.eda_plots import plot_user_recommend_rate_distribution

        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "app_id": [10, 20, 30, 40],
            "is_recommended": [1, 0, 1, 0],
            "hours": [100, 50, 300, 10],
            "date": ["2020-01-01", "2020-02-01", "2019-06-15", "2019-07-20"],
            "review_text": ["Great!", "Bad...", "Amazing!", "Meh."],
        })
        save_path = tmp_path / "recommend_rate_extra.png"
        plot_user_recommend_rate_distribution(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotPurchaseVsReviews:
    def test_creates_file(self, tmp_path):
        from src.visualization.eda_plots import plot_purchase_vs_reviews

        df = pd.DataFrame({
            "user_id": list(range(1, 11)),
            "products": [1, 5, 10, 20, 30, 50, 80, 100, 150, 200],
            "reviews": [0, 2, 5, 10, 15, 25, 40, 60, 90, 120],
        })
        save_path = tmp_path / "purchase_vs_reviews.png"
        plot_purchase_vs_reviews(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_filters_zeros_for_log_scale(self, tmp_path):
        from src.visualization.eda_plots import plot_purchase_vs_reviews

        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4],
            "products": [0, 0, 5, 10],
            "reviews": [0, 1, 0, 5],
        })
        save_path = tmp_path / "purchase_vs_reviews_zeros.png"
        plot_purchase_vs_reviews(df, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0
