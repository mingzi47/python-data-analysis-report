"""Tests for src/analysis/helpers.py — concentration metrics, extreme users, activity tiers."""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# compute_concentration_metrics
# =============================================================================

class TestComputeConcentrationMetrics:
    """Tests for compute_concentration_metrics."""

    def test_basic_concentration(self):
        """Metrics should match hand-calculated values for a known distribution."""
        from src.analysis.helpers import compute_concentration_metrics

        recs = pd.DataFrame({
            "app_id": ([1] * 10) + ([2] * 5) + ([3] * 3) + ([4] * 2),
        })
        result = compute_concentration_metrics(recs)

        assert result["total_games"] == 4
        assert result["gini"] == pytest.approx(0.325, abs=1e-6)
        assert result["top_1pct"] == pytest.approx(0.50, abs=1e-6)
        assert result["top_5pct"] == pytest.approx(0.50, abs=1e-6)
        assert result["top_10pct"] == pytest.approx(0.50, abs=1e-6)
        assert result["top_20pct"] == pytest.approx(0.50, abs=1e-6)

    def test_perfect_equality(self):
        """Gini should be 0 when every game has the same number of recs."""
        from src.analysis.helpers import compute_concentration_metrics

        recs = pd.DataFrame({
            "app_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        })
        result = compute_concentration_metrics(recs)
        assert result["total_games"] == 4
        assert result["gini"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_inequality(self):
        """Gini approaches 1-1/n as one game dominates all recs."""
        from src.analysis.helpers import compute_concentration_metrics

        recs = pd.DataFrame({
            "app_id": [1] * 100 + [2],  # game 1 dominates
        })
        result = compute_concentration_metrics(recs)
        # For 2 games where one has all but 1 rec, it's nearly maximally unequal.
        # With n_games=2 and one game holding nearly everything: gini ≈ 1 - 1/2 = 0.5
        assert result["total_games"] == 2
        # game 1 has 100/101 ≈ 99% of recs; very close to perfect inequality for n=2
        assert result["gini"] > 0.48

    def test_empty_dataframe(self):
        """Empty input should return zeros and no crash."""
        from src.analysis.helpers import compute_concentration_metrics

        recs = pd.DataFrame({"app_id": pd.Series(dtype="int64")})
        result = compute_concentration_metrics(recs)
        assert result["total_games"] == 0
        assert result["gini"] == 0.0
        assert result["top_1pct"] == 0.0
        assert result["top_5pct"] == 0.0
        assert result["top_10pct"] == 0.0
        assert result["top_20pct"] == 0.0

    def test_single_row(self):
        """Single recommendation should work without error."""
        from src.analysis.helpers import compute_concentration_metrics

        recs = pd.DataFrame({"app_id": [42]})
        result = compute_concentration_metrics(recs)
        assert result["total_games"] == 1
        # One game with all recs: top X% all share 100%
        assert result["gini"] == pytest.approx(0.0, abs=1e-10)
        assert result["top_1pct"] == pytest.approx(1.0, abs=1e-10)

    def test_top_pct_with_many_games(self):
        """Top percentages should aggregate correctly with diverse counts."""
        from src.analysis.helpers import compute_concentration_metrics

        # 100 games: game 1 has 50 recs, games 2-10 have 5 each, games 11-100 have 0 each for simplicity
        # Actually let's do it simpler: 100 games, each with its index as count
        apps = []
        for i in range(1, 101):
            apps.extend([i] * i)  # game i has i recs
        recs = pd.DataFrame({"app_id": apps})

        result = compute_concentration_metrics(recs)
        # Total games = 100, total recs = 100*101/2 = 5050
        assert result["total_games"] == 100
        # Top 20% = top 20 games. These are games 81-100.
        # Sum of counts: 81+82+...+100 = (81+100)*20/2 = 181*10 = 1810
        # Percentage: 1810/5050 ≈ 0.3584
        expected_top20 = (sum(range(81, 101))) / (sum(range(1, 101)))
        assert result["top_20pct"] == pytest.approx(expected_top20, abs=1e-6)


# =============================================================================
# analyze_extreme_users
# =============================================================================

class TestAnalyzeExtremeUsers:
    """Tests for analyze_extreme_users."""

    def test_basic_extreme_users(self):
        """Correctly identify all-positive and all-negative users."""
        from src.analysis.helpers import analyze_extreme_users

        recs = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "is_recommended": [1, 1, 1, 0, 0, 0, 1, 0, 1],
        })
        result = analyze_extreme_users(recs)

        # all-positive: user 1
        assert result["all_positive"]["count"] == 1
        assert result["all_positive"]["pct"] == pytest.approx(100.0 / 3, abs=1e-6)
        assert result["all_positive"]["avg_reviews"] == 3.0

        # all-negative: user 2
        assert result["all_negative"]["count"] == 1
        assert result["all_negative"]["pct"] == pytest.approx(100.0 / 3, abs=1e-6)
        assert result["all_negative"]["avg_reviews"] == 3.0

    def test_review_buckets(self):
        """Users should be correctly placed in review-count buckets."""
        from src.analysis.helpers import analyze_extreme_users

        # User 1: 1 review, all positive
        # User 2: 3 reviews, all positive
        # User 3: 7 reviews, all positive
        # User 4: 15 reviews, all positive
        # User 5: 2 reviews, all negative
        recs = pd.DataFrame({
            "user_id": [1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5],
            "is_recommended": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        })
        result = analyze_extreme_users(recs)

        pos_buckets = result["all_positive"]["buckets"]
        assert pos_buckets["1"] == 1       # user 1
        assert pos_buckets["2-5"] == 1     # user 2
        assert pos_buckets["6-10"] == 1    # user 3
        assert pos_buckets["10+"] == 1     # user 4

        neg_buckets = result["all_negative"]["buckets"]
        assert neg_buckets["1"] == 0
        assert neg_buckets["2-5"] == 1     # user 5

    def test_no_extreme_users(self):
        """When no users are 100% or 0%, should return zeros."""
        from src.analysis.helpers import analyze_extreme_users

        recs = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "is_recommended": [1, 0, 1, 0],
        })
        result = analyze_extreme_users(recs)

        assert result["all_positive"]["count"] == 0
        assert result["all_positive"]["pct"] == 0.0
        assert result["all_positive"]["avg_reviews"] == 0.0
        assert result["all_negative"]["count"] == 0
        assert result["all_negative"]["pct"] == 0.0
        assert result["all_negative"]["avg_reviews"] == 0.0

    def test_empty_dataframe(self):
        """Empty input should return zeros without crashing."""
        from src.analysis.helpers import analyze_extreme_users

        recs = pd.DataFrame({"user_id": pd.Series(dtype="int64"),
                             "is_recommended": pd.Series(dtype="int64")})
        result = analyze_extreme_users(recs)

        for group in ["all_positive", "all_negative"]:
            assert result[group]["count"] == 0
            assert result[group]["pct"] == 0.0
            assert result[group]["avg_reviews"] == 0.0
            for bucket in ["1", "2-5", "6-10", "10+"]:
                assert result[group]["buckets"][bucket] == 0

    def test_all_users_extreme(self):
        """All users are extreme: either 100% or 0% recommend."""
        from src.analysis.helpers import analyze_extreme_users

        recs = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3],
            "is_recommended": [1, 1, 0, 0, 1, 0],  # user 3 is mixed
        })
        result = analyze_extreme_users(recs)

        # User 1: all positive, User 2: all negative
        assert result["all_positive"]["count"] == 1
        assert result["all_negative"]["count"] == 1
        # out of 3 total users
        assert result["all_positive"]["pct"] == pytest.approx(100.0 / 3, abs=1e-6)


# =============================================================================
# compute_user_activity_tiers
# =============================================================================

class TestComputeUserActivityTiers:
    """Tests for compute_user_activity_tiers."""

    def test_basic_tier_assignment(self):
        """Users should be classified into correct activity tiers."""
        from src.analysis.helpers import compute_user_activity_tiers

        users = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "products": [0, 5, 15, 30, 50, 80, 120, 200, 350, 500],
            "reviews": [0, 2, 10, 15, 30, 40, 60, 100, 150, 200],
        })
        result = compute_user_activity_tiers(users)

        assert "activity_tier" in result.columns
        assert list(result["user_id"]) == list(range(1, 11))

        # Check tier assignments using pandas quantile (default linear interpolation)
        q25 = users["products"].quantile(0.25)   # 18.75
        q75 = users["products"].quantile(0.75)   # 180.0
        q95 = users["products"].quantile(0.95)   # 432.5

        expected = []
        for p in users["products"]:
            if p <= q25:
                expected.append("low")
            elif p <= q75:
                expected.append("medium")
            elif p <= q95:
                expected.append("high")
            else:
                expected.append("extreme")

        assert list(result["activity_tier"]) == expected

    def test_preserves_original_columns(self):
        """Original columns should remain unchanged."""
        from src.analysis.helpers import compute_user_activity_tiers

        users = pd.DataFrame({
            "user_id": [1, 2, 3],
            "products": [10, 50, 200],
            "reviews": [5, 25, 100],
        })
        result = compute_user_activity_tiers(users)

        assert list(result["user_id"]) == [1, 2, 3]
        assert list(result["products"]) == [10, 50, 200]
        assert list(result["reviews"]) == [5, 25, 100]

    def test_all_same_products(self):
        """When all users have the same product count, everyone is 'medium'."""
        from src.analysis.helpers import compute_user_activity_tiers

        users = pd.DataFrame({
            "user_id": [1, 2, 3],
            "products": [50, 50, 50],
        })
        result = compute_user_activity_tiers(users)

        # All percentiles are 50, so 50 <= q25=50, 50 <= q75=50, so they'd be low-med boundary.
        # With inclusive bounds, they land in "low" (<= q25).
        # Let's verify with actual pandas behavior:
        # q25 = q75 = q95 = 50
        # products <= 50 → first match is low
        assert all(result["activity_tier"] == "low")

    def test_single_user(self):
        """Single user should not crash."""
        from src.analysis.helpers import compute_user_activity_tiers

        users = pd.DataFrame({
            "user_id": [1],
            "products": [42],
        })
        result = compute_user_activity_tiers(users)
        assert result.loc[0, "activity_tier"] == "low"

    def test_empty_dataframe(self):
        """Empty input should return empty DataFrame with tier column."""
        from src.analysis.helpers import compute_user_activity_tiers

        users = pd.DataFrame({"user_id": pd.Series(dtype="int64"),
                              "products": pd.Series(dtype="int64")})
        result = compute_user_activity_tiers(users)
        assert "activity_tier" in result.columns
        assert len(result) == 0
