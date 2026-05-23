"""Analysis computation helpers: concentration metrics, extreme users, activity tiers."""

import numpy as np
import pandas as pd


def compute_concentration_metrics(recommendations: pd.DataFrame) -> dict:
    """Compute recommendation concentration metrics for long-tail analysis.

    Parameters
    ----------
    recommendations : pd.DataFrame
        DataFrame with an ``app_id`` column (each row is one recommendation).

    Returns
    -------
    dict
        Keys: ``gini``, ``top_1pct``, ``top_5pct``, ``top_10pct``, ``top_20pct``,
        ``total_games``.
    """
    if recommendations.empty:
        return {
            "gini": 0.0,
            "top_1pct": 0.0,
            "top_5pct": 0.0,
            "top_10pct": 0.0,
            "top_20pct": 0.0,
            "total_games": 0,
        }

    counts = recommendations["app_id"].value_counts()
    total_games = len(counts)
    total_recs = counts.sum()

    if total_recs == 0 or total_games == 0:
        return {
            "gini": 0.0,
            "top_1pct": 0.0,
            "top_5pct": 0.0,
            "top_10pct": 0.0,
            "top_20pct": 0.0,
            "total_games": total_games,
        }

    # --- Gini coefficient ---
    sorted_counts = counts.sort_values().values  # ascending
    n = len(sorted_counts)
    # Gini = (2 * sum(i * y_i) - (n+1) * sum(y_i)) / (n * sum(y_i))
    # where i is the rank (1-indexed) and y_i is sorted ascending
    i_ranks = np.arange(1, n + 1)
    gini = (2.0 * np.sum(i_ranks * sorted_counts) - (n + 1) * sorted_counts.sum()) / (n * sorted_counts.sum())
    gini = max(0.0, float(gini))

    # --- Top-X% concentration ---
    # Sort descending so the largest games come first
    sorted_desc = counts.sort_values(ascending=False)
    cumulative = sorted_desc.cumsum()

    def top_pct(pct: float) -> float:
        n_top = max(1, int(np.ceil(total_games * pct / 100.0)))
        n_top = min(n_top, total_games)
        top_recs = cumulative.iloc[n_top - 1]
        return float(top_recs / total_recs)

    return {
        "gini": gini,
        "top_1pct": top_pct(1.0),
        "top_5pct": top_pct(5.0),
        "top_10pct": top_pct(10.0),
        "top_20pct": top_pct(20.0),
        "total_games": total_games,
    }


def analyze_extreme_users(recommendations: pd.DataFrame) -> dict:
    """Identify extreme users (100% recommend or 0% recommend).

    Parameters
    ----------
    recommendations : pd.DataFrame
        DataFrame with ``user_id`` and ``is_recommended`` columns.

    Returns
    -------
    dict
        Nested dict with ``all_positive`` and ``all_negative`` keys, each
        containing ``count``, ``pct``, ``avg_reviews``, and ``buckets``.
    """
    empty_result = {
        "all_positive": {
            "count": 0, "pct": 0.0, "avg_reviews": 0.0,
            "buckets": {"1": 0, "2-5": 0, "6-10": 0, "10+": 0},
        },
        "all_negative": {
            "count": 0, "pct": 0.0, "avg_reviews": 0.0,
            "buckets": {"1": 0, "2-5": 0, "6-10": 0, "10+": 0},
        },
    }

    if recommendations.empty:
        return empty_result

    # Per-user stats
    user_stats = recommendations.groupby("user_id").agg(
        review_count=("is_recommended", "count"),
        positive_count=("is_recommended", "sum"),
    )
    user_stats["recommend_rate"] = user_stats["positive_count"] / user_stats["review_count"]

    total_users = len(user_stats)
    if total_users == 0:
        return empty_result

    def _bucket(count: int) -> str:
        if count == 1:
            return "1"
        elif count <= 5:
            return "2-5"
        elif count <= 10:
            return "6-10"
        else:
            return "10+"

    def _stats_for_group(mask: pd.Series) -> dict:
        group = user_stats[mask]
        count = len(group)
        if count == 0:
            return {
                "count": 0,
                "pct": 0.0,
                "avg_reviews": 0.0,
                "buckets": {"1": 0, "2-5": 0, "6-10": 0, "10+": 0},
            }

        buckets = {"1": 0, "2-5": 0, "6-10": 0, "10+": 0}
        for _, row in group.iterrows():
            b = _bucket(int(row["review_count"]))
            buckets[b] += 1

        return {
            "count": count,
            "pct": float(count / total_users * 100),
            "avg_reviews": float(group["review_count"].mean()),
            "buckets": buckets,
        }

    return {
        "all_positive": _stats_for_group(user_stats["recommend_rate"] == 1.0),
        "all_negative": _stats_for_group(user_stats["recommend_rate"] == 0.0),
    }


def compute_user_activity_tiers(users: pd.DataFrame) -> pd.DataFrame:
    """Classify users into activity tiers based on ``products`` column percentiles.

    Tiers:
      - ``low``:     0th -- 25th percentile (inclusive)
      - ``medium``: 25th -- 75th percentile (exclusive low, inclusive high)
      - ``high``:   75th -- 95th percentile (exclusive low, inclusive high)
      - ``extreme``: above 95th percentile

    Parameters
    ----------
    users : pd.DataFrame
        DataFrame with ``user_id`` and ``products`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``activity_tier`` column (string).
    """
    result = users.copy()

    if result.empty:
        result["activity_tier"] = pd.Series(dtype="str")
        return result

    products = result["products"]
    q25 = products.quantile(0.25)
    q75 = products.quantile(0.75)
    q95 = products.quantile(0.95)

    conditions = [
        products <= q25,
        products <= q75,
        products <= q95,
    ]
    choices = ["low", "medium", "high"]
    result["activity_tier"] = np.select(conditions, choices, default="extreme")

    return result
