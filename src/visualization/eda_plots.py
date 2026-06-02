"""EDA visualization functions.

Each function takes a DataFrame and a save_path, creates a plot,
and saves it to disk. All functions use plt.close() to avoid memory issues.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def plot_price_distribution(games: pd.DataFrame, save_path: str) -> None:
    """价格分布直方图（log轴）。"""
    prices = games["price_final"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(prices, bins=50, edgecolor="white", color="steelblue", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Game Count (log scale)")
    ax.set_title("Game Price Distribution")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rating_distribution(games: pd.DataFrame, save_path: str) -> None:
    """评分分布：数值型用直方图+KDE，文本型用柱状图。"""
    rating_series = games["rating"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(rating_series):
        sns.histplot(rating_series, bins=40, kde=True, color="steelblue", edgecolor="white", ax=ax)
        ax.set_xlabel("Rating")
    else:
        counts = rating_series.value_counts()
        sns.barplot(x=counts.values, y=counts.index, color="steelblue", ax=ax)
        ax.set_xlabel("Game Count")
        ax.set_ylabel("Rating Category")
    ax.set_ylabel("Game Count")
    ax.set_title("Game Rating Distribution")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_genre_bar(games: pd.DataFrame, save_path: str) -> None:
    """Top 20 游戏类型/标签条形图。优先用 genres，为空时回退到 tags。"""
    # Use genres if non-empty, otherwise fall back to tags (actual data only has tags)
    if "genres" in games.columns:
        all_genres = games["genres"].explode().dropna()
        all_genres = all_genres[all_genres != ""]
    else:
        all_genres = pd.Series(dtype=str)

    if len(all_genres) == 0 and "tags" in games.columns:
        all_genres = games["tags"].explode().dropna()
        all_genres = all_genres[all_genres != ""]

    genre_counts = all_genres.value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    genre_counts.plot(kind="barh", color="steelblue", edgecolor="white", ax=ax)
    ax.set_xlabel("Game Count")
    ax.set_ylabel("Genre / Tag")
    ax.set_title("Top 20 Game Genres / Tags")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_release_timeline(games: pd.DataFrame, save_path: str) -> None:
    """年度游戏发布量折线图。使用 release_year 列。"""
    yearly = games["release_year"].dropna().astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly.index, yearly.values, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Games Released")
    ax.set_title("Game Releases by Year")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_user_activity(users: pd.DataFrame, save_path: str) -> None:
    """用户评论数/购买数分布散点图。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        users["products"], users["reviews"],
        alpha=0.5, color="steelblue", edgecolors="none", s=20
    )
    ax.set_xlabel("Products Owned")
    ax.set_ylabel("Reviews Written")
    ax.set_title("User Activity: Reviews vs Products Owned")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_long_tail(recommendations: pd.DataFrame, save_path: str) -> None:
    """游戏推荐量长尾分布。按 app_id 统计推荐数，降序排列。"""
    rec_counts = recommendations["app_id"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(rec_counts) + 1), rec_counts.values, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Game Rank (by recommendation count)")
    ax.set_ylabel("Number of Recommendations")
    ax.set_title("Long Tail Distribution of Game Recommendations")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """数值变量相关性热力图。"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Correlation Heatmap of Numeric Variables")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_user_activity_distribution(users: pd.DataFrame, save_path: str) -> None:
    """用户产品数和评论数分布直方图（双子图，log y轴）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(users["products"].dropna(), bins=50, edgecolor="white", color="steelblue", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_xlabel("Products Owned")
    ax1.set_ylabel("User Count (log scale)")
    ax1.set_title("User Products Distribution")
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax2.hist(users["reviews"].dropna(), bins=50, edgecolor="white", color="steelblue", alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xlabel("Reviews Written")
    ax2.set_ylabel("User Count (log scale)")
    ax2.set_title("User Reviews Distribution")
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_user_recommend_rate_distribution(recommendations: pd.DataFrame, save_path: str) -> None:
    """用户推荐率分布直方图，标注0.5分界线和总体均值。"""
    user_rates = recommendations.groupby("user_id")["is_recommended"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(user_rates, bins=40, edgecolor="white", color="steelblue", alpha=0.85)

    overall_mean = user_rates.mean()
    ax.axvline(x=0.5, color="orange", linestyle="--", linewidth=2, label="Neutral (0.5)")
    ax.axvline(x=overall_mean, color="red", linestyle="--", linewidth=2,
               label=f"Overall Mean ({overall_mean:.3f})")

    ax.set_xlabel("Recommend Rate")
    ax.set_ylabel("User Count")
    ax.set_title("User Recommend Rate Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_purchase_vs_reviews(users: pd.DataFrame, save_path: str) -> None:
    """产品数 vs 评论数散点图（log-log轴），含Spearman相关系数和y=x参考线。"""
    # Filter out zeros for log scale and meaningful correlation
    mask = (users["products"] > 0) & (users["reviews"] > 0)
    products = users.loc[mask, "products"]
    reviews = users.loc[mask, "reviews"]

    corr, _ = spearmanr(products, reviews)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(products, reviews, alpha=0.5, color="steelblue", edgecolors="none", s=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Products Owned (log scale)")
    ax.set_ylabel("Reviews Written (log scale)")
    ax.set_title(f"Products vs Reviews (Spearman r = {corr:.3f})")
    ax.grid(True, alpha=0.3)

    # y=x reference line (use two points instead of slope for log-log compatibility)
    ax.axline((1, 1), (100, 100), color="gray", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
