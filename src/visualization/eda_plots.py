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
from pathlib import Path


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
    """评分分布直方图 + KDE。"""
    ratings = games["rating"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ratings, bins=40, kde=True, color="steelblue", edgecolor="white", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Game Count")
    ax.set_title("Game Rating Distribution")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_genre_bar(games: pd.DataFrame, save_path: str) -> None:
    """Top 20 游戏类型条形图。genres 列是 list 类型，需要展开统计。"""
    # Explode the genres list column and count occurrences
    all_genres = games["genres"].explode().dropna()
    # Filter out empty strings if any
    all_genres = all_genres[all_genres != ""]
    genre_counts = all_genres.value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    genre_counts.plot(kind="barh", color="steelblue", edgecolor="white", ax=ax)
    ax.set_xlabel("Game Count")
    ax.set_ylabel("Genre")
    ax.set_title("Top 20 Game Genres")
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
