import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def _build_game_features(games: pd.DataFrame) -> pd.DataFrame:
    if "app_id" in games.columns and games.index.name != "app_id":
        games = games.set_index("app_id")

    today = pd.Timestamp.now()
    price = games.get("price_final", pd.Series(dtype=float))
    if price.empty and "price" in games.columns:
        price = games["price"]

    feats = pd.DataFrame(index=games.index)
    feats.index.name = "app_id"
    feats["price"] = price.fillna(0).astype(float)
    feats["is_free"] = (feats["price"] == 0).astype(int)
    rating_col = games.get("rating", pd.Series(dtype=float))
    if pd.api.types.is_numeric_dtype(rating_col):
        feats["rating"] = rating_col.fillna(rating_col.median()).astype(float)
    elif "positive_ratio" in games.columns:
        feats["rating"] = games["positive_ratio"].fillna(games["positive_ratio"].median()).astype(float) / 100.0
    else:
        feats["rating"] = 0.0

    release_year = games.get("release_year")
    if release_year is None and "date_release" in games.columns:
        release_year = pd.to_datetime(games["date_release"], errors="coerce").dt.year
    feats["years_since_release"] = today.year - release_year.fillna(today.year).astype(int)

    tags = games.get("tags")
    genres = games.get("genres")
    feats["num_tags"] = tags.apply(len) if tags is not None else 0
    if genres is not None and hasattr(genres, 'apply') and genres.apply(lambda x: len(x) if isinstance(x, list) else 0).sum() > 0:
        feats["num_genres"] = genres.apply(len)
    else:
        feats["num_genres"] = feats["num_tags"]

    if "early_access" in games.columns:
        feats["early_access"] = games["early_access"].fillna(0).astype(int)

    return feats


def _build_user_features(users: pd.DataFrame) -> pd.DataFrame:
    if "user_id" in users.columns and users.index.name != "user_id":
        users = users.set_index("user_id")

    feats = pd.DataFrame(index=users.index)
    feats.index.name = "user_id"
    feats["user_products_count"] = users["products"].fillna(0).astype(int)
    feats["user_reviews_count"] = users["reviews"].fillna(0).astype(int)
    feats["review_ratio"] = feats["user_reviews_count"] / feats["user_products_count"].clip(lower=1)
    return feats


def _build_interaction_features(recs: pd.DataFrame) -> pd.DataFrame:
    df = recs.copy()

    game_aggs = df.groupby("app_id").agg(
        game_review_count=("is_recommended", "count"),
        game_recommend_rate=("is_recommended", "mean"),
        game_avg_hours=("hours", "mean"),
    )
    user_aggs = df.groupby("user_id").agg(
        user_review_count=("is_recommended", "count"),
        user_recommend_rate=("is_recommended", "mean"),
        user_avg_hours=("hours", "mean"),
    )

    df = df.merge(game_aggs, on="app_id", how="left")
    df = df.merge(user_aggs, on="user_id", how="left")
    return df


def build_features(
    recommendations: pd.DataFrame,
    games: pd.DataFrame,
    users: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = recommendations["is_recommended"].astype(int)
    groups = recommendations["user_id"]

    game_feats = _build_game_features(games)
    user_feats = _build_user_features(users)
    inter_feats = _build_interaction_features(recommendations)

    X = inter_feats.merge(game_feats, left_on="app_id", right_index=True, how="left")
    X = X.merge(user_feats, left_on="user_id", right_index=True, how="left")

    # Drop identifier, target, and non-numeric columns from feature matrix
    drop_cols = ["app_id", "user_id", "is_recommended", "date"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    return X, y, groups


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), []),
        ],
        remainder="passthrough",
    )
