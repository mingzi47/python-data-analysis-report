import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def _build_game_features(games: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.now()
    price = games.get("price_final", pd.Series(dtype=float))
    if price.empty and "price" in games.columns:
        price = games["price"]

    feats = pd.DataFrame(index=games.index)
    feats.index.name = "app_id"
    feats["price"] = price.fillna(0).astype(float)
    feats["is_free"] = (feats["price"] == 0).astype(int)
    feats["rating"] = games.get("rating", pd.Series(dtype=float)).fillna(games["rating"].median() if "rating" in games.columns else 0).astype(float)

    release_year = games.get("release_year")
    if release_year is None and "date_release" in games.columns:
        release_year = pd.to_datetime(games["date_release"], errors="coerce").dt.year
    feats["years_since_release"] = today.year - release_year.fillna(today.year).astype(int)

    tags = games.get("tags")
    genres = games.get("genres")
    feats["num_tags"] = tags.apply(len) if tags is not None else 0
    feats["num_genres"] = genres.apply(len) if genres is not None else 0

    if "early_access" in games.columns:
        feats["early_access"] = games["early_access"].fillna(0).astype(int)

    return feats


def _build_user_features(users: pd.DataFrame) -> pd.DataFrame:
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

    X = inter_feats.merge(game_feats, on="app_id", how="left")
    X = X.merge(user_feats, on="user_id", how="left")

    # Drop identifier columns and target from feature matrix
    drop_cols = ["app_id", "user_id", "is_recommended"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    return X, y, groups


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), []),
        ],
        remainder="passthrough",
    )
