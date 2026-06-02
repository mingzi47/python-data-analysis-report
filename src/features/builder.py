import pandas as pd
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
    if release_year is None:
        # 没有任何发布日期信息，全部填充为今年（即"新游戏"）
        release_year = pd.Series(today.year, index=games.index)
    feats["years_since_release"] = today.year - release_year.fillna(today.year).astype(int)

    tags = games.get("tags")
    genres = games.get("genres")
    feats["num_tags"] = tags.apply(len) if tags is not None else 0
    # genres 仅在真实存在且包含列表数据时才使用，否则设为 0（而非回退到 num_tags 造成冗余）
    if genres is not None and hasattr(genres, 'apply') and genres.apply(lambda x: len(x) if isinstance(x, list) else 0).sum() > 0:
        feats["num_genres"] = genres.apply(len)
    else:
        feats["num_genres"] = 0

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


def fit_interaction_aggregates(recs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅在训练集上计算游戏/用户聚合特征，防止数据泄漏。

    聚合结果可传给 ``_build_interaction_features`` 和 ``build_features``
    用于训练集和测试集的特征构建。

    Parameters
    ----------
    recs : pd.DataFrame
        训练集推荐记录（包含 app_id, user_id, is_recommended, hours 列）

    Returns
    -------
    game_aggs : pd.DataFrame (index=app_id)
        game_review_count, game_recommend_rate, game_avg_hours
    user_aggs : pd.DataFrame (index=user_id)
        user_review_count, user_recommend_rate, user_avg_hours
    """
    game_aggs = recs.groupby("app_id").agg(
        game_review_count=("is_recommended", "count"),
        game_recommend_rate=("is_recommended", "mean"),
        game_avg_hours=("hours", "mean"),
    )
    user_aggs = recs.groupby("user_id").agg(
        user_review_count=("is_recommended", "count"),
        user_recommend_rate=("is_recommended", "mean"),
        user_avg_hours=("hours", "mean"),
    )
    return game_aggs, user_aggs


def _build_interaction_features(
    recs: pd.DataFrame,
    game_aggs: pd.DataFrame | None = None,
    user_aggs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = recs.copy()

    if game_aggs is None:
        game_aggs = df.groupby("app_id").agg(
            game_review_count=("is_recommended", "count"),
            game_recommend_rate=("is_recommended", "mean"),
            game_avg_hours=("hours", "mean"),
        )
    if user_aggs is None:
        user_aggs = df.groupby("user_id").agg(
            user_review_count=("is_recommended", "count"),
            user_recommend_rate=("is_recommended", "mean"),
            user_avg_hours=("hours", "mean"),
        )

    df = df.merge(game_aggs, on="app_id", how="left")
    df = df.merge(user_aggs, on="user_id", how="left")

    # 填充冷启动的游戏/用户（训练集中未出现）
    for col, default in [
        ("game_review_count", 0), ("game_recommend_rate", 0.5), ("game_avg_hours", 0.0),
        ("user_review_count", 0), ("user_recommend_rate", 0.5), ("user_avg_hours", 0.0),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(default)

    return df


def build_features(
    recommendations: pd.DataFrame,
    games: pd.DataFrame,
    users: pd.DataFrame,
    game_aggs: pd.DataFrame | None = None,
    user_aggs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """构建特征矩阵 X、目标变量 y、分组标签 groups。

    为防止数据泄漏，训练集和测试集的特征构建应分开进行：
    - **训练集**: 不传 game_aggs/user_aggs（交互聚合从训练集自身计算）
    - **测试集**: 传入 fit_interaction_aggregates(train_recs) 的结果

    Parameters
    ----------
    recommendations : pd.DataFrame
        推荐记录（训练集或测试集）
    games : pd.DataFrame
        游戏信息
    users : pd.DataFrame
        用户信息
    game_aggs : pd.DataFrame or None
        预计算的游戏聚合特征（来自训练集），None 时从 recommendations 计算
    user_aggs : pd.DataFrame or None
        预计算的用户聚合特征（来自训练集），None 时从 recommendations 计算

    Returns
    -------
    X : pd.DataFrame, y : pd.Series, groups : pd.Series
    """
    y = recommendations["is_recommended"].astype(int)
    groups = recommendations["user_id"]

    game_feats = _build_game_features(games)
    user_feats = _build_user_features(users)
    inter_feats = _build_interaction_features(
        recommendations, game_aggs=game_aggs, user_aggs=user_aggs
    )

    X = inter_feats.merge(game_feats, left_on="app_id", right_index=True, how="left")
    X = X.merge(user_feats, left_on="user_id", right_index=True, how="left")

    # Drop identifier, target, and non-numeric columns from feature matrix
    drop_cols = ["app_id", "user_id", "is_recommended", "date"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    return X, y, groups


def build_preprocessor() -> StandardScaler:
    """返回用于特征缩放的 StandardScaler。

    所有数值特征经 StandardScaler 变换为均值为 0、标准差为 1 的分布。
    这对 LogisticRegression 等基于梯度的模型至关重要。
    树模型（RF、XGBoost）不受缩放影响，使用缩放后的数据也无害。

    Returns
    -------
    StandardScaler
    """
    return StandardScaler()
