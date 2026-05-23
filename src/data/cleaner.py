import pandas as pd


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove rows with null title
    before = len(df)
    df = df.dropna(subset=["title"])
    if before != len(df):
        print(f"games: 删除 title 缺失 {before - len(df)} 行")

    # Handle rating: numeric (test fixtures) vs text (real Steam data)
    rating_is_numeric = pd.api.types.is_numeric_dtype(df["rating"])

    if rating_is_numeric:
        if df["rating"].isnull().any():
            df["rating"] = df["rating"].fillna(df["rating"].median())
        outliers = df[(df["rating"] < 0) | (df["rating"] > 1)]
        if len(outliers) > 0:
            print(f"games: 评分异常值 {len(outliers)} 条 (rating 不在 [0, 1] 范围内)")
    else:
        if df["rating"].isnull().any():
            df["rating"] = df["rating"].fillna("Unknown")
        print(f"games: rating 为文本类型（{df['rating'].nunique()} 个不同值），已跳过数值清洗")

    # Fill null date_release with median year
    if df["date_release"].isnull().any():
        median_date = pd.to_datetime(df["date_release"], errors="coerce").median()
        df["date_release"] = df["date_release"].fillna(str(median_date))

    # Convert date_release to datetime and extract year/month
    df["date_release"] = pd.to_datetime(df["date_release"], errors="coerce")
    df["release_year"] = df["date_release"].dt.year
    df["release_month"] = df["date_release"].dt.month

    # Deduplicate on app_id
    before = len(df)
    df = df.drop_duplicates(subset=["app_id"], keep="first")
    if before != len(df):
        print(f"games: app_id 去重删除 {before - len(df)} 行")

    return df


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df["products"].isnull().any():
        df["products"] = df["products"].fillna(0)
    if df["reviews"].isnull().any():
        df["reviews"] = df["reviews"].fillna(0)

    return df


def clean_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove rows with null is_recommended
    before = len(df)
    df = df.dropna(subset=["is_recommended"])
    if before != len(df):
        print(f"recommendations: 删除 is_recommended 缺失 {before - len(df)} 行")

    # Convert types
    df["is_recommended"] = df["is_recommended"].astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Deduplicate on (user_id, app_id) keeping latest date
    before = len(df)
    df = df.sort_values("date", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "app_id"], keep="first")
    if before != len(df):
        print(f"recommendations: (user_id, app_id) 去重删除 {before - len(df)} 行")

    return df


def merge_metadata(games: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return games.merge(metadata, on="app_id", how="left")
