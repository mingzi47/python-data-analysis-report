from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def split_data(X, y, groups, test_size=0.2, random_state=42):
    """按用户分组拆分已构建的特征矩阵（供测试/快速原型使用）。

    注意：主流水线使用 split_recommendations 在特征构建前拆分推荐记录，
    以防止交互特征的数据泄漏。本函数仅适用于已构建好特征的场景（如单元测试）。
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def split_recommendations(recs, test_size=0.2, random_state=42):
    """在特征工程之前，按用户分组拆分推荐记录。

    这是防止数据泄漏的关键步骤 — 必须在构建交互特征之前拆分，
    确保测试集的聚合特征仅从训练集计算。

    Parameters
    ----------
    recs : pd.DataFrame
        完整推荐记录（必须包含 user_id 列）
    test_size : float
        测试集比例 (default 0.2)
    random_state : int
        随机种子

    Returns
    -------
    train_recs : pd.DataFrame, test_recs : pd.DataFrame
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(recs, groups=recs["user_id"]))
    return recs.iloc[train_idx], recs.iloc[test_idx]


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(
        max_iter=2000, random_state=random_state, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42, groups=None):
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    # 构建早停验证集：传入 groups 时使用分组拆分，否则简单留出 15%
    if groups is not None and len(X_train) >= 20:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
        train_idx, val_idx = next(gss.split(X_train, groups=groups))
        eval_set = [(X_train.iloc[train_idx], y_train.iloc[train_idx]),
                     (X_train.iloc[val_idx], y_train.iloc[val_idx])]
    elif len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=random_state
        )
        eval_set = [(X_tr, y_tr), (X_val, y_val)]
    else:
        eval_set = [(X_train, y_train)]

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=random_state, eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=10,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model
