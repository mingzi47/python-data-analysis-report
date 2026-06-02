import pandas as pd
from sklearn.dummy import DummyClassifier
from src.models.evaluator import compute_metrics


def evaluate_baselines(X_train, y_train, X_test, y_test, random_state: int = 42) -> dict[str, dict]:
    results = {}

    # Baseline 1: Uniform random
    dummy_uniform = DummyClassifier(strategy="uniform", random_state=random_state)
    dummy_uniform.fit(X_train, y_train)
    y_pred = dummy_uniform.predict(X_test)
    y_proba = dummy_uniform.predict_proba(X_test)[:, 1]
    results["DummyUniform"] = compute_metrics(y_test, y_pred, y_proba)

    # Baseline 2: Most frequent
    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(X_train, y_train)
    y_pred = dummy_mf.predict(X_test)
    y_proba = dummy_mf.predict_proba(X_test)[:, 1]
    results["DummyMostFrequent"] = compute_metrics(y_test, y_pred, y_proba)

    # Baseline 3: Game recommend rate rule
    game_recommend_col = "game_recommend_rate"
    if game_recommend_col in X_train.columns:
        y_pred = (X_test[game_recommend_col] > 0.5).astype(int)
        rule_scores = X_test[game_recommend_col].fillna(0.5)
    else:
        y_pred = dummy_mf.predict(X_test)
        rule_scores = pd.Series([0.5] * len(X_test))

    results["GameRateRule"] = compute_metrics(y_test, y_pred, rule_scores)

    return results
