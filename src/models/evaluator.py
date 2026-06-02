import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    y_proba : array-like or pd.Series
        Predicted probabilities for the positive class.

    Returns
    -------
    dict with accuracy, precision, recall, f1, roc_auc
    """
    y_proba_arr = y_proba.values if isinstance(y_proba, pd.Series) else y_proba
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba_arr),
    }


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df = df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    return df
