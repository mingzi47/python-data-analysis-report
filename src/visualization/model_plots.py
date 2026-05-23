import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, roc_auc_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve


def plot_roc_curves(models: dict, X_test, y_test, save_path: str) -> None:
    """Plot ROC curves for multiple models on the same figure.

    Each model in `models` (dict of {name: model}) that has a predict_proba
    method gets its ROC curve drawn.  The legend labels include the AUC value.
    Models without predict_proba are silently skipped.

    Parameters
    ----------
    models : dict
        {model_name: fitted_model}
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    save_path : str
        Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        RocCurveDisplay.from_predictions(
            y_test, y_proba, name=f"{name} (AUC={auc:.3f})", ax=ax
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model, X_test, y_test, save_path: str) -> None:
    """Plot a row-normalised confusion matrix heatmap.

    Uses seaborn heatmap.  Each row sums to 1.

    Parameters
    ----------
    model : fitted classifier
    X_test : array-like
    y_test : array-like
    save_path : str
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Row-Normalised)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance_15(importances, names, save_path: str) -> None:
    """Horizontal bar chart of the top 15 feature importances.

    If fewer than 15 features are provided all of them are shown.

    Parameters
    ----------
    importances : list of float
        Feature importance values.
    names : list of str
        Feature names (same length as importances).
    save_path : str
    """
    # Sort by importance descending, take top 15
    pairs = sorted(
        zip(names, importances), key=lambda x: x[1], reverse=True
    )[:15]
    names_sorted = [p[0] for p in pairs][::-1]
    imps_sorted = [p[1] for p in pairs][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names_sorted, imps_sorted, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance Top 15")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_partial_dependence(model, X, features, save_path: str) -> None:
    """Plot partial dependence for the given feature indices.

    Uses sklearn.inspection.PartialDependenceDisplay.from_estimator.

    Parameters
    ----------
    model : fitted classifier
    X : array-like
        Training data.
    features : list of int
        Column indices to plot.
    save_path : str
    """
    display = PartialDependenceDisplay.from_estimator(
        model, X.astype(float), features, grid_resolution=20
    )
    display.figure_.tight_layout()
    display.figure_.savefig(save_path, dpi=150)
    plt.close(display.figure_)


def plot_learning_curve(model, X, y, save_path: str) -> None:
    """Plot a learning curve using sklearn.model_selection.learning_curve.

    Uses 3-fold CV, ROC AUC scoring and a small set of training sizes.

    Parameters
    ----------
    model : fitted or unfitted estimator
    X : array-like
    y : array-like
    save_path : str
    """
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring="roc_auc",
        train_sizes=train_sizes, n_jobs=1, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std,
        alpha=0.15, color="steelblue"
    )
    ax.fill_between(
        train_sizes_abs, test_mean - test_std, test_mean + test_std,
        alpha=0.15, color="orange"
    )
    ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue", label="Training Score")
    ax.plot(train_sizes_abs, test_mean, "o-", color="orange", label="Cross-Validation Score")

    ax.set_xlabel("Training Examples")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Learning Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
