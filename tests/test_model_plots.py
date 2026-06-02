import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def synthetic_data():
    """Generate synthetic classification data for plot testing."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6, random_state=42
    )
    # Use last 60 samples as test set
    X_train, X_test = X[:140], X[140:]
    y_train, y_test = y[:140], y[140:]
    return X_train, X_test, y_train, y_test, X, y


class TestPlotRocCurves:
    def test_creates_file(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_roc_curves

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        models = {"LogisticRegression": lr, "RandomForest": rf}
        save_path = str(tmp_path / "roc_curves.png")
        plot_roc_curves(models, X_test, y_test, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_skips_model_without_predict_proba(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_roc_curves

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)

        # A dummy model without predict_proba
        class DummyModel:
            def predict(self, X):
                return np.zeros(X.shape[0])

        models = {"LogisticRegression": lr, "Dummy": DummyModel()}
        save_path = str(tmp_path / "roc_curves_skip.png")
        # Should not crash; should create file with LR's curve only
        plot_roc_curves(models, X_test, y_test, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


class TestPlotConfusionMatrix:
    def test_creates_file(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_confusion_matrix

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        save_path = str(tmp_path / "confusion_matrix.png")
        plot_confusion_matrix(model, X_test, y_test, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


class TestPlotFeatureImportance15:
    def test_creates_file(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_feature_importance_15

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        importances = rf.feature_importances_.tolist()
        names = [f"feature_{i}" for i in range(10)]

        save_path = str(tmp_path / "feature_importance.png")
        plot_feature_importance_15(importances, names, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_fewer_than_15_features(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_feature_importance_15

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        # Only 3 features
        importances = [0.5, 0.3, 0.2]
        names = ["a", "b", "c"]

        save_path = str(tmp_path / "feature_importance_few.png")
        plot_feature_importance_15(importances, names, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


class TestPlotPartialDependence:
    def test_creates_file(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_partial_dependence

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        features = [0, 1, 2, 3]
        save_path = str(tmp_path / "partial_dependence.png")
        plot_partial_dependence(model, X_train, features, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


class TestPlotLearningCurve:
    def test_creates_file(self, synthetic_data, tmp_path):
        from src.visualization.model_plots import plot_learning_curve

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        model = LogisticRegression(max_iter=1000, random_state=42)

        save_path = str(tmp_path / "learning_curve.png")
        plot_learning_curve(model, X_train, y_train, save_path)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_creates_file_with_groups(self, synthetic_data, tmp_path):
        """传入 groups 时应使用 GroupShuffleSplit 而非普通 KFold。"""
        from src.visualization.model_plots import plot_learning_curve

        X_train, X_test, y_train, y_test, X, y = synthetic_data

        model = LogisticRegression(max_iter=1000, random_state=42)
        # 模拟用户分组
        groups = np.repeat(range(28), 5)[:140]

        save_path = str(tmp_path / "learning_curve_grouped.png")
        plot_learning_curve(model, X_train, y_train, save_path, groups=groups)

        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
