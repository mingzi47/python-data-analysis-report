import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture
def sample_split_data():
    """Generate sample data that mimics real features."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y, name="is_recommended")
    groups = pd.Series(np.repeat(range(40), 5)[:200], name="user_id")
    return X, y, groups


class TestSplitData:
    def test_returns_train_test_splits(self, sample_split_data):
        from src.models.trainer import split_data

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        assert len(X_train) > len(X_test)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_no_user_overlap(self, sample_split_data):
        from src.models.trainer import split_data

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        train_users = set(groups.iloc[X_train.index])
        test_users = set(groups.iloc[X_test.index])
        assert train_users.isdisjoint(test_users)


class TestEvaluateBaselines:
    def test_returns_dict_with_three_baselines(self, sample_split_data):
        from src.models.trainer import split_data
        from src.models.baseline import evaluate_baselines

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        results = evaluate_baselines(X_train, y_train, X_test, y_test)

        assert isinstance(results, dict)
        assert len(results) == 3

    def test_each_baseline_has_metrics(self, sample_split_data):
        from src.models.trainer import split_data
        from src.models.baseline import evaluate_baselines

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        results = evaluate_baselines(X_train, y_train, X_test, y_test)

        for name, metrics in results.items():
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                assert key in metrics, f"{name} missing {key}"


class TestTrainModels:
    def test_train_logistic_regression(self, sample_split_data):
        from src.models.trainer import split_data, train_logistic_regression

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model = train_logistic_regression(X_train, y_train)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_random_forest(self, sample_split_data):
        from src.models.trainer import split_data, train_random_forest

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model = train_random_forest(X_train, y_train)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_xgboost(self, sample_split_data):
        from src.models.trainer import split_data, train_xgboost

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model = train_xgboost(X_train, y_train)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestEvaluateModel:
    def test_returns_metrics_dict(self, sample_split_data):
        from src.models.trainer import split_data, train_logistic_regression
        from src.models.evaluator import evaluate_model

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)
        model = train_logistic_regression(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics
            assert 0 <= metrics[key] <= 1

    def test_metrics_are_reasonable(self, sample_split_data):
        from src.models.trainer import split_data, train_logistic_regression
        from src.models.evaluator import evaluate_model

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)
        model = train_logistic_regression(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["accuracy"] > 0.4


class TestCompareModels:
    def test_returns_dataframe(self, sample_split_data):
        from src.models.trainer import split_data, train_logistic_regression, train_random_forest
        from src.models.evaluator import evaluate_model, compare_models

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model1 = train_logistic_regression(X_train, y_train)
        model2 = train_random_forest(X_train, y_train)

        results = {
            "LogisticRegression": evaluate_model(model1, X_test, y_test),
            "RandomForest": evaluate_model(model2, X_test, y_test),
        }
        comparison = compare_models(results)

        assert isinstance(comparison, pd.DataFrame)
        assert "LogisticRegression" in comparison.index
        assert "RandomForest" in comparison.index
