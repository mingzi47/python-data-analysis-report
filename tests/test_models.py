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


class TestSplitRecommendations:
    def test_splits_dataframe_by_user(self):
        """拆分后的训练集和测试集不应有用户重叠。"""
        import pandas as pd
        from src.models.trainer import split_recommendations

        recs = pd.DataFrame({
            "app_id": [730, 570, 440, 730, 570, 440, 730, 570],
            "user_id": [1, 1, 2, 3, 3, 4, 5, 5],
            "is_recommended": [1, 1, 0, 1, 0, 1, 0, 1],
            "hours": [10.0] * 8,
        })
        train, test = split_recommendations(recs, test_size=0.25, random_state=42)

        train_users = set(train["user_id"])
        test_users = set(test["user_id"])
        assert train_users.isdisjoint(test_users)
        assert len(train) > 0 and len(test) > 0

    def test_preserves_columns(self):
        import pandas as pd
        from src.models.trainer import split_recommendations

        recs = pd.DataFrame({
            "app_id": [730, 570, 440],
            "user_id": [1, 1, 2],
            "is_recommended": [1, 0, 1],
            "hours": [10.0, 20.0, 30.0],
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        })
        train, test = split_recommendations(recs, test_size=0.3, random_state=42)

        for col in recs.columns:
            assert col in train.columns
            assert col in test.columns


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

    def test_logistic_regression_accepts_random_state(self, sample_split_data):
        from src.models.trainer import split_data, train_logistic_regression

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model1 = train_logistic_regression(X_train, y_train, random_state=123)
        model2 = train_logistic_regression(X_train, y_train, random_state=123)
        # 相同 random_state 应产生相同系数
        assert np.allclose(model1.coef_, model2.coef_)

    def test_random_forest_accepts_random_state(self, sample_split_data):
        from src.models.trainer import split_data, train_random_forest

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model1 = train_random_forest(X_train, y_train, random_state=99)
        model2 = train_random_forest(X_train, y_train, random_state=99)
        assert np.allclose(model1.feature_importances_, model2.feature_importances_)

    def test_logistic_regression_has_class_weight(self, sample_split_data):
        from src.models.trainer import split_data, train_logistic_regression

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model = train_logistic_regression(X_train, y_train)
        assert model.class_weight == "balanced"

    def test_xgboost_has_scale_pos_weight(self, sample_split_data):
        from src.models.trainer import split_data, train_xgboost

        X, y, groups = sample_split_data
        X_train, X_test, y_train, y_test = split_data(X, y, groups)

        model = train_xgboost(X_train, y_train)
        assert model.scale_pos_weight is not None
        assert model.scale_pos_weight > 0


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
