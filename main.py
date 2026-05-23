"""游戏推荐数据分析 — 主流水线入口"""

import os
from pathlib import Path
import pandas as pd

from src.utils.config import Config
from src.data.loader import download_dataset, load_games, load_users, load_recommendations, load_metadata
from src.data.cleaner import clean_games, clean_users, clean_recommendations, merge_metadata
from src.features.builder import build_features, build_preprocessor
from src.models.trainer import split_data, train_logistic_regression, train_random_forest, train_xgboost
from src.models.baseline import evaluate_baselines
from src.models.evaluator import evaluate_model, compare_models
from src.analysis.helpers import (
    compute_concentration_metrics,
    analyze_extreme_users,
    compute_user_activity_tiers,
)
from src.visualization.eda_plots import (
    plot_price_distribution,
    plot_rating_distribution,
    plot_genre_bar,
    plot_release_timeline,
    plot_user_activity,
    plot_user_activity_distribution,
    plot_user_recommend_rate_distribution,
    plot_purchase_vs_reviews,
    plot_long_tail,
    plot_correlation_heatmap,
)
from src.visualization.model_plots import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance_15,
    plot_partial_dependence,
    plot_learning_curve,
)


def main():
    config = Config()
    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    # ============================================================
    # 阶段 1: 数据加载
    # ============================================================
    print("=" * 60)
    print("阶段 1: 数据加载")
    print("=" * 60)

    data_path = download_dataset()
    print(f"数据集路径: {data_path}")

    games_df = load_games(data_path / "games.csv")
    print(f"games: {games_df.shape}")

    users_df = load_users(data_path / "users.csv")
    print(f"users: {users_df.shape}")

    recs_df = load_recommendations(data_path / "recommendations.csv", nrows=config.sample_size)
    print(f"recommendations (sampled): {recs_df.shape}")

    metadata_df = load_metadata(data_path / "games_metadata.json")
    print(f"metadata: {metadata_df.shape}")

    # ============================================================
    # 阶段 2: 数据清洗
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 2: 数据清洗")
    print("=" * 60)

    games_df = clean_games(games_df)
    users_df = clean_users(users_df)
    recs_df = clean_recommendations(recs_df)
    games_df = merge_metadata(games_df, metadata_df)
    print(f"清洗后 games: {games_df.shape}, users: {users_df.shape}, recs: {recs_df.shape}")

    # ============================================================
    # 阶段 3-4: EDA 可视化
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 3-4: EDA 可视化")
    print("=" * 60)

    plot_price_distribution(games_df, str(config.figure_dir / "price_distribution.png"))
    plot_rating_distribution(games_df, str(config.figure_dir / "rating_distribution.png"))
    plot_genre_bar(games_df, str(config.figure_dir / "genre_bar.png"))
    plot_release_timeline(games_df, str(config.figure_dir / "release_timeline.png"))
    plot_user_activity(users_df, str(config.figure_dir / "user_activity.png"))
    plot_long_tail(recs_df, str(config.figure_dir / "long_tail.png"))
    plot_correlation_heatmap(games_df, str(config.figure_dir / "correlation_heatmap.png"))
    plot_user_activity_distribution(users_df, str(config.figure_dir / "user_activity_distribution.png"))
    plot_user_recommend_rate_distribution(recs_df, str(config.figure_dir / "user_recommend_rate.png"))
    plot_purchase_vs_reviews(users_df, str(config.figure_dir / "purchase_vs_reviews.png"))

    # 分析指标
    conc = compute_concentration_metrics(recs_df)
    print(f"推荐量集中度: Gini={conc['gini']:.3f}, Top1%={conc['top_1pct']:.1%}, Top5%={conc['top_5pct']:.1%}, Top20%={conc['top_20pct']:.1%}")

    extreme = analyze_extreme_users(recs_df)
    print(f"极端用户: 纯好评={extreme['all_positive_pct']:.1%}, 纯差评={extreme['all_negative_pct']:.1%}")

    users_df = compute_user_activity_tiers(users_df)
    print(f"活跃度分层: {users_df['activity_tier'].value_counts().to_dict()}")
    print("EDA 图表已生成")

    # ============================================================
    # 阶段 5: 特征工程
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 5: 特征工程")
    print("=" * 60)

    X, y, groups = build_features(recs_df, games_df, users_df)
    print(f"特征矩阵: {X.shape}, 目标变量: {y.shape}")

    # ============================================================
    # 阶段 6: 建模
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 6: 建模")
    print("=" * 60)

    X_train, X_test, y_train, y_test = split_data(X, y, groups, test_size=config.test_size)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 基线模型
    baseline_results = evaluate_baselines(X_train, y_train, X_test, y_test)
    print("\n基线模型结果:")
    for name, metrics in baseline_results.items():
        print(f"  {name}: Accuracy={metrics['accuracy']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")

    # 机器学习模型
    print("\n训练逻辑回归...")
    lr = train_logistic_regression(X_train, y_train)

    print("训练随机森林...")
    rf = train_random_forest(X_train, y_train)

    print("训练 XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    # ============================================================
    # 阶段 7: 模型评估
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 7: 模型评估")
    print("=" * 60)

    models = {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "XGBoost": xgb,
    }
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)

    all_results = {**baseline_results, **results}
    comparison = compare_models(all_results)
    print("\n模型对比:")
    print(comparison.to_string())

    comparison.to_csv(config.model_dir / "comparison.csv")
    print(f"\n对比表已保存: {config.model_dir / 'comparison.csv'}")

    # 模型可视化
    plot_roc_curves(models, X_test, y_test, str(config.figure_dir / "roc_curves.png"))
    best_model = rf
    plot_confusion_matrix(best_model, X_test, y_test, str(config.figure_dir / "confusion_matrix.png"))

    if hasattr(best_model, "feature_importances_"):
        feature_names = X.columns.tolist()
        plot_feature_importance_15(
            best_model.feature_importances_,
            feature_names,
            str(config.figure_dir / "feature_importance.png"),
        )

    plot_partial_dependence(best_model, X_train, X_train.columns[:4].tolist(),
                            str(config.figure_dir / "partial_dependence.png"))
    plot_learning_curve(best_model, X_train, y_train,
                        str(config.figure_dir / "learning_curve.png"))
    print("模型评估图表已生成")

    # ============================================================
    # 阶段 8: 结论
    # ============================================================
    print("\n" + "=" * 60)
    print("阶段 8: 分析总结")
    print("=" * 60)

    best_name = comparison["roc_auc"].idxmax()
    best_auc = comparison.loc[best_name, "roc_auc"]
    print(f"最佳模型: {best_name} (ROC-AUC: {best_auc:.4f})")

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        top_indices = importances.argsort()[-5:][::-1]
        print("\nTop 5 重要特征:")
        for i in top_indices:
            print(f"  {X.columns[i]}: {importances[i]:.4f}")

    print(f"\n所有图表已保存至: {config.figure_dir}")
    print(f"模型对比表已保存至: {config.model_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
