#!/usr/bin/env bash
# ============================================================
# 项目运行脚本 — 交互式管理数据流水线
#
# 用法:
#   ./scripts/run.sh             交互模式（显示菜单）
#   ./scripts/run.sh full        完整流水线
#   ./scripts/run.sh quick       快速运行（小样本）
#   ./scripts/run.sh eda         仅 EDA（阶段 1-4）
#   ./scripts/run.sh ml          仅建模（阶段 5-8）
#   ./scripts/run.sh clean       清理输出目录
#   ./scripts/run.sh help        显示帮助
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="uv run python"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

check_venv() {
    if ! command -v uv &>/dev/null || [ ! -d "$PROJECT_DIR/.venv" ]; then
        echo -e "${RED}错误: 未找到 uv 或虚拟环境。请先运行: uv sync${NC}"
        exit 1
    fi
}

run_full() {
    print_header "完整流水线运行"
    echo -e "样本量: ${YELLOW}500,000${NC} (默认)"
    echo -e "预计耗时: ${YELLOW}3-5 分钟${NC}"
    echo ""
    $PYTHON main.py
    echo ""
    echo -e "${GREEN}✓ 流水线执行完成${NC}"
}

run_quick() {
    print_header "快速运行（小样本）"
    echo -e "样本量: ${YELLOW}50,000${NC}"
    echo -e "预计耗时: ${YELLOW}30-60 秒${NC}"
    echo ""
    $PYTHON -c "
from src.utils.config import Config
import main
config = Config(sample_size=50_000)
import src.utils.config as cfg
original = cfg.Config
cfg.Config = lambda: config
main.main()
"
    echo ""
    echo -e "${GREEN}✓ 快速运行完成${NC}"
}

run_eda() {
    print_header "EDA 阶段运行（阶段 1-4）"
    echo -e "包含: 数据加载 → 清洗 → EDA 可视化 → 分析指标"
    echo -e "预计耗时: ${YELLOW}1-2 分钟${NC}"
    echo ""
    $PYTHON -c "
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.data.loader import download_dataset, load_games, load_users, load_recommendations, load_metadata
from src.data.cleaner import clean_games, clean_users, clean_recommendations, merge_metadata
from src.analysis.helpers import compute_concentration_metrics, analyze_extreme_users, compute_user_activity_tiers
from src.visualization.eda_plots import (
    plot_price_distribution, plot_rating_distribution, plot_genre_bar,
    plot_release_timeline, plot_user_activity, plot_user_activity_distribution,
    plot_user_recommend_rate_distribution, plot_purchase_vs_reviews,
    plot_long_tail, plot_correlation_heatmap,
)

config = Config()
os.makedirs(config.figure_dir, exist_ok=True)
os.makedirs(config.model_dir, exist_ok=True)

data_path = download_dataset()
print(f'数据集路径: {data_path}')

games_df = load_games(data_path / 'games.csv')
users_df = load_users(data_path / 'users.csv')
recs_df = load_recommendations(data_path / 'recommendations.csv', nrows=config.sample_size)
metadata_df = load_metadata(data_path / 'games_metadata.json')

games_df = clean_games(games_df)
users_df = clean_users(users_df)
recs_df = clean_recommendations(recs_df)
games_df = merge_metadata(games_df, metadata_df)

print('生成 EDA 图表...')
plot_price_distribution(games_df, str(config.figure_dir / 'price_distribution.png'))
plot_rating_distribution(games_df, str(config.figure_dir / 'rating_distribution.png'))
plot_genre_bar(games_df, str(config.figure_dir / 'genre_bar.png'))
plot_release_timeline(games_df, str(config.figure_dir / 'release_timeline.png'))
plot_user_activity(users_df, str(config.figure_dir / 'user_activity.png'))
plot_long_tail(recs_df, str(config.figure_dir / 'long_tail.png'))
plot_correlation_heatmap(games_df, str(config.figure_dir / 'correlation_heatmap.png'))
plot_user_activity_distribution(users_df, str(config.figure_dir / 'user_activity_distribution.png'))
plot_user_recommend_rate_distribution(recs_df, str(config.figure_dir / 'user_recommend_rate.png'))
plot_purchase_vs_reviews(users_df, str(config.figure_dir / 'purchase_vs_reviews.png'))

conc = compute_concentration_metrics(recs_df)
print(f\"推荐量集中度: Gini={conc['gini']:.3f}, Top1%={conc['top_1pct']:.1%}, Top5%={conc['top_5pct']:.1%}, Top20%={conc['top_20pct']:.1%}\")

extreme = analyze_extreme_users(recs_df)
print(f\"极端用户: 纯好评={extreme['all_positive']['pct']:.1f}%, 纯差评={extreme['all_negative']['pct']:.1f}%\")

users_df = compute_user_activity_tiers(users_df)
print(f\"活跃度分层: {users_df['activity_tier'].value_counts().to_dict()}\")
print('EDA 图表已生成')
"
    echo ""
    echo -e "${GREEN}✓ EDA 阶段完成${NC}"
}

run_ml() {
    print_header "建模阶段运行（阶段 5-8）"
    echo -e "前置条件: 已完成数据加载和 EDA"
    echo -e "包含: 特征工程 → 建模 → 评估 → 结论"
    echo -e "预计耗时: ${YELLOW}2-4 分钟${NC}"
    echo ""
    $PYTHON -c "
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.data.loader import download_dataset, load_games, load_users, load_recommendations, load_metadata
from src.data.cleaner import clean_games, clean_users, clean_recommendations, merge_metadata
from src.features.builder import build_features
from src.models.trainer import split_data, train_logistic_regression, train_random_forest, train_xgboost
from src.models.baseline import evaluate_baselines
from src.models.evaluator import evaluate_model, compare_models
from src.visualization.model_plots import (
    plot_roc_curves, plot_confusion_matrix, plot_feature_importance_15,
    plot_partial_dependence, plot_learning_curve,
)

config = Config()
os.makedirs(config.figure_dir, exist_ok=True)
os.makedirs(config.model_dir, exist_ok=True)

data_path = download_dataset()

games_df = load_games(data_path / 'games.csv')
users_df = load_users(data_path / 'users.csv')
recs_df = load_recommendations(data_path / 'recommendations.csv', nrows=config.sample_size)
metadata_df = load_metadata(data_path / 'games_metadata.json')

games_df = clean_games(games_df)
users_df = clean_users(users_df)
recs_df = clean_recommendations(recs_df)
games_df = merge_metadata(games_df, metadata_df)

X, y, groups = build_features(recs_df, games_df, users_df)
X_train, X_test, y_train, y_test = split_data(X, y, groups, test_size=config.test_size)

baseline_results = evaluate_baselines(X_train, y_train, X_test, y_test)
print('基线模型:')
for name, m in baseline_results.items():
    print(f'  {name}: Accuracy={m[\"accuracy\"]:.3f}, ROC-AUC={m[\"roc_auc\"]:.3f}')

print('训练模型...')
lr = train_logistic_regression(X_train, y_train)
rf = train_random_forest(X_train, y_train)
xgb = train_xgboost(X_train, y_train)

models = {'LogisticRegression': lr, 'RandomForest': rf, 'XGBoost': xgb}
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_test, y_test)

all_results = {**baseline_results, **results}
comparison = compare_models(all_results)
print(comparison.to_string())
comparison.to_csv(config.model_dir / 'comparison.csv')

plot_roc_curves(models, X_test, y_test, str(config.figure_dir / 'roc_curves.png'))
plot_confusion_matrix(rf, X_test, y_test, str(config.figure_dir / 'confusion_matrix.png'))

if hasattr(rf, 'feature_importances_'):
    plot_feature_importance_15(rf.feature_importances_, X.columns.tolist(),
                                str(config.figure_dir / 'feature_importance.png'))

best_name = comparison['roc_auc'].idxmax()
print(f'最佳模型: {best_name} (ROC-AUC: {comparison.loc[best_name, \"roc_auc\"]:.4f})')
print('建模阶段完成')
"
    echo ""
    echo -e "${GREEN}✓ 建模阶段完成${NC}"
}

clean_outputs() {
    print_header "清理输出目录"
    if [ -d "$PROJECT_DIR/outputs/figures" ] && [ "$(ls -A "$PROJECT_DIR/outputs/figures" 2>/dev/null)" ]; then
        echo "清理 figures/ ..."
        rm -rf "$PROJECT_DIR/outputs/figures"/*
    fi
    if [ -d "$PROJECT_DIR/outputs/models" ] && [ "$(ls -A "$PROJECT_DIR/outputs/models" 2>/dev/null)" ]; then
        echo "清理 models/ ..."
        rm -rf "$PROJECT_DIR/outputs/models"/*
    fi
    echo -e "${GREEN}✓ 输出目录已清理${NC}"
}

show_help() {
    echo "用法: ./scripts/run.sh [命令]"
    echo ""
    echo "命令:"
    echo "  full     完整流水线 (sample_size=500,000)"
    echo "  quick    快速运行 (sample_size=50,000)"
    echo "  eda      仅 EDA 阶段 (数据加载 → 清洗 → 可视化)"
    echo "  ml       仅建模阶段 (特征工程 → 训练 → 评估)"
    echo "  clean    清理 outputs/ 目录"
    echo "  help     显示此帮助信息"
    echo ""
    echo "无参数运行时进入交互菜单。"
}

show_menu() {
    check_venv

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║         ${CYAN}游戏推荐数据分析 — 运行管理${NC}         ${BOLD}║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}1${NC}) 完整流水线                              ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}2${NC}) 快速运行 (小样本 ~30s)                  ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}3${NC}) 仅 EDA 阶段 (数据→清洗→可视化)          ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}4${NC}) 仅建模阶段 (特征→训练→评估)             ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}5${NC}) 清理输出目录                            ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}0${NC}) 退出                                    ${BOLD}║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    read -r -p "请选择 [0-5]: " choice
    echo ""

    case "$choice" in
        1) run_full ;;
        2) run_quick ;;
        3) run_eda ;;
        4) run_ml ;;
        5) clean_outputs ;;
        0) echo "已取消"; exit 0 ;;
        *) echo -e "${RED}无效选项${NC}"; exit 1 ;;
    esac
}

# 入口
if [ $# -eq 0 ]; then
    show_menu
else
    check_venv
    case "${1:-}" in
        full)   run_full ;;
        quick)  run_quick ;;
        eda)    run_eda ;;
        ml)     run_ml ;;
        clean)  clean_outputs ;;
        help|--help|-h) show_help ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            show_help
            exit 1
            ;;
    esac
fi
