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
    $PYTHON main.py --mode full
    echo ""
    echo -e "${GREEN}✓ 流水线执行完成${NC}"
}

run_quick() {
    print_header "快速运行（小样本）"
    echo -e "样本量: ${YELLOW}50,000${NC}"
    echo -e "预计耗时: ${YELLOW}30-60 秒${NC}"
    echo ""
    $PYTHON main.py --quick
    echo ""
    echo -e "${GREEN}✓ 快速运行完成${NC}"
}

run_eda() {
    print_header "EDA 阶段运行（阶段 1-4）"
    echo -e "包含: 数据加载 → 清洗 → EDA 可视化 → 分析指标"
    echo -e "预计耗时: ${YELLOW}1-2 分钟${NC}"
    echo ""
    $PYTHON main.py --mode eda
    echo ""
    echo -e "${GREEN}✓ EDA 阶段完成${NC}"
}

run_ml() {
    print_header "建模阶段运行（阶段 5-8）"
    echo -e "前置条件: 已完成数据加载和 EDA"
    echo -e "包含: 特征工程 → 建模 → 评估 → 结论"
    echo -e "预计耗时: ${YELLOW}2-4 分钟${NC}"
    echo ""
    $PYTHON main.py --mode ml
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
