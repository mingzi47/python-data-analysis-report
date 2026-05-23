#!/usr/bin/env bash
# ============================================================
# 项目测试脚本 — 交互式测试管理
#
# 用法:
#   ./scripts/test.sh             交互模式（显示菜单）
#   ./scripts/test.sh all         运行所有测试
#   ./scripts/test.sh module      交互式选择测试模块
#   ./scripts/test.sh failed      仅运行上次失败的测试 (--lf)
#   ./scripts/test.sh quick       快速测试（不生成图表输出）
#   ./scripts/test.sh single      运行单个测试函数（交互式）
#   ./scripts/test.sh help        显示帮助
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

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

# 获取所有测试模块列表
get_test_modules() {
    find "$PROJECT_DIR/tests" -maxdepth 1 -name "test_*.py" -exec basename {} .py \; | sort
}

# 获取指定模块中的所有测试函数
get_test_functions() {
    local module="$1"
    uv run pytest "tests/${module}.py" --collect-only -q 2>/dev/null \
        | grep -oP 'tests/\S+::\K\S+' \
        | head -20
}

run_all() {
    print_header "运行所有测试"
    uv run pytest tests/ -v "$@"
    echo ""
    echo -e "${GREEN}✓ 测试完成${NC}"
}

run_module() {
    local module="$1"
    shift 2>/dev/null || true
    print_header "运行测试模块: ${module}"
    uv run pytest "tests/${module}.py" -v "$@"
    echo ""
    echo -e "${GREEN}✓ ${module} 测试完成${NC}"
}

run_failed() {
    print_header "运行上次失败的测试 (--lf)"
    uv run pytest tests/ -v --lf "$@"
    echo ""
    echo -e "${GREEN}✓ 测试完成${NC}"
}

run_quick() {
    print_header "快速测试（跳过图表输出验证）"
    uv run pytest tests/ -v -k "not figure and not plot" "$@"
    echo ""
    echo -e "${GREEN}✓ 快速测试完成${NC}"
}

run_single() {
    local module="$1"
    local func="$2"
    shift 2 2>/dev/null || true
    print_header "运行单个测试: ${module}::${func}"
    uv run pytest "tests/${module}.py::${func}" -v "$@"
    echo ""
    echo -e "${GREEN}✓ 测试完成${NC}"
}

show_help() {
    echo "用法: ./scripts/test.sh [命令] [参数...]"
    echo ""
    echo "命令:"
    echo "  all                     运行所有测试"
    echo "  module [name]           运行指定测试模块（不指定则交互选择）"
    echo "  failed                  仅运行上次失败的测试 (--lf)"
    echo "  quick                   快速测试（跳过图表输出验证）"
    echo "  single [module] [func]  运行单个测试函数（不指定则交互选择）"
    echo "  help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./scripts/test.sh all"
    echo "  ./scripts/test.sh module test_loader"
    echo "  ./scripts/test.sh single test_loader test_load_games_csv"
    echo "  ./scripts/test.sh failed"
    echo ""
    echo "额外 pytest 参数会透传:"
    echo "  ./scripts/test.sh all -x         # 遇错即停"
    echo "  ./scripts/test.sh all --tb=short # 简短回溯"
}

# 交互式选择测试模块
interactive_module() {
    local modules
    mapfile -t modules < <(get_test_modules)

    echo ""
    echo -e "${BOLD}可用测试模块:${NC}"
    echo ""
    for i in "${!modules[@]}"; do
        printf "  ${GREEN}%2d${NC}) %s\n" "$((i + 1))" "${modules[$i]}"
    done
    echo ""
    read -r -p "请选择模块 [1-${#modules[@]}]: " mod_choice

    if [[ "$mod_choice" =~ ^[0-9]+$ ]] && [ "$mod_choice" -ge 1 ] && [ "$mod_choice" -le "${#modules[@]}" ]; then
        local selected="${modules[$((mod_choice - 1))]}"
        run_module "$selected"
    else
        echo -e "${RED}无效选择${NC}"
        exit 1
    fi
}

# 交互式选择单个测试
interactive_single() {
    local modules
    mapfile -t modules < <(get_test_modules)

    echo ""
    echo -e "${BOLD}选择测试模块:${NC}"
    for i in "${!modules[@]}"; do
        printf "  ${GREEN}%2d${NC}) %s\n" "$((i + 1))" "${modules[$i]}"
    done
    echo ""
    read -r -p "请选择模块 [1-${#modules[@]}]: " mod_choice

    if ! [[ "$mod_choice" =~ ^[0-9]+$ ]] || [ "$mod_choice" -lt 1 ] || [ "$mod_choice" -gt "${#modules[@]}" ]; then
        echo -e "${RED}无效选择${NC}"
        exit 1
    fi

    local module="${modules[$((mod_choice - 1))]}"

    # 列出该模块的测试函数
    echo ""
    echo -e "${BOLD}${module} 中的测试函数:${NC}"
    echo ""
    local funcs
    mapfile -t funcs < <(get_test_functions "$module")

    if [ "${#funcs[@]}" -eq 0 ]; then
        echo -e "${YELLOW}无法解析测试函数，直接运行整个模块${NC}"
        run_module "$module"
        return
    fi

    for i in "${!funcs[@]}"; do
        printf "  ${GREEN}%2d${NC}) %s\n" "$((i + 1))" "${funcs[$i]}"
    done
    printf "  ${GREEN}%2d${NC}) %s\n" "0" "全部（运行整个模块）"
    echo ""
    read -r -p "请选择测试 [0-${#funcs[@]}]: " func_choice

    if [ "$func_choice" = "0" ] || [ "$func_choice" = "" ]; then
        run_module "$module"
    elif [[ "$func_choice" =~ ^[0-9]+$ ]] && [ "$func_choice" -ge 1 ] && [ "$func_choice" -le "${#funcs[@]}" ]; then
        local func="${funcs[$((func_choice - 1))]}"
        run_single "$module" "$func"
    else
        echo -e "${RED}无效选择${NC}"
        exit 1
    fi
}

show_menu() {
    check_venv

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║          ${CYAN}游戏推荐数据分析 — 测试管理${NC}         ${BOLD}║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}1${NC}) 运行所有测试                            ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}2${NC}) 选择测试模块运行                        ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}3${NC}) 运行上次失败的测试 (--lf)               ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}4${NC}) 快速测试（跳过图表验证）                ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}5${NC}) 运行单个测试函数                        ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}0${NC}) 退出                                    ${BOLD}║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    read -r -p "请选择 [0-5]: " choice
    echo ""

    case "$choice" in
        1) run_all ;;
        2) interactive_module ;;
        3) run_failed ;;
        4) run_quick ;;
        5) interactive_single ;;
        0) echo "已取消"; exit 0 ;;
        *) echo -e "${RED}无效选项${NC}"; exit 1 ;;
    esac
}

# 入口
if [ $# -eq 0 ]; then
    show_menu
else
    check_venv
    cmd="${1:-}"
    shift 2>/dev/null || true

    case "$cmd" in
        all)
            run_all "$@"
            ;;
        module)
            if [ $# -gt 0 ]; then
                run_module "$@"
            else
                interactive_module
            fi
            ;;
        failed)
            run_failed "$@"
            ;;
        quick)
            run_quick "$@"
            ;;
        single)
            if [ $# -ge 2 ]; then
                run_single "$1" "$2"
            else
                interactive_single
            fi
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $cmd${NC}"
            show_help
            exit 1
            ;;
    esac
fi
