# 03 — 代码架构

## 设计原则

- **模块化：** 每个 Pipeline 阶段对应独立的 Python 模块，职责单一
- **可复现：** 所有随机种子统一配置，数据处理流水线可复用
- **渐进式：** 先用采样数据快速验证每个模块，再扩展到全量数据

## 目录结构

```
report/
├── main.py                         # 入口：编排完整流水线
├── pyproject.toml                  # 项目元数据和依赖声明
├── README.md                       # 项目概览 + 快速开始
│
├── scripts/                        # 交互式管理脚本
│   ├── run.sh                      # 运行管理（完整/快速/EDA/建模/清理）
│   └── test.sh                     # 测试管理（全部/模块/失败重跑/快速/单函数）
│
├── docs/                           # 文档
│   ├── 01-background.md            # 项目背景与研究问题
│   ├── 02-methodology.md           # 各阶段分析方法
│   ├── 03-architecture.md          # 本文档：代码架构设计
│   └── 04-data-dictionary.md       # 数据集字段说明
│
├── src/                            # 源代码
│   ├── __init__.py
│   │
│   ├── data/                       # 数据层：加载与清洗
│   │   ├── __init__.py
│   │   ├── loader.py               # 数据下载与读取
│   │   └── cleaner.py              # 缺失值、异常值、类型转换、去重
│   │
│   ├── features/                   # 特征层：特征工程
│   │   ├── __init__.py
│   │   └── builder.py              # 特征构造 + ColumnTransformer 流水线
│   │
│   ├── analysis/                    # 分析辅助：集中度、极端用户、活跃度分层
│   │   ├── __init__.py
│   │   └── helpers.py               # 计算 Gini 系数、识别极端用户、活跃度分层
│   │
│   ├── models/                     # 模型层：训练、调参、评估
│   │   ├── __init__.py
│   │   ├── baseline.py             # DummyClassifier + 简单规则基线
│   │   ├── trainer.py              # 模型训练 + RandomizedSearchCV
│   │   └── evaluator.py            # 性能指标计算 + 模型对比
│   │
│   ├── visualization/              # 可视化层：图表生成
│   │   ├── __init__.py
│   │   ├── eda_plots.py            # EDA 阶段图表（问题 1 & 2）
│   │   └── model_plots.py          # 模型评估图表（ROC、混淆矩阵、PDP、学习曲线）
│   │
│   └── utils/                      # 工具层
│       ├── __init__.py
│       └── config.py               # 路径常量、随机种子、全局配置
│
├── notebooks/                      # Jupyter 交互式探索
│   └── eda.ipynb                   # EDA 交互笔记本
│
├── outputs/                        # 输出（gitignored）
│   ├── figures/                    # 图表 PNG/SVG
│   └── models/                     # 序列化模型 .pkl
│
└── data/                           # 本地数据缓存（gitignored）
    └── .gitkeep
```

## 模块职责

### `main.py` — 入口与编排

对外暴露唯一入口。按顺序调用各模块，组合成完整流水线：

```python
def main():
    config = Config()

    # 阶段 1-2: 数据
    datasets = load_all_data(config)
    datasets = clean_all_data(datasets)

    # 阶段 3: 分析辅助
    concentration = compute_concentration_metrics(datasets["recommendations"])
    extreme = analyze_extreme_users(datasets["recommendations"])
    datasets["users"] = compute_user_activity_tiers(datasets["users"])

    # 阶段 4: EDA + 可视化
    run_eda(datasets, config)
    generate_eda_figures(datasets, config)

    # 阶段 5: 特征工程
    X, y, groups = build_features(datasets)

    # 阶段 6: 建模
    X_train, X_test, y_train, y_test = split(X, y, groups)
    models = train_all_models(X_train, y_train)

    # 阶段 7: 评估
    results = evaluate_all(models, X_test, y_test)
    generate_model_figures(models, results, config)

    # 阶段 8: 结论
    print_conclusions(results)
```

### `src/data/loader.py` — 数据加载

```python
def download_dataset() -> Path:
    """使用 kagglehub 下载数据集，返回缓存路径"""

def load_games(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """加载 games.csv"""

def load_users(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """加载 users.csv"""

def load_recommendations(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """加载 recommendations.csv（默认采样 50 万条）"""

def load_metadata(path: Path) -> pd.DataFrame:
    """加载 games_metadata.json，提取 app_id、tags（实际数据无 genres/type/early_access）"""
```

### `src/data/cleaner.py` — 数据清洗

```python
def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值、转换日期类型；rating 数值型/文本型自适应处理"""

def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值、标记逻辑异常用户"""

def clean_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """去除目标变量缺失、去重、类型转换"""

def merge_metadata(games: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """将展平后的 metadata 合并到 games 表"""
```

### `src/features/builder.py` — 特征工程

```python
def build_features(
    recommendations: pd.DataFrame,
    games: pd.DataFrame,
    users: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    构造特征矩阵、目标向量和分组标识。

    Returns:
        X: 特征矩阵
        y: is_recommended（0/1）
        groups: user_id（用于分组切分）
    """

def build_preprocessor() -> ColumnTransformer:
    """构建 sklearn 特征处理流水线（StandardScaler + OneHotEncoder）"""
```

### `src/analysis/helpers.py` — 分析辅助

```python
def compute_concentration_metrics(recommendations: pd.DataFrame) -> dict:
    """计算推荐量集中度：Gini系数、Top1%/5%/10%/20%占比"""

def analyze_extreme_users(recommendations: pd.DataFrame) -> dict:
    """识别极端用户：纯好评/纯差评用户占比、平均评价数、按评价数分桶"""

def compute_user_activity_tiers(users: pd.DataFrame) -> pd.DataFrame:
    """按产品数分位数划分活跃度层级：low/medium/high/extreme"""
```

### `src/models/baseline.py` — 基线模型

```python
def evaluate_baselines(X_train, y_train, X_test, y_test) -> dict[str, dict]:
    """计算三个基线的性能指标并返回"""
```

### `src/models/trainer.py` — 模型训练

```python
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """训练逻辑回归（含 RandomizedSearchCV 调参）"""

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """训练随机森林（含 RandomizedSearchCV 调参）"""

def train_xgboost(X_train, y_train) -> XGBClassifier:
    """训练 XGBoost（含早停）"""

def split_data(X, y, groups) -> tuple:
    """按用户分组切分训练/测试集"""
```

### `src/models/evaluator.py` — 模型评估

```python
def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """计算 Accuracy, Precision, Recall, F1, ROC-AUC"""

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """生成模型对比表格"""

def plot_feature_importance(model, feature_names, top_n=15):
    """特征重要性（仅对树模型有效）"""
```

### `src/visualization/eda_plots.py` — EDA 图表

```python
def plot_price_distribution(games: pd.DataFrame, save_path: str):
    """价格分布直方图（log 轴）"""

def plot_rating_distribution(games: pd.DataFrame, save_path: str):
    """评分分布：数值型用直方图+KDE，文本型用柱状图"""

def plot_genre_bar(games: pd.DataFrame, save_path: str):
    """Top 20 游戏类型/标签条形图（genres 为空时回退到 tags）"""

def plot_release_timeline(games: pd.DataFrame, save_path: str):
    """年度游戏发布量折线图"""

def plot_user_activity(users: pd.DataFrame, save_path: str):
    """用户评论数/购买数分布（log-log）"""

def plot_user_activity_distribution(users: pd.DataFrame, save_path: str):
    """用户产品数/评论数双直方图（log轴）"""

def plot_user_recommend_rate_distribution(recommendations: pd.DataFrame, save_path: str):
    """用户推荐率分布直方图 + 0.5/均值标注线"""

def plot_purchase_vs_reviews(users: pd.DataFrame, save_path: str):
    """购买数 vs 评论数散点图（log-log）+ Spearman相关 + y=x线"""

def plot_long_tail(recommendations: pd.DataFrame, save_path: str):
    """游戏推荐量长尾分布"""

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """数值变量相关性热力图"""
```

### `src/visualization/model_plots.py` — 模型评估图表

```python
def plot_roc_curves(models: dict, X_test, y_test, save_path: str):
    """多模型 ROC 曲线叠加图"""

def plot_confusion_matrix(model, X_test, y_test, save_path: str):
    """混淆矩阵热力图"""

def plot_feature_importance_15(importances, names, save_path: str):
    """特征重要性 Top 15 条形图"""

def plot_partial_dependence(model, X, features, save_path: str):
    """部分依赖图（3-5 个子图）"""

def plot_learning_curve(model, X, y, save_path: str):
    """学习曲线"""
```

### `src/utils/config.py` — 全局配置

```python
@dataclass
class Config:
    random_seed: int = 42
    sample_size: int | None = 500_000    # 采样量，None 为全量
    test_size: float = 0.2               # 测试集比例
    cv_folds: int = 3                    # 交叉验证折数
    cv_iter: int = 20                    # RandomSearch 迭代次数
    data_dir: Path = Path("data")        # 数据集下载目录（gitignore）
    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("outputs/figures")
    model_dir: Path = Path("outputs/models")
```

### `scripts/run.sh` — 交互式运行管理

支持两种调用模式：
- **命令行模式：** `./scripts/run.sh full|quick|eda|ml|all|clean` 直接执行
- **交互模式：** 无参数运行显示选项菜单

| 命令 | 功能 | 样本量 | 预计耗时 |
|------|------|--------|----------|
| `full` | 完整 8 阶段流水线 | 500K | 3-5 分钟 |
| `quick` | 快速验证（`--quick` 参数覆写 `Config.sample_size=50_000`） | 50K | ~30 秒 |
| `eda` | 阶段 1-4：数据加载 → 清洗 → EDA 图表 → 分析指标 | 500K | 1-2 分钟 |
| `ml` | 阶段 5-8：特征工程 → 建模 → 评估 → 结论 | 500K | 2-4 分钟 |
| `all` | 全量数据完整流水线（`--all`，41M+ 行，需 8GB+ 内存） | 全量 | 30 分钟+ |
| `clean` | 清空 `outputs/figures/` 和 `outputs/models/` | — | <1 秒 |

关键设计：
- Python 逻辑全部在 `main.py` 中，通过 `--mode full|eda|ml`、`--quick`、`--all` CLI 参数控制
- `--all` 将 `Config.sample_size` 设为 `None`，透传给 `pd.read_csv(nrows=None)` 读取全量数据
- `run.sh` 仅做薄包装层（颜色输出、耗时提示、菜单交互），直接调用 `uv run python main.py <args>`
- `set -euo pipefail` 确保任何命令失败立即终止

### `scripts/test.sh` — 交互式测试管理

支持命令行模式和交互模式，pytest 额外参数透传。

| 命令 | 功能 |
|------|------|
| `all` | 运行所有 88 个测试 |
| `module [name]` | 运行指定模块（不指定则交互选择） |
| `failed` | `--lf` 仅重跑上次失败的测试 |
| `quick` | `-k "not figure and not plot"` 跳过图表验证 |
| `single [module] [func]` | 运行单个测试函数（不指定则交互选择） |

关键设计：
- 自动扫描 `tests/test_*.py` 并以序号菜单呈现
- 通过 `pytest --collect-only -q` 解析测试函数列表
- 额外参数直接透传给 pytest：`./scripts/test.sh all -x --tb=short`

## 数据流

```
games.csv ────────────┐
users.csv ────────────┤
                       ├─→ loader.py ─→ cleaner.py ─→ merge ─→ builder.py
recommendations.csv ──┤                                        │
games_metadata.json ──┘                                        │
                                                               ▼
                                                        X, y, groups
                                                               │
                                                               ▼
                                                split_data() ──→ trainer.py
                                                               │
                                                               ▼
                                                         evaluator.py
                                                               │
                                                               ▼
                                                       model_plots.py
                                                               │
                                                               ▼
                                                       outputs/figures/
```

## 依赖声明

```toml
# pyproject.toml
[project]
dependencies = [
    "kagglehub",        # 数据集下载
    "pandas",           # 数据处理
    "numpy",            # 数值计算
    "matplotlib",       # 可视化底层
    "seaborn",          # 统计图高层 API
    "scikit-learn",     # 建模、预处理、评估
    "xgboost",          # 梯度提升分类器
]
```
