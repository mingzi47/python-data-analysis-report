# 基于 antonkozyriev/game-recommendations-on-steam 数据集的数据分析

## 目标

完整数据分析案例——从原始数据到可落地结论的数据科学流水线：

```
背景分析 → 数据读取 → 清洗预处理 → EDA → 可视化 → 特征工程 → sklearn 建模优化 → 模型评估 → 结果可视化 → 结论
```

---

## 快速开始

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost
python main.py
```

## 数据集

使用 `kagglehub` 自动下载 [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)，包含 4 个文件、4,100 万+ 推荐记录。详见 [docs/04-data-dictionary.md](docs/04-data-dictionary.md)。

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [docs/01-background.md](docs/01-background.md) | 项目背景、四项核心研究问题、技术挑战 |
| [docs/02-methodology.md](docs/02-methodology.md) | 8 个阶段的详细分析方法与操作步骤 |
| [docs/03-architecture.md](docs/03-architecture.md) | 代码架构、模块职责、数据流、目录结构 |
| [docs/04-data-dictionary.md](docs/04-data-dictionary.md) | 数据集各文件的字段定义、类型、使用优先级 |
| [docs/05-game-ecosystem-summary.md](docs/05-game-ecosystem-summary.md) | 游戏生态画像总结 |
| [docs/06-user-behavior-summary.md](docs/06-user-behavior-summary.md) | 用户行为画像总结 |
| [docs/07-findings-summary.md](docs/07-findings-summary.md) | 研究发现汇总 |
| [docs/08-business-recommendations.md](docs/08-business-recommendations.md) | 业务建议 |
| [docs/09-limitations.md](docs/09-limitations.md) | 局限性与后续方向 |

---

## 核心研究问题

1. **游戏生态画像** — Steam 市场的价格、评分、类型、时间趋势和头部集中度特征
2. **用户行为画像** — 用户活跃度分布、推荐偏好、极端行为识别
3. **推荐预测建模** — 基于用户+游戏特征预测推荐行为，对比逻辑回归/随机森林/XGBoost
4. **驱动因素分析** — 特征重要性、部分依赖图、冷启动问题的业务建议

---

## 项目结构

```
report/
├── main.py                     # 入口：编排完整流水线
├── docs/                       # 项目文档（9 份）
├── src/                        # 源代码
│   ├── analysis/helpers.py     #   分析辅助函数
│   ├── data/loader.py          #   数据下载与读取
│   ├── data/cleaner.py         #   清洗与预处理
│   ├── features/builder.py     #   特征工程流水线
│   ├── models/baseline.py      #   基线模型
│   ├── models/trainer.py       #   模型训练与调参
│   ├── models/evaluator.py     #   模型评估与对比
│   ├── visualization/          #   图表生成（EDA + 模型评估）
│   └── utils/config.py         #   全局配置
├── notebooks/eda.ipynb         # EDA 交互笔记本
└── outputs/                    # 输出（图表 + 模型）
```
