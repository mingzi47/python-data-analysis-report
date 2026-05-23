# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Data analysis project using the `antonkozyriev/game-recommendations-on-steam` dataset (41M+ Steam user recommendations). Full pipeline: background analysis → data reading → cleaning/preprocessing → EDA → visualization → feature engineering → sklearn modeling/optimization → model evaluation/analysis → visualization → results/application/conclusions.

## Environment

- Python 3.13+ (tested on 3.14, see `.python-version`)
- Dependencies: `kagglehub`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `scipy`
- Install: `pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost scipy`
- Virtual environment: `.venv/`
- Run tests: `.venv/bin/python -m pytest tests/ -v`

## Running

```bash
# 完整流水线
python main.py

# 交互式脚本
./scripts/run.sh          # 运行管理：完整/快速/EDA/建模/清理
./scripts/test.sh         # 测试管理：全部/模块/失败重跑/快速/单函数

# 常用快捷命令
./scripts/run.sh quick    # 快速运行 (50K 样本，~30s)
./scripts/run.sh eda      # 仅 EDA 阶段
./scripts/test.sh all     # 全部 88 个测试
./scripts/test.sh failed  # 仅重跑上次失败的
```

## Architecture

```
scripts/
├── run.sh              # 交互式运行管理（bash）
└── test.sh             # 交互式测试管理（bash）

src/
├── analysis/
│   └── helpers.py       # Gini coefficient, extreme user detection, activity tiers
├── data/
│   ├── loader.py       # kagglehub download + pd.read_csv / json parsing
│   └── cleaner.py      # missing values, type conversion, outliers, dedup, metadata merge
├── features/
│   └── builder.py      # game/user/interaction feature construction + ColumnTransformer pipeline
├── models/
│   ├── baseline.py     # DummyClassifier + simple rule baselines
│   ├── trainer.py      # LogisticRegression, RandomForest, XGBoost + GroupShuffleSplit
│   └── evaluator.py    # metrics calculation, model comparison table
├── visualization/
│   ├── eda_plots.py    # 10 EDA charts (histograms, bar, scatter, heatmaps)
│   └── model_plots.py  # ROC curves, confusion matrix, feature importance, PDP, learning curves
└── utils/
    └── config.py       # Config dataclass: paths, random_seed, sample_size, cv settings

tests/
├── conftest.py          # shared fixtures (sample DataFrames)
├── test_config.py       # 2 tests
├── test_loader.py       # 11 tests
├── test_cleaner.py      # 12 tests
├── test_builder.py      # 11 tests
├── test_models.py       # 10 tests
├── test_eda_plots.py    # 19 tests
├── test_model_plots.py  # 7 tests
└── test_helpers.py      # 16 tests
```

`main.py` orchestrates the pipeline by calling modules in order. Each module exposes a small set of functions (see `docs/03-architecture.md` for full signatures).

## Task-Based Workflow

~~Implementation is organized as 44 granular tasks across 8 phases. Each task file in `tasks/` follows a consistent template: description → dependencies → inputs → outputs → steps → acceptance criteria.~~

**Status: ALL 44 TASKS COMPLETE (2026-05-23).**

All 8 phases have been implemented using TDD (test-first development). 88 tests pass across 8 test files. See commit history for detailed progression.

<details>
<summary>Task archive (click to expand)</summary>

```
tasks/
├── 01-data-loading/          (6 tasks) ✓ DONE
├── 02-data-cleaning/         (5 tasks) ✓ DONE
├── 03-eda-game-profile/      (6 tasks) ✓ DONE
├── 04-eda-user-profile/      (5 tasks) ✓ DONE
├── 05-feature-engineering/   (5 tasks) ✓ DONE
├── 06-modeling/              (6 tasks) ✓ DONE
├── 07-evaluation/            (7 tasks) ✓ DONE
└── 08-conclusions/           (3 tasks) ✓ DONE
```

</details>

## Documentation

- `README.md` — project overview + quick start + doc navigation
- `docs/01-background.md` — dataset description, 4 research questions, technical challenges
- `docs/02-methodology.md` — detailed methods for all 8 pipeline stages
- `docs/03-architecture.md` — code architecture, module responsibilities, data flow
- `docs/04-data-dictionary.md` — field definitions, types, usage priority for all 4 data files
- `docs/05-game-ecosystem-summary.md` — 游戏生态画像总结
- `docs/06-user-behavior-summary.md` — 用户行为画像总结
- `docs/07-findings-summary.md` — 研究发现汇总（4 个研究问题）
- `docs/08-business-recommendations.md` — 业务建议（开发者 + 平台运营）
- `docs/09-limitations.md` — 局限性与后续改进方向

## Key Conventions

- **Data leakage prevention:** Split by `GroupShuffleSplit(groups=user_id)`, never `train_test_split`. Interaction features (aggregates from `recommendations.csv`) must be computed on training set only, then mapped to test set.
- **Memory management:** `recommendations.csv` is 41M+ rows. Use `nrows=500_000` sampling during development. Full-data run is optional at the end.
- **Reproducibility:** All random seeds come from `Config.random_seed` (default 42). No hardcoded seeds in individual modules.
- **Output paths:** Charts go to `outputs/figures/`, serialized models to `outputs/models/`. Both directories are gitignored.
- **Chinese comments/UI:** Project documentation and printed output are in Chinese. Code identifiers (variable names, function names) are in English.
- **No notebooks for pipeline code:** Jupyter notebooks (`notebooks/`) are for interactive EDA exploration only. All pipeline logic lives in `src/` Python modules.
- **TDD required:** All new features/bugfixes must follow test-first development. Write test → watch fail → implement → watch pass → commit. 88 tests exist as of completion.
- **Commit per task:** Each logical unit of work gets its own commit with a descriptive message.
- **Data vs. dictionary discrepancies:** The actual dataset differs from `docs/04-data-dictionary.md`'s original description in several ways:
  - `games.rating` is **text** ("Very Positive", "Mixed", etc.), NOT float. Use `games.positive_ratio` (int, 0-100) for numeric rating.
  - `games.csv` has `win`, `mac`, `linux`, `user_reviews` columns but NO `user_score` or `owners` column.
  - `games_metadata.json` only has `app_id`, `description`, `tags` — NO `genres`, `type`, or `early_access` fields.
  - All code must handle both the test-fixture format (numeric rating, both tags+genres) and the real-data format (text rating, tags only). Use `pd.api.types.is_numeric_dtype()` to detect and branch.
