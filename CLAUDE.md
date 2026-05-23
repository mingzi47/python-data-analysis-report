# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Data analysis project using the `antonkozyriev/game-recommendations-on-steam` dataset (41M+ Steam user recommendations). Full pipeline: background analysis → data reading → cleaning/preprocessing → EDA → visualization → feature engineering → sklearn modeling/optimization → model evaluation/analysis → visualization → results/application/conclusions.

## Environment

- Python 3.13 (see `.python-version`)
- Dependencies: `kagglehub`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- Install: `pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost`

## Running

```bash
python main.py
```

## Architecture

```
src/
├── data/
│   ├── loader.py       # kagglehub download + pd.read_csv / json parsing
│   └── cleaner.py      # missing values, type conversion, outliers, dedup, metadata merge
├── features/
│   └── builder.py      # game/user/interaction feature construction + ColumnTransformer pipeline
├── models/
│   ├── baseline.py     # DummyClassifier + simple rule baselines
│   ├── trainer.py      # LogisticRegression, RandomForest, XGBoost + RandomizedSearchCV
│   └── evaluator.py    # metrics calculation, model comparison table
├── visualization/
│   ├── eda_plots.py    # EDA charts (histograms, bar charts, scatter, box plots, heatmaps)
│   └── model_plots.py  # ROC curves, confusion matrix, feature importance, PDP, learning curves
└── utils/
    └── config.py       # Config dataclass: paths, random_seed, sample_size, cv settings
```

`main.py` orchestrates the pipeline by calling modules in order. Each module exposes a small set of functions (see `docs/03-architecture.md` for full signatures).

## Task-Based Workflow

Implementation is organized as 44 granular tasks across 8 phases. Each task file in `tasks/` follows a consistent template: description → dependencies → inputs → outputs → steps → acceptance criteria.

Work through tasks in numeric order within each phase. Phases are sequential (01 → 02 → ... → 08), but some tasks within a phase can run in parallel (e.g., models 03/04/05 in phase 06).

```
tasks/
├── 01-data-loading/          (6 tasks)
├── 02-data-cleaning/         (5 tasks)
├── 03-eda-game-profile/      (6 tasks)
├── 04-eda-user-profile/      (5 tasks)
├── 05-feature-engineering/   (5 tasks)
├── 06-modeling/              (6 tasks)
├── 07-evaluation/            (7 tasks)
└── 08-conclusions/           (3 tasks)
```

## Documentation

- `README.md` — project overview + quick start + doc navigation
- `docs/01-background.md` — dataset description, 4 research questions, technical challenges
- `docs/02-methodology.md` — detailed methods for all 8 pipeline stages
- `docs/03-architecture.md` — code architecture, module responsibilities, data flow
- `docs/04-data-dictionary.md` — field definitions, types, usage priority for all 4 data files

## Key Conventions

- **Data leakage prevention:** Split by `GroupShuffleSplit(groups=user_id)`, never `train_test_split`. Interaction features (aggregates from `recommendations.csv`) must be computed on training set only, then mapped to test set.
- **Memory management:** `recommendations.csv` is 41M+ rows. Use `nrows=500_000` sampling during development. Full-data run is optional at the end.
- **Reproducibility:** All random seeds come from `Config.random_seed` (default 42). No hardcoded seeds in individual modules.
- **Output paths:** Charts go to `outputs/figures/`, serialized models to `outputs/models/`. Both directories are gitignored.
- **Chinese comments/UI:** Project documentation and printed output are in Chinese. Code identifiers (variable names, function names) are in English.
- **No notebooks for pipeline code:** Jupyter notebooks (`notebooks/`) are for interactive EDA exploration only. All pipeline logic lives in `src/` Python modules.
