# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Data analysis project using the `antonkozyriev/game-recommendations-on-steam` dataset. The full pipeline, per README: background analysis → data reading → cleaning/preprocessing → EDA → visualization → feature engineering → sklearn modeling/optimization → model evaluation/analysis → visualization → results/application/conclusions.

## Environment

- Python 3.13 (see `.python-version`)
- Dependencies managed via `pyproject.toml` (currently no dependencies declared)
- Install: `pip install -e .` (once dependencies are added)

## Running

```bash
python main.py
```

## Architecture

`main.py` is the single entry point. As the analysis grows, break out separate modules for each pipeline phase (e.g., `data_loading.py`, `preprocessing.py`, `modeling.py`, `visualization.py`).

## Key Libraries (planned per README)

- **sklearn** — modeling, optimization, evaluation
- **Visualization** — matplotlib and/or seaborn (the README calls for visualization at two stages: EDA and final results)
