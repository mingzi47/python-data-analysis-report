# Integration Tests Design — Steam Game Recommendation Analysis

**Date:** 2026-06-02
**Status:** Approved

## Context

107 unit tests cover individual modules (loader, cleaner, builder, models, plots, helpers). No tests verify that the full pipeline in `main.py` works end-to-end. A refactoring in one module could break the orchestration logic without any test catching it.

## Approach: Full-Pipeline End-to-End

One test class `TestFullPipeline` with 3 tests, all running the complete `load_and_clean → run_eda → run_modeling` flow using sample test fixtures.

### Test File

`tests/test_integration.py`

### Fixtures

- Reuse existing `conftest.py` fixtures: `sample_games_csv`, `sample_users_csv`, `sample_recommendations_csv`, `sample_metadata_json`
- New `integration_config(tmp_path)` fixture: creates `Config` with `data_dir` and `output_dir` pointing to `tmp_path`
- `monkeypatch` to stub `download_dataset` (return `tmp_path` directly, no kagglehub)

### Test Cases

| Test | What It Verifies |
|------|-----------------|
| `test_full_pipeline_runs_without_error` | `load_and_clean` → `run_eda` → `run_modeling` completes without exception |
| `test_data_flow_shapes_consistent` | Train/test user disjoint, feature matrices have matching dimensions, y in [0,1], predictions produce valid output |
| `test_output_files_generated` | `figures/` contains PNG files, `models/` contains `comparison.csv` |

### Key Decisions

- **No kagglehub download**: `monkeypatch` replaces `download_dataset` with a function returning `tmp_path`
- **No numeric assertions**: Integration tests verify structural integrity (no crash, shapes match, files exist), not model performance
- **Tiny data**: test fixtures have 3 games, 3 users, 5 recommendations — tests complete in < 5 seconds
- **All in one module**: single `test_integration.py` file, following the existing test convention

### Scope Boundaries

- **In scope**: pipeline orchestration, data flow consistency, output artifact generation
- **Out of scope**: chart visual correctness, model accuracy thresholds, performance benchmarks
