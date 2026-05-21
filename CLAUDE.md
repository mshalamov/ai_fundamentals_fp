# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

A course final-project deliverable: a simple linear regression model trained on synthetic data and served through a Streamlit web app. Spec lives at `docs/ai_final_project.md`; implementation plan at `docs/superpowers/plans/2026-05-21-ml-streamlit-app.md`. Dependencies are managed by **uv** (single source of truth: `pyproject.toml`, with a committed `uv.lock`). The 12-test pytest suite and ruff lint are both expected to pass clean.

## Commands

```bash
# Install/sync deps (creates .venv if missing). Installs main + test + linter groups by default.
uv sync

# Step 1: train + serialize the model and datasets.
# Writes linear_regression_model.joblib, X.joblib, y.joblib into the CWD (i.e. src/).
cd src && uv run python model.py

# Step 2: serve the Streamlit UI (must be run from src/, since app.py
# loads the .joblib artifacts via bare filenames from the CWD).
cd src && uv run streamlit run app.py

# Tests + lint
uv run pytest -v
uv run ruff check .
```

## Architecture

Two-stage pipeline with on-disk handoff via joblib — there is no shared module state between training and serving:

1. **`src/model.py`** — generates a synthetic 1-feature regression dataset (`sklearn.datasets.make_regression`, n=100, noise=20, seed=42), rescales X to `[-3, 3]`, train/test-splits, fits `LinearRegression`, prints MSE, then dumps three artifacts: `linear_regression_model.joblib`, `X.joblib`, `y.joblib`.
2. **`src/app.py`** — Streamlit UI loads those three joblib files at predict time. The feature slider is hard-coded to `[-3.0, 3.0]` to match model.py's rescaling — keep these in sync if you change either. `visualize_difference` finds the "actual" target by nearest-X lookup (`_index_of_closest`) rather than re-running the data generator, so the X/y joblibs are load-bearing for the plot.

The default model filename `"linear_regression_model.joblib"` is duplicated as a default arg in both `save_regression_model` (model.py) and `load_and_predict` (app.py) — change both if renaming.

## UI

The app renders: title "Simple Regression model prediction", a slider "Input Feature for Prediction" (-3.00 to 3.00), a "Predict value" button, the numeric prediction (e.g. `Prediction: 26.51…`), and a "Prediction vs Actual Target" scatter (grey dataset dots, blue actual, red predicted, dashed vertical line annotated with `Difference = …`).

## Tests

12 tests live under `tests/`:
- `tests/conftest.py` — shared fixtures: `tiny_dataset` (deterministic `y = 2x + 1`), `tiny_model`, and `artifacts_cwd` (chdirs into a tmp dir pre-populated with the three joblib artifacts so `src/app.py`'s CWD-relative loads work).
- `tests/test_model.py` — one direct test per public function in `src/model.py`.
- `tests/test_app.py` — direct tests for `load_and_predict` and `visualize_difference`, plus two `streamlit.testing.v1.AppTest` cases covering the UI rendering and click flow. Note: `st.write` output is asserted via `at.markdown` (current Streamlit versions consolidate writes into the markdown stream); `getattr(at, "write", [])` is used defensively for forward-compat.

When adding tests, prefer the `artifacts_cwd` fixture over manual joblib setup.
