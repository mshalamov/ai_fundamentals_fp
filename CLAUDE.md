# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a course final-project skeleton: a simple linear regression model trained on synthetic data and served through a Streamlit web app. `src/model.py` and `src/app.py` ship with `# TODO: your code here` placeholders that the student fills in. Spec lives at `docs/ai_final_project.md`. There is no test suite, lint config, or CI.

## Commands

```bash
# Install deps
pip install -r requirements.txt

# Step 1: train + serialize the model and datasets.
# Writes linear_regression_model.joblib, X.joblib, y.joblib into the CWD.
cd src && python model.py

# Step 2: serve the Streamlit UI (must be run from src/, since app.py
# loads the .joblib artifacts via bare filenames from the CWD).
cd src && streamlit run app.py
```

## Architecture

Two-stage pipeline with on-disk handoff via joblib — there is no shared module state between training and serving:

1. **`src/model.py`** — generates a synthetic 1-feature regression dataset (`sklearn.datasets.make_regression`, n=100, noise=20, seed=42), rescales X to `[-3, 3]`, train/test-splits, fits `LinearRegression`, prints MSE, then dumps three artifacts: `linear_regression_model.joblib`, `X.joblib`, `y.joblib`.
2. **`src/app.py`** — Streamlit UI loads those three joblib files at predict time. The feature slider is hard-coded to `[-3.0, 3.0]` to match model.py's rescaling — keep these in sync if you change either. `visualize_difference` finds the "actual" target by nearest-X lookup (`_index_of_closest`) rather than re-running the data generator, so the X/y joblibs are load-bearing for the plot.

The default model filename `"linear_regression_model.joblib"` is duplicated as a default arg in both `save_regression_model` (model.py) and `load_and_predict` (app.py) — change both if renaming.

## Expected Completed UI

Per `docs/ai_final_project.md`, the finished app must show: title "Simple Regression model prediction", a slider "Input Feature for Prediction" (-3.00 to 3.00), a "Predict value" button, the numeric prediction, and a "Prediction vs Actual Target" scatter (grey dataset dots, blue actual, red predicted, dashed line annotated with the difference).
