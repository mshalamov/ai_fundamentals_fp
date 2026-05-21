# AI Final Project — Linear Regression + Streamlit

A small end-to-end ML deliverable: train a scikit-learn `LinearRegression` model on a synthetic 1-feature dataset, serialize the model and data to disk, then serve interactive predictions through a Streamlit web app that visualizes how each prediction compares to the nearest real sample.

Full assignment spec: [`docs/ai_final_project.md`](docs/ai_final_project.md).

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) for dependency / virtual-environment management

That's it — uv handles the rest from `pyproject.toml` / `uv.lock`.

## Quickstart

```bash
# 1. Install dependencies (creates .venv automatically).
uv sync

# 2. Train the model and dump the joblib artifacts into src/.
cd src && uv run python model.py
# → prints "Mean Squared Error: <value>" and writes
#   linear_regression_model.joblib, X.joblib, y.joblib

# 3. Launch the Streamlit UI (still from src/, so it can find the artifacts).
uv run streamlit run app.py
```

Open the URL Streamlit prints (typically <http://localhost:8501>), pick a feature value with the slider, click **Predict value**, and the page shows the numeric prediction plus a scatter plot with the actual-vs-predicted difference annotated.

> Both the training script and the Streamlit app must be run from inside `src/` — `app.py` loads the joblib artifacts via bare filenames, so the working directory matters.

## Project Layout

```
.
├── pyproject.toml          # uv-managed deps + ruff config (single source of truth)
├── uv.lock                 # committed lockfile
├── pytest.ini              # pytest pythonpath = src
├── src/
│   ├── model.py            # training pipeline (generates dataset, fits, dumps joblib)
│   └── app.py              # Streamlit UI (load model, predict, plot)
├── tests/
│   ├── conftest.py         # tiny_dataset / tiny_model / artifacts_cwd fixtures
│   ├── test_model.py       # one test per public function in src/model.py
│   └── test_app.py         # load_and_predict, visualize_difference, AppTest UI cases
└── docs/
    ├── ai_final_project.md # assignment spec
    └── superpowers/plans/  # implementation plan
```

The three `*.joblib` files generated in `src/` are gitignored — regenerate them by re-running `uv run python model.py` whenever the model definition changes.

## Tests & Lint

```bash
uv run pytest -v        # 12 tests, runs in well under a second
uv run ruff check .     # rules: E, F, I, W (pyflakes + pycodestyle + isort)
```

`uv sync` installs both the `test` and `linter` dependency groups by default (configured via `[tool.uv].default-groups` in `pyproject.toml`).

## How It Works

Two-stage pipeline with on-disk handoff via joblib — no shared module state between training and serving:

1. **`src/model.py`** uses `sklearn.datasets.make_regression` (n=100, noise=20, seed=42) to build a 1-feature dataset, rescales `X` into `[-3, 3]`, splits 80/20, fits `LinearRegression`, prints the MSE on the test split, then dumps the trained model plus the full `X` / `y` arrays.

2. **`src/app.py`** loads the three artifacts at predict time. The slider range `[-3.0, 3.0]` deliberately matches the rescaling in step 1. `visualize_difference` finds the "actual" target for the chosen slider value via nearest-X lookup against the saved `X` / `y` (helper: `_index_of_closest`) and draws the dataset, the actual point, the predicted point, a dashed line between them, and a `Difference = …` annotation.
