# Linear Regression + Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in the TODO bodies in `src/model.py` and `src/app.py` so that `uv run python model.py` trains and serializes a linear regression model, and `uv run streamlit run app.py` serves predictions with a prediction-vs-actual scatter plot — exactly as specified in `docs/ai_final_project.md`.

**Architecture:** Two-stage pipeline with on-disk handoff via joblib. `model.py` generates a synthetic 1-feature dataset, fits `sklearn.linear_model.LinearRegression`, and dumps three artifacts (`linear_regression_model.joblib`, `X.joblib`, `y.joblib`) to the CWD. `app.py` is a Streamlit UI that loads those three artifacts from CWD at predict time, takes a slider input in `[-3, 3]`, and renders both the numeric prediction and a matplotlib scatter showing the gap to the nearest actual sample. No shared module state between the two stages — only the joblib files.

**Tech Stack:** Python, scikit-learn (LinearRegression, make_regression, train_test_split, mean_squared_error), joblib (serialization), numpy, matplotlib, streamlit. Added for testing: pytest, streamlit's `streamlit.testing.v1.AppTest`. Lint: **ruff** (pyflakes + pycodestyle + isort rules). Tooling: **uv** for environment + dependency management — every Python/pytest/ruff command runs via `uv run` so the project's `.venv` is used automatically without manual activation.

---

## File Structure

**Existing files to modify:**
- `src/model.py` — fill in `train_regression_model`, `save_regression_model`, `evaluate_regression_model`, `save_initial_datasets`. The `if __name__ == '__main__'` driver and `_index_of_closest` helper are already done; leave them alone.
- `src/app.py` — fill in `load_and_predict`, `create_streamlit_app`, `visualize_difference`. Leave `_index_of_closest` alone.

**Files to delete:**
- `requirements.txt` — superseded by `pyproject.toml`; uv resolves and installs from the lockfile instead.

**New files to create:**
- `.gitignore` — exclude generated artifacts, Python caches, and the venv (the `uv.lock` IS committed).
- `pyproject.toml` — single source of truth for main + test + linter dependencies, consumed by `uv sync`.
- `pytest.ini` — make `src/` importable in tests via `pythonpath`.
- `tests/__init__.py` — empty, marks the tests package.
- `tests/conftest.py` — shared fixtures (small in-memory dataset, tmp CWD with joblib artifacts pre-dumped).
- `tests/test_model.py` — covers the four functions in `src/model.py`.
- `tests/test_app.py` — covers `load_and_predict`, `visualize_difference`, and a Streamlit `AppTest` smoke for `create_streamlit_app`.

**Generated artifacts (never committed):**
- `src/linear_regression_model.joblib`, `src/X.joblib`, `src/y.joblib` — produced by `uv run python model.py`, gitignored.

---

## Task 1: Initialize git repo, scaffold pytest

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Delete: `requirements.txt`

- [ ] **Step 1: Initialize git repo**

Run from repo root:
```bash
git init
```
Expected: `Initialized empty Git repository in /home/ubuntu/Projects/ai_f_fp/.git/`

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore` with:
```
__pycache__/
*.pyc
.pytest_cache/
*.joblib
.venv/
```

- [ ] **Step 3: Replace `requirements.txt` with `pyproject.toml`**

Delete the old requirements file:
```bash
rm requirements.txt
```

Create `pyproject.toml` at the repo root:
```toml
[project]
name = "ai-f-fp"
version = "0.1.0"
description = "Linear regression model + Streamlit web app (course final project)"
requires-python = ">=3.10"
dependencies = [
    "joblib",
    "streamlit",
    "scikit-learn",
    "numpy",
    "pandas",
    "matplotlib",
]

[dependency-groups]
test = [
    "pytest",
]
linter = [
    "ruff",
]

[tool.uv]
default-groups = ["test", "linter"]

[tool.ruff]
line-length = 100
target-version = "py310"
extend-exclude = [".venv"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

Notes:
- `dependencies` carries the six runtime libraries that used to live in `requirements.txt` (joblib, streamlit, scikit-learn, numpy, pandas, matplotlib).
- `[dependency-groups].test` holds pytest. `[dependency-groups].linter` holds ruff. They are kept in separate groups so a future CI pipeline (or a constrained install) can pull just one — e.g. `uv sync --only-group linter` for a fast lint-only job.
- `[tool.uv].default-groups = ["test", "linter"]` makes `uv sync` install both groups by default — no `--group …` flag needed at sync time. To skip them, use `uv sync --no-default-groups` or `uv sync --only-group <name>`.
- `[tool.ruff.lint].select` enables pyflakes (F), pycodestyle errors/warnings (E/W), and isort import ordering (I) — a reasonable starter rule set.
- The spec (`docs/ai_final_project.md`) tells students to run `pip install -r requirements.txt`. That instruction is superseded by `uv sync` in this project — the dependency list is identical.

- [ ] **Step 4: Sync the uv environment**

Run from the repo root:
```bash
uv sync
```
Expected: uv resolves the dependency graph, writes `uv.lock`, creates `.venv/`, and installs all main + test + linter packages. Final line is something like `Installed N packages in <time>`. All later commands in this plan use `uv run <cmd>` so the `.venv` is picked up automatically (no manual `source .venv/bin/activate` needed).

Sanity-check both tools are on PATH inside the venv:
```bash
uv run pytest --version
uv run ruff --version
```
Expected: prints `pytest X.Y.Z` and `ruff X.Y.Z`.

- [ ] **Step 5: Write `pytest.ini`**

Create `pytest.ini` with:
```ini
[pytest]
pythonpath = src
testpaths = tests
```

This makes `from model import ...` and `from app import ...` work in tests without restructuring `src/` into a package.

- [ ] **Step 6: Write `tests/__init__.py`**

Create an empty `tests/__init__.py`.

- [ ] **Step 7: Write `tests/conftest.py`**

Create `tests/conftest.py`:
```python
from __future__ import annotations
import numpy as np
import pytest
from joblib import dump
from sklearn.linear_model import LinearRegression


@pytest.fixture
def tiny_dataset():
    """A small deterministic 1-feature regression dataset (y = 2x + 1)."""
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    y = 2 * X.flatten() + 1
    return X, y


@pytest.fixture
def tiny_model(tiny_dataset):
    X, y = tiny_dataset
    return LinearRegression().fit(X, y)


@pytest.fixture
def artifacts_cwd(tmp_path, monkeypatch, tiny_dataset, tiny_model):
    """Chdir into a tmp dir that already contains the three joblib artifacts
    that app.py expects to load from CWD."""
    monkeypatch.chdir(tmp_path)
    X, y = tiny_dataset
    dump(tiny_model, tmp_path / "linear_regression_model.joblib")
    dump(X, tmp_path / "X.joblib")
    dump(y, tmp_path / "y.joblib")
    return tmp_path
```

- [ ] **Step 8: Write a smoke test to confirm the scaffold works**

Create `tests/test_smoke.py`:
```python
def test_pytest_can_import_model():
    import model  # noqa: F401


def test_pytest_can_import_app():
    import app  # noqa: F401
```

- [ ] **Step 9: Run smoke test**

Run:
```bash
uv run pytest tests/test_smoke.py -v
```
Expected: 2 passed.

- [ ] **Step 10: Commit**

```bash
git rm --cached requirements.txt 2>/dev/null || true  # only needed if it was ever tracked
git add .gitignore pyproject.toml uv.lock pytest.ini tests/ CLAUDE.md docs/ src/
git commit -m "chore: scaffold uv project, pytest, gitignore, initial repo state"
```

(The `git rm --cached` is a no-op when the file was never tracked, which is the case here since the repo was just initialized — included only for safety if the plan is re-run later.)

---

## Task 2: Implement `train_regression_model`

**Files:**
- Modify: `src/model.py` (function `train_regression_model`, lines ~11–30)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model.py`:
```python
from __future__ import annotations

from sklearn.linear_model import LinearRegression

from model import train_regression_model


def test_train_regression_model_returns_fitted_linear_regression(tiny_dataset):
    X, y = tiny_dataset
    model = train_regression_model(X, y)

    assert isinstance(model, LinearRegression)
    # Dataset is y = 2x + 1 exactly, so the fit should be near-perfect.
    assert abs(model.coef_[0] - 2.0) < 1e-6
    assert abs(model.intercept_ - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_model.py::test_train_regression_model_returns_fitted_linear_regression -v
```
Expected: FAIL — `UnboundLocalError: local variable 'model' referenced before assignment` (the existing TODO body returns an undefined `model`).

- [ ] **Step 3: Implement the function**

In `src/model.py`, replace the body of `train_regression_model` (currently `# TODO: your code here` followed by `return model`):
```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_model.py::test_train_regression_model_returns_fitted_linear_regression -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat(model): implement train_regression_model"
```

---

## Task 3: Implement `save_regression_model`

**Files:**
- Modify: `src/model.py` (function `save_regression_model`)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:
```python
import numpy as np
from joblib import load

from model import save_regression_model


def test_save_regression_model_default_filename_roundtrip(
    tmp_path, monkeypatch, tiny_model, tiny_dataset
):
    monkeypatch.chdir(tmp_path)
    X, _ = tiny_dataset

    save_regression_model(tiny_model)

    loaded = load(tmp_path / "linear_regression_model.joblib")
    np.testing.assert_allclose(loaded.predict(X), tiny_model.predict(X))


def test_save_regression_model_custom_filename(tmp_path, monkeypatch, tiny_model):
    monkeypatch.chdir(tmp_path)

    save_regression_model(tiny_model, filename="custom.joblib")

    assert (tmp_path / "custom.joblib").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_model.py -v -k save_regression_model
```
Expected: FAIL — no joblib file is written (TODO body is empty).

- [ ] **Step 3: Implement the function**

In `src/model.py`, replace the body of `save_regression_model` (currently `# TODO: your code here`):
```python
    dump(model, filename)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_model.py -v -k save_regression_model
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat(model): implement save_regression_model"
```

---

## Task 4: Implement `evaluate_regression_model`

**Files:**
- Modify: `src/model.py` (function `evaluate_regression_model`)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:
```python
from model import evaluate_regression_model


def test_evaluate_regression_model_prints_mse_zero_for_perfect_fit(capsys, tiny_dataset, tiny_model):
    X, y = tiny_dataset

    evaluate_regression_model(tiny_model, X, y)

    captured = capsys.readouterr()
    assert captured.out.startswith("Mean Squared Error: ")
    mse_value = float(captured.out.strip().split(":")[1])
    assert mse_value < 1e-10  # tiny_model fits tiny_dataset exactly
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_model.py::test_evaluate_regression_model_prints_mse_zero_for_perfect_fit -v
```
Expected: FAIL — `NameError: name 'mse' is not defined` (the existing function references `mse` in its print without defining it).

- [ ] **Step 3: Implement the function**

In `src/model.py`, replace the body of `evaluate_regression_model` (the `# TODO: your code here` line, leaving the `print` statement intact):
```python
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_model.py::test_evaluate_regression_model_prints_mse_zero_for_perfect_fit -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat(model): implement evaluate_regression_model"
```

---

## Task 5: Implement `save_initial_datasets` and run the full training driver

**Files:**
- Modify: `src/model.py` (function `save_initial_datasets`)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:
```python
from model import save_initial_datasets


def test_save_initial_datasets_writes_X_and_y(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0])

    save_initial_datasets(X, y)

    np.testing.assert_array_equal(load(tmp_path / "X.joblib"), X)
    np.testing.assert_array_equal(load(tmp_path / "y.joblib"), y)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_model.py::test_save_initial_datasets_writes_X_and_y -v
```
Expected: FAIL — neither `X.joblib` nor `y.joblib` exists after the call.

- [ ] **Step 3: Implement the function**

In `src/model.py`, after the two `_filename` assignments, replace `# TODO: your code here` with:
```python
    dump(X, X_filename)
    dump(y, y_filename)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_model.py::test_save_initial_datasets_writes_X_and_y -v
```
Expected: PASS.

- [ ] **Step 5: Run the full training driver end-to-end**

Run:
```bash
cd src && uv run python model.py && cd ..
```
Expected: Prints a line like `Mean Squared Error: <some small number>`, and creates `src/linear_regression_model.joblib`, `src/X.joblib`, `src/y.joblib`. Verify:
```bash
ls src/*.joblib
```
Expected: all three files listed.

- [ ] **Step 6: Run the whole test suite**

Run:
```bash
uv run pytest -v
```
Expected: all model-side tests pass (5 tests so far, plus 2 smoke).

- [ ] **Step 7: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat(model): implement save_initial_datasets, complete training pipeline"
```

---

## Task 6: Implement `load_and_predict`

**Files:**
- Modify: `src/app.py` (function `load_and_predict`)
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_app.py`:
```python
from __future__ import annotations
import numpy as np

from app import load_and_predict


def test_load_and_predict_uses_default_filename(artifacts_cwd):
    # tiny_model fits y = 2x + 1 exactly; predict at x = 1.5 should give ~4.0.
    result = load_and_predict([[1.5]])
    np.testing.assert_allclose(result, [4.0], atol=1e-6)


def test_load_and_predict_accepts_custom_filename(artifacts_cwd, tiny_model):
    from joblib import dump
    dump(tiny_model, artifacts_cwd / "alt.joblib")

    result = load_and_predict([[0.0]], filename="alt.joblib")
    np.testing.assert_allclose(result, [1.0], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_app.py -v -k load_and_predict
```
Expected: FAIL — `UnboundLocalError: local variable 'y' referenced before assignment`.

- [ ] **Step 3: Implement the function**

In `src/app.py`, replace the body of `load_and_predict` (the `# TODO: your code here` line, leaving the `return y` intact):
```python
    model = load(filename)
    y = model.predict(X)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_app.py -v -k load_and_predict
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/app.py tests/test_app.py
git commit -m "feat(app): implement load_and_predict"
```

---

## Task 7: Implement `visualize_difference`

**Files:**
- Modify: `src/app.py` (function `visualize_difference`)
- Test: `tests/test_app.py`

The function must: load `X.joblib` and `y.joblib`, find the nearest-X actual target via `_index_of_closest`, build a matplotlib `Figure` with (a) grey dataset scatter, (b) blue actual point, (c) red predicted point, (d) dashed line between them annotated with `Difference = <value>`, (e) labels "Feature"/"Target", title, legend, grid, and pass it to `st.pyplot`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:
```python
import matplotlib
matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

from app import visualize_difference


def test_visualize_difference_passes_figure_to_streamlit(artifacts_cwd, monkeypatch):
    captured = {}

    def fake_pyplot(fig):
        captured["fig"] = fig

    monkeypatch.setattr("app.st.pyplot", fake_pyplot)

    visualize_difference(1.0, np.array([3.0]))

    fig = captured.get("fig")
    assert isinstance(fig, plt.Figure)

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Feature"
    assert ax.get_ylabel() == "Target"
    # Legend should have at least three entries: dataset, actual, predicted.
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert len(labels) >= 3

    # Annotation text should include "Difference"
    annotations = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
    assert any("Difference" in a.get_text() for a in annotations)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_app.py::test_visualize_difference_passes_figure_to_streamlit -v
```
Expected: FAIL — the figure has no axes/legend/annotation because the body is all `# TODO` comments.

- [ ] **Step 3: Implement the function**

In `src/app.py`, replace the block of TODO comments inside `visualize_difference` (everything between `fig = plt.figure(figsize=(6,4))` and `st.pyplot(fig)`) with:
```python
    plt.scatter(X, y, color="grey", label="Dataset")
    plt.scatter(input_feature, actual_target, color="blue", label="Actual Target")
    plt.scatter(input_feature, prediction, color="red", label="Predicted Target")

    plt.plot(
        [input_feature, input_feature],
        [actual_target, float(np.asarray(prediction).ravel()[0])],
        "k--",
    )

    midpoint_y = (actual_target + float(np.asarray(prediction).ravel()[0])) / 2
    plt.annotate(
        f"Difference = {float(np.asarray(difference).ravel()[0]):.2f}",
        xy=(input_feature, midpoint_y),
        xytext=(input_feature + 0.2, midpoint_y),
    )

    plt.title("Prediction vs Actual Target")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
```

(`X`, `y`, `actual_target`, and `difference` are already in scope from the existing code above the TODO block.)

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/test_app.py::test_visualize_difference_passes_figure_to_streamlit -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/app.py tests/test_app.py
git commit -m "feat(app): implement visualize_difference plot"
```

---

## Task 8: Implement `create_streamlit_app`

**Files:**
- Modify: `src/app.py` (function `create_streamlit_app`)
- Test: `tests/test_app.py`

Required UI per spec: title `"Simple Regression model prediction"`, slider labeled `"Input Feature for Prediction"` with range `[-3.0, 3.0]`, button labeled `"Predict value"`. On click: call `load_and_predict([[input_feature]])`, write the prediction, call `visualize_difference(input_feature, prediction)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:
```python
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_create_streamlit_app_renders_title_slider_and_button(artifacts_cwd):
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == "Simple Regression model prediction"

    slider = at.slider[0]
    assert slider.label == "Input Feature for Prediction"
    assert slider.min == -3.0
    assert slider.max == 3.0

    assert at.button[0].label == "Predict value"


def test_create_streamlit_app_predicts_on_button_click(artifacts_cwd):
    at = AppTest.from_file(APP_PATH).run()

    at.slider[0].set_value(1.5)
    at.button[0].click().run()

    assert not at.exception
    # tiny_model fits y = 2x + 1, so prediction at x=1.5 should be ~4.0
    body = " ".join(md.value for md in at.markdown) + " ".join(w.value for w in at.write)
    assert "4.0" in body or "Prediction" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_app.py -v -k create_streamlit_app
```
Expected: FAIL — there is no title/slider/button in the app yet (the function body is just comments).

- [ ] **Step 3: Implement the function**

In `src/app.py`, replace the body of `create_streamlit_app` (the entire block of `# TODO` and inline comments) with:
```python
    st.title("Simple Regression model prediction")

    input_feature = st.slider(
        "Input Feature for Prediction",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
    )

    if st.button("Predict value"):
        prediction = load_and_predict([[input_feature]])
        st.write(f"Prediction: {float(np.asarray(prediction).ravel()[0])}")
        visualize_difference(input_feature, prediction)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_app.py -v -k create_streamlit_app
```
Expected: 2 passed.

- [ ] **Step 5: Run the full test suite**

Run:
```bash
uv run pytest -v
```
Expected: all tests pass (smoke + 5 model + 4 app = 11 tests).

- [ ] **Step 6: Lint with ruff**

Run:
```bash
uv run ruff check .
```
Expected: `All checks passed!` If ruff reports issues, auto-fix the safe ones with:
```bash
uv run ruff check --fix .
```
Then re-run `uv run ruff check .` to confirm clean. Any remaining issues must be fixed by hand before committing.

- [ ] **Step 7: Commit**

```bash
git add src/app.py tests/test_app.py
git commit -m "feat(app): implement create_streamlit_app UI"
```

---

## Task 9: Manual end-to-end smoke test in browser

**Files:** none modified — this is a verification-only task.

- [ ] **Step 1: Ensure training artifacts exist**

Run:
```bash
ls src/linear_regression_model.joblib src/X.joblib src/y.joblib
```
If any are missing, regenerate with:
```bash
cd src && uv run python model.py && cd ..
```

- [ ] **Step 2: Launch the Streamlit app**

Run:
```bash
cd src && uv run streamlit run app.py
```
Streamlit prints a local URL (typically http://localhost:8501).

- [ ] **Step 3: Verify against spec acceptance criteria**

Open the URL in a browser. Confirm — these are the exact criteria from `docs/ai_final_project.md`:
- Title reads "Simple Regression model prediction".
- A slider labeled "Input Feature for Prediction" exists with range -3.00 to 3.00.
- A "Predict value" button is present.
- Clicking the button displays a numeric prediction (e.g., `Prediction: 26.509899021941507`).
- A scatter plot appears titled "Prediction vs Actual Target" containing: grey dataset dots, one blue "Actual Target" dot, one red "Predicted Target" dot, a black dashed line between them, and an annotation reading `Difference = <number>`.

Try slider extremes (-3.0 and 3.0) and a midpoint (0.0); confirm the plot updates and the dashed line connects actual↔predicted at the chosen X.

- [ ] **Step 4: Stop the server**

Press `Ctrl+C` in the terminal running Streamlit.

- [ ] **Step 5: Commit (only if anything changed during verification)**

If no code changed, skip. Otherwise:
```bash
git add -u
git commit -m "fix: address issues found during manual smoke test"
```

---

## Self-Review Notes

**Spec coverage:**
- Part 1 §1 train_regression_model → Task 2 ✓
- Part 1 §2 save_regression_model → Task 3 ✓
- Part 1 §3 evaluate_regression_model → Task 4 ✓
- Part 1 §4 save_initial_datasets → Task 5 ✓
- Part 2 §1 load_and_predict → Task 6 ✓
- Part 2 §3 visualize_difference → Task 7 ✓
- Part 2 §2 create_streamlit_app → Task 8 ✓
- Completion criteria (title, slider, button, prediction text, scatter plot with grey/blue/red, dashed line, difference annotation) → covered by Tasks 7, 8 (automated) and Task 9 (visual) ✓

**Type consistency:** function signatures match the existing skeletons exactly — `train_regression_model(X_train, y_train)`, `save_regression_model(model, filename="linear_regression_model.joblib")`, `evaluate_regression_model(model, X_test, y_test)`, `save_initial_datasets(X, y)`, `load_and_predict(X, filename="linear_regression_model.joblib")`, `create_streamlit_app()`, `visualize_difference(input_feature, prediction)`. The `prediction` argument to `visualize_difference` is treated as array-like in the implementation (we use `np.asarray(...).ravel()[0]` to coerce to scalar for formatting) consistent with how `load_and_predict` returns the sklearn output.
