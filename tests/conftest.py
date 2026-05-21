from __future__ import annotations
from pathlib import Path
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
