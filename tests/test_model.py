from __future__ import annotations

import numpy as np
from joblib import load
from sklearn.linear_model import LinearRegression

from model import save_regression_model, train_regression_model


def test_train_regression_model_returns_fitted_linear_regression(tiny_dataset):
    X, y = tiny_dataset
    model = train_regression_model(X, y)

    assert isinstance(model, LinearRegression)
    # Dataset is y = 2x + 1 exactly, so the fit should be near-perfect.
    assert abs(model.coef_[0] - 2.0) < 1e-6
    assert abs(model.intercept_ - 1.0) < 1e-6


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
