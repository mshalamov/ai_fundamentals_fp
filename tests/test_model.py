from __future__ import annotations

import numpy as np
from joblib import load
from sklearn.linear_model import LinearRegression

from model import (
    evaluate_regression_model,
    save_regression_model,
    train_regression_model,
)


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


def test_evaluate_regression_model_prints_mse_zero_for_perfect_fit(
    capsys, tiny_dataset, tiny_model
):
    X, y = tiny_dataset

    evaluate_regression_model(tiny_model, X, y)

    captured = capsys.readouterr()
    assert captured.out.startswith("Mean Squared Error: ")
    mse_value = float(captured.out.strip().split(":")[1])
    assert mse_value < 1e-10  # tiny_model fits tiny_dataset exactly
