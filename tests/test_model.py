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
