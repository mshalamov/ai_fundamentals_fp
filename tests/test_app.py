from __future__ import annotations

import numpy as np
from joblib import dump

from app import load_and_predict


def test_load_and_predict_uses_default_filename(artifacts_cwd):
    # tiny_model fits y = 2x + 1 exactly; predict at x = 1.5 should give ~4.0.
    result = load_and_predict([[1.5]])
    np.testing.assert_allclose(result, [4.0], atol=1e-6)


def test_load_and_predict_accepts_custom_filename(artifacts_cwd, tiny_model):
    dump(tiny_model, artifacts_cwd / "alt.joblib")

    result = load_and_predict([[0.0]], filename="alt.joblib")
    np.testing.assert_allclose(result, [1.0], atol=1e-6)
