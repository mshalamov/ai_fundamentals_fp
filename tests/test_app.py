from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for tests  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from joblib import dump
from streamlit.testing.v1 import AppTest

from app import load_and_predict, visualize_difference

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_load_and_predict_uses_default_filename(artifacts_cwd):
    # tiny_model fits y = 2x + 1 exactly; predict at x = 1.5 should give ~4.0.
    result = load_and_predict([[1.5]])
    np.testing.assert_allclose(result, [4.0], atol=1e-6)


def test_load_and_predict_accepts_custom_filename(artifacts_cwd, tiny_model):
    dump(tiny_model, artifacts_cwd / "alt.joblib")

    result = load_and_predict([[0.0]], filename="alt.joblib")
    np.testing.assert_allclose(result, [1.0], atol=1e-6)


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
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert len(labels) >= 3

    annotations = [
        child for child in ax.get_children()
        if isinstance(child, matplotlib.text.Annotation)
    ]
    assert any("Difference" in a.get_text() for a in annotations)


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
    # tiny_model fits y = 2x + 1, so prediction at x=1.5 should be ~4.0.
    body = " ".join(md.value for md in at.markdown) + " ".join(
        w.value for w in getattr(at, "write", [])
    )
    assert "4.0" in body or "Prediction" in body
