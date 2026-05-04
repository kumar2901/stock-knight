"""Smoke tests for the new XGBoost and Random Forest trainers."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from stock_knight.train.models import TrainedModel, train_rf, train_xgb


def test_random_forest_trains_and_predicts(
    small_training_set: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = small_training_set
    tm = train_rf(X, y, horizon="7d")
    assert isinstance(tm, TrainedModel)
    assert tm.kind == "rf"
    assert len(tm.feature_cols) > 0
    pred = tm.predict(X.head(50))
    assert pred.shape == (50,)
    assert np.isfinite(pred).all()


def test_xgboost_trains_and_predicts(
    small_training_set: tuple[pd.DataFrame, pd.Series],
) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost not installed")
    X, y = small_training_set
    tm = train_xgb(X, y, horizon="7d")
    assert isinstance(tm, TrainedModel)
    assert tm.kind == "xgb"
    pred = tm.predict(X.head(50))
    assert pred.shape == (50,)
    assert np.isfinite(pred).all()


def test_save_and_load_roundtrip(
    small_training_set: tuple[pd.DataFrame, pd.Series], tmp_path
) -> None:
    from stock_knight.train.models import load_model, save_model

    X, y = small_training_set
    tm = train_rf(X, y, horizon="7d")
    p = tmp_path / "rf_7d.joblib"
    save_model(p, tm)
    loaded = load_model(p)
    assert loaded.kind == "rf"
    assert loaded.feature_cols == tm.feature_cols
    np.testing.assert_allclose(
        loaded.predict(X.head(20)), tm.predict(X.head(20)), rtol=1e-6
    )
