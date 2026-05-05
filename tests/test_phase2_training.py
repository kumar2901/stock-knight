"""Smoke tests for Phase-2 training upgrades: ensemble, LambdaRank, winsorize."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from stock_knight.train.models import (
    TrainedModel,
    load_model,
    save_model,
    train_ensemble,
    train_lgbm_rank,
    winsorize_per_date,
)


def test_ensemble_trains_and_predicts(
    small_training_set: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = small_training_set
    tm = train_ensemble(X, y, horizon="7d")
    assert isinstance(tm, TrainedModel)
    assert tm.kind == "ensemble"
    assert len(tm.member_kinds) >= 1
    pred = tm.predict(X.head(50))
    assert pred.shape == (50,)
    # Rank-averaging produces values in [0, 1].
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)
    assert np.isfinite(pred).all()


def test_ensemble_save_load_roundtrip(
    small_training_set: tuple[pd.DataFrame, pd.Series], tmp_path
) -> None:
    X, y = small_training_set
    tm = train_ensemble(X, y, horizon="7d", members=("ridge", "rf"))
    p = tmp_path / "ensemble_7d.joblib"
    save_model(p, tm)
    loaded = load_model(p)
    assert loaded.kind == "ensemble"
    assert set(loaded.member_kinds) == {"ridge", "rf"}
    np.testing.assert_allclose(
        loaded.predict(X.head(30)), tm.predict(X.head(30)), rtol=1e-6
    )


def test_lambdarank_trains_and_predicts(
    small_training_set: tuple[pd.DataFrame, pd.Series],
) -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed")
    X, y = small_training_set
    tm = train_lgbm_rank(X, y, horizon="7d")
    assert tm.kind == "lgbm_rank"
    pred = tm.predict(X.head(50))
    assert pred.shape == (50,)
    assert np.isfinite(pred).all()


def test_winsorize_per_date_clips_extremes() -> None:
    dates = pd.to_datetime(["2024-01-01"] * 100 + ["2024-01-02"] * 100)
    syms = [f"S{i}" for i in range(100)] * 2
    idx = pd.MultiIndex.from_arrays([dates, syms], names=["date", "symbol"])
    rng = np.random.default_rng(7)
    base_d1 = rng.normal(0.0, 0.02, 99).tolist() + [5.0]   # outlier +5.0 on day 1
    base_d2 = rng.normal(0.0, 0.02, 99).tolist() + [-5.0]  # outlier -5.0 on day 2
    y = pd.Series(base_d1 + base_d2, index=idx, name="r_fwd")

    w = winsorize_per_date(y, lower=0.01, upper=0.99)
    # Outliers must be pulled in significantly (down from |5| to ~ 0.05).
    assert w.loc[("2024-01-01", "S99")] < 0.5
    assert w.loc[("2024-01-02", "S99")] > -0.5
    # Cohort-bulk values (well within the percentile range) should be unchanged.
    assert w.loc[("2024-01-01", "S5")] == pytest.approx(base_d1[5], abs=1e-9)
