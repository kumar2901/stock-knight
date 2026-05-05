"""Tests for the regime trend feature (Phase 3b)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_knight.features.price_features import compute_price_features
from stock_knight.features.regime import compute_regime_trend


def _toy_ohlcv() -> pd.DataFrame:
    """Build a tiny OHLCV frame with a strong trending phase then a chop phase."""
    rng = np.random.default_rng(11)
    n_days = 80
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    syms = ["A", "B", "C"]

    rows = []
    for sym in syms:
        # First 40 days: strong uptrend (~1%/day). Next 40: chop (~0%).
        rets = np.r_[
            rng.normal(0.01, 0.005, 40),
            rng.normal(0.0, 0.005, 40),
        ]
        prices = 100.0 * np.cumprod(1 + rets)
        for d, p in zip(dates, prices, strict=True):
            rows.append((d, sym, p, p, p, p, 1_000_000))
    return pd.DataFrame(
        rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"]
    )


def test_regime_trend_positive_during_uptrend_lower_during_chop() -> None:
    ohlcv = _toy_ohlcv()
    regime = compute_regime_trend(ohlcv)
    assert "regime_trend_z" in regime.columns

    # Sample mid-uptrend and mid-chop dates from the same series.
    by_date = regime.groupby(level=0)["regime_trend_z"].first()
    uptrend_z = by_date.loc[pd.Timestamp("2024-02-15")]   # ~30 days into uptrend
    chop_z = by_date.loc[pd.Timestamp("2024-04-15")]      # ~20+ days into chop

    assert uptrend_z > 0
    assert uptrend_z > chop_z


def test_compute_price_features_emits_regime_trend_z() -> None:
    ohlcv = _toy_ohlcv()
    feats = compute_price_features(ohlcv)
    assert "regime_trend_z" in feats.columns
    # Same value for every symbol on the same date (it's a market-wide signal).
    sample = feats.xs(pd.Timestamp("2024-02-20"), level=0)["regime_trend_z"]
    assert np.allclose(sample.to_numpy(), sample.iloc[0])
