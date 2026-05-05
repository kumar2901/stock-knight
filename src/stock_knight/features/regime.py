"""
Market regime features.

A per-DATE feature that indicates whether the broad market is in a trending
or choppy regime. Tree-based models (lgbm/xgb/rf) can split on this column
to learn regime-conditional patterns ("momentum works in trending markets,
mean-reversion works in chop"). Ridge will mostly ignore it (it's a per-day
intercept shift), which is fine.

Construction
------------
1. Daily market return = cross-sectional mean of per-stock 1-day returns.
2. ``mkt_20d_trend`` = rolling 20-day SUM of market returns.
3. ``mkt_20d_vol``   = rolling 20-day STD of market returns.
4. ``regime_trend`` = sign(trend) * min(|trend| / vol, 3.0).
   - Positive in strong-uptrend regimes.
   - Near-zero in choppy markets.
   - Negative (and bounded at -3) in strong-downtrend regimes.
5. ``regime_trend_z`` = global (over-time) standardisation of ``regime_trend``
   so it sits roughly in [-2, 2].

We standardise over time (not cross-sectionally per date) because the value
is constant across symbols within a date — a cross-sectional z-score would
collapse to zero. Keeping the column name ending in ``_z`` lets the existing
``feature_cols = [c for c in df.columns if c.endswith("_z")]`` selector pick
it up automatically in trainers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_TREND_CAP: float = 3.0
REGIME_LOOKBACK_DAYS: int = 20


def compute_regime_trend(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the regime_trend (un-standardised) per (date, symbol).

    The value is identical for every symbol on a given date — it's a
    market-wide signal broadcast for tree models. Returns a DataFrame
    indexed by (date, symbol) with a single ``regime_trend_z`` column
    (already standardised globally over time).
    """
    df = ohlcv[["date", "symbol", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"])

    df["_ret"] = df.groupby("symbol")["close"].transform(lambda s: s.pct_change())

    mkt = (
        df.dropna(subset=["_ret"])
        .groupby("date")["_ret"]
        .mean()
        .sort_index()
    )

    trend = mkt.rolling(REGIME_LOOKBACK_DAYS).sum()
    vol = mkt.rolling(REGIME_LOOKBACK_DAYS).std()

    # Avoid div-by-zero on flat days.
    safe_vol = vol.replace(0, np.nan)
    raw = np.sign(trend) * (trend.abs() / safe_vol).clip(upper=REGIME_TREND_CAP)

    # Global (over-time) standardisation.
    mu = float(raw.mean())
    sd = float(raw.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        z = raw * 0.0
    else:
        z = (raw - mu) / sd

    z.name = "regime_trend_z"

    # Broadcast back to (date, symbol). We keep the same row index that the
    # rest of the feature pipeline produces.
    out = (
        df[["date", "symbol"]]
        .assign(regime_trend_z=df["date"].map(z))
        .set_index(["date", "symbol"])
        .sort_index()
    )
    out["regime_trend_z"] = out["regime_trend_z"].astype(float).fillna(0.0)
    return out
