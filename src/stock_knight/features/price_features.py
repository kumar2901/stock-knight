from __future__ import annotations

import numpy as np
import pandas as pd


def _cs_zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def compute_price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-safe features from OHLCV.

    Input: long-format dataframe with columns:
      date, symbol, open, high, low, close, volume

    Output: long-format features indexed by (date, symbol) with numeric columns.
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"])

    g = df.groupby("symbol", group_keys=False)
    close = g["close"]

    # Returns
    df["ret_1d"] = close.pct_change()
    for n in [5, 21, 63, 126, 252]:
        df[f"mom_{n}d"] = close.pct_change(n)

    # Volatility (realized)
    for n in [10, 21, 63]:
        df[f"vol_{n}d"] = g["ret_1d"].rolling(n).std().reset_index(level=0, drop=True)

    # Trend (moving average distance)
    for n in [20, 50, 200]:
        ma = close.rolling(n).mean().reset_index(level=0, drop=True)
        df[f"ma_dist_{n}"] = (df["close"] / ma) - 1.0

    # Liquidity proxies
    df["traded_value"] = df["close"] * df["volume"]
    for n in [21, 63]:
        tv_med = g["traded_value"].rolling(n).median().reset_index(level=0, drop=True)
        df[f"tv_med_{n}"] = tv_med

    # Cross-sectional z-scores per day for selected columns
    feature_cols = [
        "mom_5d",
        "mom_21d",
        "mom_63d",
        "mom_126d",
        "mom_252d",
        "vol_10d",
        "vol_21d",
        "vol_63d",
        "ma_dist_20",
        "ma_dist_50",
        "ma_dist_200",
        "tv_med_21",
        "tv_med_63",
    ]
    out = df[["date", "symbol"] + feature_cols].copy()
    out = out.set_index(["date", "symbol"]).sort_index()

    for c in feature_cols:
        out[c + "_z"] = out.groupby(level=0)[c].transform(_cs_zscore)

    return out.drop(columns=feature_cols)

