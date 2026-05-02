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


def _grouped_ewm(df: pd.DataFrame, col: str, alpha: float) -> pd.Series:
    """EWM per symbol, returns a Series aligned with df.index."""
    return df.groupby("symbol")[col].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()
    )


def _grouped_rolling_std(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(n).std())


def _grouped_rolling_mean(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(n).mean())


def _grouped_rolling_median(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(n).median())


def _grouped_rolling_max(df: pd.DataFrame, col: str, n: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or n
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(n, min_periods=mp).max())


def _grouped_rolling_min(df: pd.DataFrame, col: str, n: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or n
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(n, min_periods=mp).min())


def _grouped_pct_change(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    return df.groupby("symbol")[col].transform(lambda s: s.pct_change(n))


def _grouped_shift(df: pd.DataFrame, col: str, n: int = 1) -> pd.Series:
    return df.groupby("symbol")[col].transform(lambda s: s.shift(n))


def compute_price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-safe features from OHLCV.

    Input: long-format dataframe with columns:
      date, symbol, open, high, low, close, volume

    Output: (date, symbol) indexed DataFrame. Every column ends with _z
    (cross-sectional z-score across all stocks on that day).
    30 features covering momentum, RSI, MACD, Bollinger Bands, ATR,
    volatility, trend, 52-week levels, volume, and liquidity.
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g_close = df.groupby("symbol")["close"]

    # ── Returns ───────────────────────────────────────────────────────────────
    df["ret_1d"] = g_close.transform(lambda s: s.pct_change())

    # ── Momentum ──────────────────────────────────────────────────────────────
    for n in [5, 21, 63, 126, 252]:
        df[f"mom_{n}d"] = _grouped_pct_change(df, "close", n)

    # Acceleration: short momentum minus longer momentum (momentum of momentum)
    df["mom_accel_5_21"] = df["mom_5d"] - df["mom_21d"]
    df["mom_accel_21_63"] = df["mom_21d"] - df["mom_63d"]

    # ── RSI ───────────────────────────────────────────────────────────────────
    for period in [7, 14]:
        def _rsi(s: pd.Series, p: int = period) -> pd.Series:
            delta = s.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        df[f"rsi_{period}"] = g_close.transform(_rsi)

    # ── MACD (normalized by price) ────────────────────────────────────────────
    ema12 = _grouped_ewm(df, "close", alpha=2 / 13)
    ema26 = _grouped_ewm(df, "close", alpha=2 / 27)
    df["_macd_line"] = ema12 - ema26
    macd_signal = df.groupby("symbol")["_macd_line"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    price_base = df["close"].replace(0, np.nan)
    df["macd_norm"] = df["_macd_line"] / price_base
    df["macd_hist_norm"] = (df["_macd_line"] - macd_signal) / price_base
    df.drop(columns=["_macd_line"], inplace=True)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    ma20 = _grouped_rolling_mean(df, "close", 20)
    std20 = _grouped_rolling_std(df, "close", 20)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    df["bb_pct_20"] = (df["close"] - lower) / band_width
    df["bb_width_20"] = band_width / ma20.replace(0, np.nan)

    # ── ATR % (volatility normalized by price) ────────────────────────────────
    prev_close = _grouped_shift(df, "close", 1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df["_tr"] = tr
    atr14 = df.groupby("symbol")["_tr"].transform(
        lambda s: s.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )
    df["atr_pct"] = atr14 / price_base
    df.drop(columns=["_tr"], inplace=True)

    # ── Realized volatility ───────────────────────────────────────────────────
    for n in [10, 21, 63]:
        df[f"vol_{n}d"] = _grouped_rolling_std(df, "ret_1d", n)

    df["vol_ratio_10_63"] = df["vol_10d"] / df["vol_63d"].replace(0, np.nan)

    # ── Moving average distance ───────────────────────────────────────────────
    for n in [20, 50, 200]:
        ma = _grouped_rolling_mean(df, "close", n).replace(0, np.nan)
        df[f"ma_dist_{n}"] = (df["close"] / ma) - 1.0

    # ── 52-week high/low distance ─────────────────────────────────────────────
    high_252 = _grouped_rolling_max(df, "high", 252, min_periods=63).replace(0, np.nan)
    low_252 = _grouped_rolling_min(df, "low", 252, min_periods=63).replace(0, np.nan)
    df["dist_52wk_high"] = (df["close"] / high_252) - 1.0
    df["dist_52wk_low"] = (df["close"] / low_252) - 1.0

    # ── Overnight gap ─────────────────────────────────────────────────────────
    df["overnight_gap"] = (df["open"] / prev_close.replace(0, np.nan)) - 1.0

    # ── Intraday range (21-day avg) ───────────────────────────────────────────
    df["_hl_range"] = (df["high"] - df["low"]) / price_base
    df["hl_range_21d"] = _grouped_rolling_mean(df, "_hl_range", 21)
    df.drop(columns=["_hl_range"], inplace=True)

    # ── Volume momentum ───────────────────────────────────────────────────────
    vol_ma_5 = _grouped_rolling_mean(df, "volume", 5)
    vol_ma_21 = _grouped_rolling_mean(df, "volume", 21)
    vol_ma_63 = _grouped_rolling_mean(df, "volume", 63)
    df["vol_mom_5_21"] = (vol_ma_5 / vol_ma_21.replace(0, np.nan)) - 1.0
    df["vol_mom_21_63"] = (vol_ma_21 / vol_ma_63.replace(0, np.nan)) - 1.0

    # ── Liquidity (traded value median) ───────────────────────────────────────
    df["_tv"] = df["close"] * df["volume"]
    for n in [21, 63]:
        df[f"tv_med_{n}"] = _grouped_rolling_median(df, "_tv", n)
    df.drop(columns=["_tv"], inplace=True)

    # ── Cross-sectional z-score ────────────────────────────────────────────────
    feature_cols = [
        "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
        "mom_accel_5_21", "mom_accel_21_63",
        "rsi_7", "rsi_14",
        "macd_norm", "macd_hist_norm",
        "bb_pct_20", "bb_width_20",
        "atr_pct",
        "vol_10d", "vol_21d", "vol_63d", "vol_ratio_10_63",
        "ma_dist_20", "ma_dist_50", "ma_dist_200",
        "dist_52wk_high", "dist_52wk_low",
        "overnight_gap", "hl_range_21d",
        "vol_mom_5_21", "vol_mom_21_63",
        "tv_med_21", "tv_med_63",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    out = df[["date", "symbol"] + feature_cols].copy()
    out = out.set_index(["date", "symbol"]).sort_index()

    for c in feature_cols:
        out[c + "_z"] = out.groupby(level=0)[c].transform(_cs_zscore)

    return out.drop(columns=feature_cols)
