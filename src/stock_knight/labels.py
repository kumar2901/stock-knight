from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Horizons:
    h1d: int = 1
    h7d: int = 5
    h1m: int = 21
    h6m: int = 126
    h12m: int = 252


def make_forward_returns(
    prices: pd.DataFrame,
    horizons: Horizons = Horizons(),
) -> pd.DataFrame:
    """
    Input: long-format dataframe with columns: date, symbol, close
    Output: DataFrame indexed by (date, symbol) with forward returns columns.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"])
    g = df.groupby("symbol", group_keys=False)

    out = df[["date", "symbol"]].copy()

    def fwd(n: int) -> pd.Series:
        future = g["close"].shift(-n)
        return (future / df["close"]) - 1.0

    out["r_fwd_1d"] = fwd(horizons.h1d)
    out["r_fwd_7d"] = fwd(horizons.h7d)
    out["r_fwd_1m"] = fwd(horizons.h1m)
    out["r_fwd_6m"] = fwd(horizons.h6m)
    out["r_fwd_12m"] = fwd(horizons.h12m)

    return out.set_index(["date", "symbol"]).sort_index()


def make_excess_returns(
    forward_returns: pd.DataFrame,
    benchmark_forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Subtract benchmark forward returns (by date) from each stock forward return.

    benchmark_forward_returns: indexed by date with columns matching forward_returns columns.
    """
    if not isinstance(benchmark_forward_returns.index, pd.DatetimeIndex):
        raise ValueError("benchmark_forward_returns must be indexed by date")

    out = forward_returns.copy()
    cols = [c for c in out.columns if c.startswith("r_fwd_")]
    # align on date (level 0)
    for c in cols:
        b = benchmark_forward_returns[c].reindex(out.index.get_level_values(0))
        out[c + "_excess"] = out[c] - b.to_numpy()
    return out[[c + "_excess" for c in cols]]

