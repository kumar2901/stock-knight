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


def make_sector_residual_returns(
    forward_returns: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    Compute sector-and-date-neutral forward returns.

    For each ``r_fwd_h`` column we subtract the within-(date, sector) median.
    This removes the sector factor: a model trained on the resulting
    ``r_fwd_h_sector_residual`` series cannot get credit for picking pharma
    names just because the pharma sector outperformed the broad index.

    Parameters
    ----------
    forward_returns
        DataFrame indexed by ``(date, symbol)`` with columns ``r_fwd_*``.
    sector_map
        Series mapping ``symbol -> sector`` (e.g. from
        :func:`stock_knight.data.sectors.load_sector_map`). Symbols missing
        from the map are pooled into a single "Others" bucket so they still
        get a residual computed (against each other).
    """
    if not isinstance(forward_returns.index, pd.MultiIndex):
        raise ValueError("forward_returns must have a (date, symbol) MultiIndex")

    cols = [c for c in forward_returns.columns if c.startswith("r_fwd_")]
    if not cols:
        return forward_returns.copy()

    df = forward_returns.copy()
    syms = df.index.get_level_values(1)
    sectors = sector_map.reindex(syms).fillna("Others").to_numpy()
    df = df.assign(_sector=sectors)

    dates = df.index.get_level_values(0)

    out_cols: dict[str, pd.Series] = {}
    for c in cols:
        # Median per (date, sector) — broadcast back to every row.
        med = df.groupby([dates, df["_sector"]], sort=False)[c].transform("median")
        out_cols[c + "_sector_residual"] = (df[c] - med).astype(float)

    return pd.DataFrame(out_cols, index=forward_returns.index)

