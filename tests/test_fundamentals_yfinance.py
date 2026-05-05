"""Unit tests for yfinance fundamentals merge (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_knight.features.fundamentals_yfinance import merge_fundamentals_into_features


def test_merge_adds_z_columns_and_sector_spread() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"])
    syms = ["A", "B", "C"]
    idx = pd.MultiIndex.from_arrays([dates, syms], names=["date", "symbol"])
    feats = pd.DataFrame({"dummy_z": [0.0, 0.0, 0.0]}, index=idx)

    fund = pd.DataFrame(
        {
            "yf_trailing_pe": [10.0, 20.0, 15.0],
            "yf_pb": [1.0, 2.0, 1.5],
            "yf_dividend_yield": [0.02, 0.01, 0.015],
            "yf_debt_to_equity": [np.nan, np.nan, np.nan],
            "yf_current_ratio": [np.nan, np.nan, np.nan],
            "yf_roe": [np.nan, np.nan, np.nan],
            "yf_profit_margin": [np.nan, np.nan, np.nan],
            "yf_forward_pe": [np.nan, np.nan, np.nan],
            "yf_current_ratio_bs": [np.nan, np.nan, np.nan],
            "yf_debt_to_equity_bs": [np.nan, np.nan, np.nan],
        },
        index=pd.Index(["A", "B", "C"], name="symbol"),
    )

    smap = pd.Series({"A": "S1", "B": "S1", "C": "S2"}, name="sector")

    out = merge_fundamentals_into_features(feats, fund, sector_map=smap)

    assert "yf_trailing_pe_z" in out.columns
    assert "yf_pb_z" in out.columns
    assert "yf_pe_vs_sector_z" in out.columns
    assert "yf_pb_vs_sector_z" in out.columns
    assert "dummy_z" in out.columns
    assert "yf_trailing_pe" in out.columns
    assert out.loc[idx[0], "yf_trailing_pe"] == pytest.approx(10.0)
