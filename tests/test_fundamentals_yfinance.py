"""Unit tests for yfinance fundamentals merge (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_knight.features.fundamentals_yfinance import (
    merge_fundamentals_into_features,
    _balance_sheet_metrics,
    _roe_from_statements,
)


def test_balance_sheet_working_capital_current_ratio() -> None:
    """When CA is missing but Working Capital + CL exist, derive CA/CL = 1 + WC/CL."""
    # yfinance shape: index = line items, columns = fiscal periods
    bs = pd.DataFrame(
        {"2024-03-31": [100.0, 50.0]},
        index=["Working Capital", "Current Liabilities"],
    )
    t = type("T", (), {})()
    t.balance_sheet = bs
    m = _balance_sheet_metrics(t)
    assert m["yf_current_ratio_bs"] == pytest.approx(3.0)  # 1 + 100/50


def test_roe_from_statements() -> None:
    fin = pd.DataFrame(
        {"2024-03-31": [80.0]},
        index=["Net Income Common Stockholders"],
    )
    bs = pd.DataFrame(
        {"2024-03-31": [400.0]},
        index=["Stockholders Equity"],
    )
    t = type("T", (), {})()
    t.financials = fin
    t.balance_sheet = bs
    assert _roe_from_statements(t) == pytest.approx(0.2)


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


def test_enrich_day_slice_adds_z_columns(monkeypatch) -> None:
    from stock_knight.features import fundamentals_yfinance as fy

    def fake_fetch(symbols: list[str], delay_s: float = 0.2):
        return pd.DataFrame(
            {
                "yf_trailing_pe": [10.0, 20.0],
                "yf_forward_pe": [np.nan, np.nan],
                "yf_pb": [1.0, 2.0],
                "yf_dividend_yield": [0.01, 0.02],
                "yf_debt_to_equity": [np.nan, np.nan],
                "yf_current_ratio": [np.nan, np.nan],
                "yf_roe": [np.nan, np.nan],
                "yf_profit_margin": [np.nan, np.nan],
                "yf_current_ratio_bs": [np.nan, np.nan],
                "yf_debt_to_equity_bs": [np.nan, np.nan],
            },
            index=pd.Index([s.upper() for s in symbols], name="symbol"),
        )

    monkeypatch.setattr(fy, "fetch_fundamentals_yfinance", fake_fetch)
    day = pd.DataFrame({"ret_z": [0.1, -0.1]}, index=pd.Index(["AAA", "BBB"], name="symbol"))
    out = fy.enrich_day_slice_with_yfinance(
        day, pd.Timestamp("2024-06-01"), sector_map=None, delay_s=0.0
    )
    assert "yf_trailing_pe_z" in out.columns
    assert len(out) == 2


def test_merge_tolerates_duplicate_sector_index() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
    syms = ["A", "B"]
    idx = pd.MultiIndex.from_arrays([dates, syms], names=["date", "symbol"])
    feats = pd.DataFrame({"dummy_z": [0.0, 0.0]}, index=idx)
    fund = pd.DataFrame(
        {
            "yf_trailing_pe": [10.0, 20.0],
            "yf_pb": [1.0, 2.0],
            "yf_dividend_yield": [0.02, 0.02],
            "yf_debt_to_equity": [np.nan, np.nan],
            "yf_current_ratio": [np.nan, np.nan],
            "yf_roe": [np.nan, np.nan],
            "yf_profit_margin": [np.nan, np.nan],
            "yf_forward_pe": [np.nan, np.nan],
            "yf_current_ratio_bs": [np.nan, np.nan],
            "yf_debt_to_equity_bs": [np.nan, np.nan],
        },
        index=pd.Index(["A", "B"], name="symbol"),
    )
    smap = pd.Series(["S1", "S1", "OLD"], index=["A", "A", "B"])  # duplicate A
    out = merge_fundamentals_into_features(feats, fund, sector_map=smap)
    assert "yf_pe_vs_sector_z" in out.columns
