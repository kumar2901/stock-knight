"""Tests for sector-residual labels (Phase 3a)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_knight.data.sectors import sector_for_symbols
from stock_knight.labels import make_sector_residual_returns


def _toy_returns() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01"] * 6 + ["2024-01-02"] * 6)
    syms = ["A", "B", "C", "D", "E", "F"] * 2
    idx = pd.MultiIndex.from_arrays([dates, syms], names=["date", "symbol"])
    # Day 1: pharma cohort (A,B,C) up 5%, banks (D,E,F) up 1%.
    # Day 2: pharma down 2%, banks up 3%.
    r = (
        [0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
        + [-0.02, -0.02, -0.02, 0.03, 0.03, 0.03]
    )
    return pd.DataFrame({"r_fwd_7d": r}, index=idx)


def test_sector_residual_zeros_within_homogeneous_sector(tmp_path) -> None:
    sec_csv = tmp_path / "sectors.csv"
    sec_csv.write_text(
        "symbol,sector\nA,Pharma\nB,Pharma\nC,Pharma\nD,Banks\nE,Banks\nF,Banks\n"
    )
    smap = sector_for_symbols(["A", "B", "C", "D", "E", "F"], sectors_csv=sec_csv)

    returns = _toy_returns()
    resid = make_sector_residual_returns(returns, sector_map=smap)
    col = "r_fwd_7d_sector_residual"
    assert col in resid.columns

    # All names are equal within each sector on each day → residuals == 0.
    np.testing.assert_allclose(resid[col].to_numpy(), 0.0, atol=1e-12)


def test_sector_residual_separates_within_sector_winners(tmp_path) -> None:
    sec_csv = tmp_path / "sectors.csv"
    sec_csv.write_text(
        "symbol,sector\nA,Pharma\nB,Pharma\nC,Pharma\nD,Banks\nE,Banks\nF,Banks\n"
    )
    smap = sector_for_symbols(["A", "B", "C", "D", "E", "F"], sectors_csv=sec_csv)

    # Make A the within-sector winner on day 1 (10% vs 5% sector median),
    # D the within-sector winner on day 2 (8% vs 3% sector median).
    dates = pd.to_datetime(["2024-01-01"] * 6 + ["2024-01-02"] * 6)
    syms = ["A", "B", "C", "D", "E", "F"] * 2
    idx = pd.MultiIndex.from_arrays([dates, syms], names=["date", "symbol"])
    r = (
        [0.10, 0.05, 0.05, 0.01, 0.01, 0.01]   # day 1
        + [-0.02, -0.02, -0.02, 0.08, 0.03, 0.03]  # day 2
    )
    returns = pd.DataFrame({"r_fwd_7d": r}, index=idx)

    resid = make_sector_residual_returns(returns, sector_map=smap)
    col = "r_fwd_7d_sector_residual"
    # Within Pharma on day 1, A's residual = 10% - median(10,5,5)% = 5%.
    assert resid.loc[(pd.Timestamp("2024-01-01"), "A"), col] == 0.05
    # Within Banks on day 2, D's residual = 8% - median(8,3,3)% = 5%.
    assert resid.loc[(pd.Timestamp("2024-01-02"), "D"), col] == 0.05
    # And other names get a non-positive residual (median or below).
    assert resid.loc[(pd.Timestamp("2024-01-01"), "B"), col] <= 0.0
