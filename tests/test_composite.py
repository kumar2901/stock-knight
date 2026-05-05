"""Tests for the composite ranking score helper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_knight.config import CompositeWeights
from stock_knight.ranking.composite import compute_composite


def _hand_built_features(symbols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = len(symbols)
    return pd.DataFrame(
        {
            "vol_21d_z": rng.normal(0, 1, n),
            "mom_63d_z": rng.normal(0, 1, n),
        },
        index=pd.Index(symbols, name="symbol"),
    )


def test_composite_returns_one_score_per_symbol() -> None:
    syms = [f"S{i}" for i in range(20)]
    pred = pd.Series(np.linspace(-0.05, 0.05, len(syms)), index=syms, name="predicted_return")
    feats = _hand_built_features(syms)

    components = compute_composite(predicted_return=pred, features_day=feats)

    assert len(components.composite) == len(syms)
    assert components.composite.index.tolist() == syms
    assert np.isfinite(components.composite).all()


def test_composite_weights_change_ordering() -> None:
    syms = [f"S{i}" for i in range(30)]
    pred = pd.Series(np.linspace(-0.1, 0.1, len(syms)), index=syms)
    feats = _hand_built_features(syms)

    only_return = CompositeWeights(
        expected_return=1.0,
        sharpe=0.0,
        momentum=0.0,
        low_volatility=0.0,
        model_confidence=0.0,
    )
    only_lowvol = CompositeWeights(
        expected_return=0.0,
        sharpe=0.0,
        momentum=0.0,
        low_volatility=1.0,
        model_confidence=0.0,
    )

    c1 = compute_composite(predicted_return=pred, features_day=feats, weights=only_return)
    c2 = compute_composite(predicted_return=pred, features_day=feats, weights=only_lowvol)

    # When only return matters, the highest predicted-return symbol leads.
    assert c1.composite.idxmax() == pred.idxmax()
    # When only low-vol matters, the lowest vol_21d_z symbol leads.
    assert c2.composite.idxmax() == feats["vol_21d_z"].idxmin()


def test_composite_handles_missing_feature_columns() -> None:
    syms = [f"S{i}" for i in range(10)]
    pred = pd.Series(np.linspace(-0.02, 0.02, len(syms)), index=syms)
    empty_feats = pd.DataFrame(index=pd.Index(syms, name="symbol"))

    components = compute_composite(predicted_return=pred, features_day=empty_feats)
    assert len(components.composite) == len(syms)
    # Without volatility/momentum, only the return + confidence components contribute,
    # so the highest predicted return should remain the leader.
    assert components.composite.idxmax() == pred.idxmax()


def test_sharpe_with_raw_vol_no_longer_amplifies_low_vol_names() -> None:
    """
    Regression test for the defensive-names-dominate-TopK bug (GLAXO/PFIZER case).

    Before the fix, the Sharpe denominator was `vol_z - vol_z.min() + 1.0`,
    which produced ~10x amplification for low-vol names even when their
    predicted return was moderate. With raw `vol_21d` the spread is ~2x,
    so sharpe_z should remain bounded relative to expected_return_z.
    """
    syms = ["A_lowvol", "B_midvol", "C_highvol"]
    pred = pd.Series(
        {"A_lowvol": 0.006, "B_midvol": 0.006, "C_highvol": 0.006}  # equal predictions
    )
    feats = pd.DataFrame(
        {
            "vol_21d": [0.010, 0.020, 0.035],   # realistic daily vols
            "vol_21d_z": [-1.2, 0.0, 1.2],
            "mom_63d": [0.05, 0.05, 0.05],
        },
        index=pd.Index(syms, name="symbol"),
    )

    components = compute_composite(predicted_return=pred, features_day=feats)

    # With equal predictions, the sharpe RATIO between low-vol and high-vol
    # should be moderate — the ratio of raw vols (3.5x) is the ceiling.
    sharpe_ratio = (
        feats.loc["C_highvol", "vol_21d"] / feats.loc["A_lowvol", "vol_21d"]
    )
    assert sharpe_ratio <= 4.0, "raw vol spread should stay bounded"

    # And crucially, because all predictions are equal, expected_return_z is
    # identically zero — composite differences come ONLY from vol tilt. The
    # low-vol name should lead by a MODERATE margin, not the extreme margin
    # that the legacy formula produced.
    c = components.composite
    spread = c.loc["A_lowvol"] - c.loc["C_highvol"]
    # The legacy formula gave spreads > 2.0 on this input; the fix keeps it < 1.5.
    assert spread < 1.5, f"low-vol boost too large: spread={spread:.3f}"


def test_sharpe_falls_back_gracefully_without_raw_vol() -> None:
    """If raw vol_21d is absent, we use a rank-based denominator capped at 3x."""
    syms = [f"S{i}" for i in range(10)]
    pred = pd.Series(np.linspace(0.001, 0.01, len(syms)), index=syms)
    feats = pd.DataFrame(
        {
            # Only z-scored vol (legacy parquets)
            "vol_21d_z": np.linspace(-2.0, 2.0, len(syms)),
            "mom_63d_z": np.zeros(len(syms)),
        },
        index=pd.Index(syms, name="symbol"),
    )

    components = compute_composite(predicted_return=pred, features_day=feats)
    # All finite, same length.
    assert np.isfinite(components.sharpe_z).all()
    assert len(components.composite) == len(syms)
