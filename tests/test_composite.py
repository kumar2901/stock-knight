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
