"""Shared fixtures for stock-knight unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_knight.brokers.mock import MockBrokerEODClient
from stock_knight.features.price_features import compute_price_features
from stock_knight.labels import make_forward_returns


@pytest.fixture(scope="session")
def mock_ohlcv() -> pd.DataFrame:
    """A 2-year, 30-symbol synthetic OHLCV frame."""
    client = MockBrokerEODClient(n_symbols=30, seed=11)
    syms = client.list_symbols().symbols
    return client.fetch_ohlcv(
        symbols=syms,
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2023-12-31"),
    )


@pytest.fixture(scope="session")
def features_labels(mock_ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = compute_price_features(mock_ohlcv)
    labels = make_forward_returns(mock_ohlcv[["date", "symbol", "close"]])
    return feats, labels


@pytest.fixture()
def small_training_set(
    features_labels: tuple[pd.DataFrame, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    feats, labels = features_labels
    df = feats.join(labels, how="inner")
    df = df.dropna(subset=["r_fwd_7d"])
    feature_cols = [c for c in feats.columns if c.endswith("_z")]
    usable = [c for c in feature_cols if df[c].notna().mean() >= 0.8]
    df = df[usable + ["r_fwd_7d"]].dropna()
    if len(df) > 4000:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(df), size=4000, replace=False)
        df = df.iloc[idx].sort_index()
    return df, df["r_fwd_7d"]
