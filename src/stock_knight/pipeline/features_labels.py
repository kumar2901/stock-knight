from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_knight.features.price_features import compute_price_features
from stock_knight.labels import Horizons, make_forward_returns
from stock_knight.storage import DataPaths, read_df_parquet, write_df_parquet


def build_features_and_labels(
    data_root: Path,
    ohlcv_parquet: Path,
    horizons: Horizons = Horizons(),
) -> tuple[Path, Path]:
    """
    Reads OHLCV parquet, writes:
      - features parquet (indexed by date,symbol)
      - labels parquet (indexed by date,symbol)
    """
    paths = DataPaths(root=data_root)
    ohlcv = read_df_parquet(ohlcv_parquet)

    feats = compute_price_features(ohlcv)
    labels = make_forward_returns(ohlcv[["date", "symbol", "close"]], horizons=horizons)

    feats_out = paths.features / f"features_{ohlcv_parquet.stem}.parquet"
    labels_out = paths.labels / f"labels_{ohlcv_parquet.stem}.parquet"

    write_df_parquet(feats_out, feats)
    write_df_parquet(labels_out, labels)

    return feats_out, labels_out

