from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stock_knight.storage import DataPaths, read_df_parquet, write_json
from stock_knight.train.models import TrainedModel, train_lgbm, train_ridge, save_model
from stock_knight.train.walk_forward import iter_walk_forward_splits


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics_path: Path


def _spearman_ic(pred: np.ndarray, y: np.ndarray) -> float:
    s1 = pd.Series(pred).rank(pct=False).to_numpy()
    s2 = pd.Series(y).rank(pct=False).to_numpy()
    if np.std(s1) == 0 or np.std(s2) == 0:
        return float("nan")
    return float(np.corrcoef(s1, s2)[0, 1])


def _topk_spread(pred: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) < k:
        return float("nan")
    idx = np.argsort(pred)[::-1]
    top = y[idx[:k]].mean()
    base = y.mean()
    return float(top - base)


def train_walk_forward(
    data_root: Path,
    features_parquet: Path,
    labels_parquet: Path,
    horizons: Iterable[str] = ("1d", "7d", "1m", "6m", "12m"),
    model_kind: str = "ridge",
    out_dir: Path | None = None,
) -> list[TrainResult]:
    """
    Train one model per horizon using a simple walk-forward procedure.

    This is an MVP trainer (baseline ridge or lgbm). It logs fold metrics and
    trains a final model on the full dataset at the end.
    """
    out_dir = out_dir or (Path("artifacts") / "models")
    X = read_df_parquet(features_parquet)
    ydf = read_df_parquet(labels_parquet)

    # Join on (date, symbol)
    df = X.join(ydf, how="inner")
    feature_cols = [c for c in X.columns if c.endswith("_z")]

    results: list[TrainResult] = []
    horizon_to_col = {
        "1d": "r_fwd_1d",
        "7d": "r_fwd_7d",
        "1m": "r_fwd_1m",
        "6m": "r_fwd_6m",
        "12m": "r_fwd_12m",
    }

    for h in horizons:
        ycol = horizon_to_col[h]
        # Avoid dropping everything when long-lookback features are unavailable (e.g., <252d history).
        usable = []
        for c in feature_cols:
            frac = float(df[c].notna().mean())
            if frac >= 0.8:
                usable.append(c)
        if len(usable) == 0:
            raise ValueError("No usable features found (all too sparse). Increase history or relax threshold.")

        hdf = df[usable + [ycol]].dropna()
        if len(hdf) == 0:
            raise ValueError(f"No training rows available for horizon={h} after dropping NaNs")

        all_dates = pd.DatetimeIndex(hdf.index.get_level_values(0).unique())
        fold_metrics: list[dict] = []

        for split in iter_walk_forward_splits(
            all_dates, train_years=1, valid_months=2, step_months=2
        ):
            train_mask = (hdf.index.get_level_values(0) >= split.train_start) & (
                hdf.index.get_level_values(0) <= split.train_end
            )
            valid_mask = (hdf.index.get_level_values(0) >= split.valid_start) & (
                hdf.index.get_level_values(0) <= split.valid_end
            )

            train_df = hdf.loc[train_mask]
            valid_df = hdf.loc[valid_mask]
            if len(train_df) == 0 or len(valid_df) == 0:
                continue

            if model_kind == "ridge":
                tm = train_ridge(train_df, train_df[ycol], horizon=h)
            elif model_kind == "lgbm":
                tm = train_lgbm(train_df, train_df[ycol], horizon=h)
            else:
                raise ValueError(f"unknown model_kind={model_kind}")

            pred = tm.predict(valid_df)
            yv = valid_df[ycol].to_numpy()

            fold_metrics.append(
                {
                    "horizon": h,
                    "train_start": str(split.train_start.date()),
                    "train_end": str(split.train_end.date()),
                    "valid_start": str(split.valid_start.date()),
                    "valid_end": str(split.valid_end.date()),
                    "n_train": int(len(train_df)),
                    "n_valid": int(len(valid_df)),
                    "ic_spearman": _spearman_ic(pred, yv),
                    "top10_spread": _topk_spread(pred, yv, k=10),
                    "top20_spread": _topk_spread(pred, yv, k=20),
                    "top30_spread": _topk_spread(pred, yv, k=30),
                }
            )

        # Fit final model on full dataset (MVP)
        if model_kind == "ridge":
            final_tm = train_ridge(hdf, hdf[ycol], horizon=h)
        elif model_kind == "lgbm":
            final_tm = train_lgbm(hdf, hdf[ycol], horizon=h)
        else:
            raise ValueError(f"unknown model_kind={model_kind}")

        model_path = out_dir / f"{model_kind}_{h}.joblib"
        save_model(model_path, final_tm)

        metrics_path = out_dir / f"{model_kind}_{h}_metrics.json"
        write_json(metrics_path, {"horizon": h, "model_kind": model_kind, "folds": fold_metrics})

        results.append(TrainResult(model_path=model_path, metrics_path=metrics_path))

    return results

