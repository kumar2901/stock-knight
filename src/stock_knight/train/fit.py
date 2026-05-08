from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stock_knight.storage import DataPaths, read_df_parquet, write_json
from stock_knight.train.artifact_names import model_artifact_path
from stock_knight.train.models import (
    TrainedModel,
    save_model,
    train_ensemble,
    train_lgbm,
    train_lgbm_rank,
    train_rf,
    train_ridge,
    train_xgb,
    winsorize_per_date,
)
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


def _fit_model(kind: str, X: pd.DataFrame, y: pd.Series, horizon: str) -> TrainedModel:
    if kind == "ridge":
        return train_ridge(X, y, horizon=horizon)
    if kind == "lgbm":
        return train_lgbm(X, y, horizon=horizon)
    if kind == "lgbm_rank":
        return train_lgbm_rank(X, y, horizon=horizon)
    if kind == "xgb":
        return train_xgb(X, y, horizon=horizon)
    if kind == "rf":
        return train_rf(X, y, horizon=horizon)
    if kind == "ensemble":
        return train_ensemble(X, y, horizon=horizon)
    raise ValueError(f"unknown model_kind={kind}")


def train_walk_forward(
    data_root: Path,
    features_parquet: Path,
    labels_parquet: Path,
    horizons: Iterable[str] = ("1d", "7d", "1m", "6m", "12m"),
    model_kind: str = "ridge",
    target: str = "raw",
    out_dir: Path | None = None,
    winsorize: bool = True,
) -> list[TrainResult]:
    """
    Train one model per horizon using a simple walk-forward procedure.

    Logs per-fold metrics (Spearman IC + Top-K spreads) and trains a final
    model on the full dataset at the end. When ``winsorize`` is True (default)
    forward returns are clipped at the 1st / 99th percentile per date before
    fitting, which dramatically reduces the influence of single-day shocks
    (COVID lockdown, earnings surprises) on the learned coefficients.
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
    if target == "excess":
        horizon_to_col = {k: v + "_excess" for k, v in horizon_to_col.items()}
    elif target == "sector_residual":
        horizon_to_col = {k: v + "_sector_residual" for k, v in horizon_to_col.items()}

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

        if winsorize:
            hdf = hdf.copy()
            hdf[ycol] = winsorize_per_date(hdf[ycol], lower=0.01, upper=0.99)

        all_dates = pd.DatetimeIndex(hdf.index.get_level_values(0).unique())
        fold_metrics: list[dict] = []

        for split in iter_walk_forward_splits(
            all_dates, train_years=2, valid_months=2, step_months=2
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

            tm = _fit_model(model_kind, train_df, train_df[ycol], horizon=h)

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

        final_tm = _fit_model(model_kind, hdf, hdf[ycol], horizon=h)

        model_path = model_artifact_path(out_dir, model_kind, h, target)
        save_model(model_path, final_tm)

        metrics_path = out_dir / f"{model_path.stem}_metrics.json"
        write_json(metrics_path, {"horizon": h, "model_kind": model_kind, "folds": fold_metrics})

        results.append(TrainResult(model_path=model_path, metrics_path=metrics_path))

    return results

