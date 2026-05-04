"""
Composite ranking score combining multiple cross-sectional factors.

All inputs are z-scored cross-sectionally on the same as-of date so the weights
in `CompositeWeights` are unitless. Output is a `RankComponents` per-symbol
table with `composite` plus the breakdown columns used by the dashboard "why"
explanation.

Components
----------
- `expected_return`  : the model's predicted forward return.
- `sharpe`           : `predicted_return / vol_21d` (risk-adjusted return).
- `momentum`         : raw 63-day momentum (`mom_63d`).
- `low_volatility`   : negative of `vol_21d` (rewards stable names).
- `model_confidence` : `|predicted_return| / cross_sectional_std(predicted_return)`
                       (proxy: how far from the mean a name's prediction is).

Notes
-----
- Volatility (`vol_21d`) and momentum (`mom_63d`) are read from the *raw*
  (pre-z-score) columns in the features parquet. The features pipeline writes
  only the `_z` versions on disk, so we accept the z-scored series and treat
  them as the inputs (rank order is identical anyway).
- If a needed column is missing, the component is treated as zero — the
  composite degrades gracefully to whatever components are available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_knight.config import CompositeWeights


@dataclass(frozen=True)
class RankComponents:
    """Per-symbol breakdown of the composite ranking score on one as-of date."""

    composite: pd.Series
    expected_return_z: pd.Series
    sharpe_z: pd.Series
    momentum_z: pd.Series
    low_volatility_z: pd.Series
    model_confidence_z: pd.Series


def _zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def compute_composite(
    *,
    predicted_return: pd.Series,
    features_day: pd.DataFrame,
    weights: CompositeWeights | None = None,
) -> RankComponents:
    """
    Compute the composite ranking score for a single as-of date.

    Parameters
    ----------
    predicted_return
        Series indexed by symbol containing the model's predicted forward
        return on the as-of date. NaNs are filled with 0.
    features_day
        Features for the same as-of date, indexed by symbol. Used to extract
        `vol_21d_z` (for Sharpe and low-vol) and `mom_63d_z` (for momentum).
        Z-scored versions are accepted because the feature pipeline only
        persists the `_z` columns; rank order matches the raw values.
    weights
        Optional override; defaults to `CompositeWeights()`.

    Returns
    -------
    RankComponents
        Composite series + each z-scored component, all aligned to
        `predicted_return.index`.
    """
    w = weights or CompositeWeights()

    pred = predicted_return.astype(float).fillna(0.0)
    pred_z = _zscore(pred)

    vol_z = (
        features_day["vol_21d_z"].reindex(pred.index)
        if "vol_21d_z" in features_day.columns
        else pd.Series(0.0, index=pred.index)
    )
    vol_z = vol_z.fillna(0.0).astype(float)

    mom_z = (
        features_day["mom_63d_z"].reindex(pred.index)
        if "mom_63d_z" in features_day.columns
        else pd.Series(0.0, index=pred.index)
    )
    mom_z = mom_z.fillna(0.0).astype(float)

    # Sharpe: predicted return / realized vol. Use z-scored vol; add a tiny
    # offset so we never divide by zero in extremes.
    safe_vol = vol_z - vol_z.min() + 1.0
    sharpe = pred / safe_vol
    sharpe_z = _zscore(sharpe)

    low_vol_component = -vol_z

    sd = pred.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        confidence = pred * 0.0
    else:
        confidence = pred.abs() / sd
    confidence_z = _zscore(confidence)

    composite = (
        w.expected_return * pred_z
        + w.sharpe * sharpe_z
        + w.momentum * mom_z
        + w.low_volatility * low_vol_component
        + w.model_confidence * confidence_z
    )

    return RankComponents(
        composite=composite.rename("composite"),
        expected_return_z=pred_z.rename("expected_return_z"),
        sharpe_z=sharpe_z.rename("sharpe_z"),
        momentum_z=mom_z.rename("momentum_z"),
        low_volatility_z=low_vol_component.rename("low_volatility_z"),
        model_confidence_z=confidence_z.rename("model_confidence_z"),
    )
