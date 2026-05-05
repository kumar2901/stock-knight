"""
Composite ranking score combining multiple cross-sectional factors.

All inputs are z-scored cross-sectionally on the same as-of date so the weights
in `CompositeWeights` are unitless. Output is a `RankComponents` per-symbol
table with `composite` plus the breakdown columns used by the dashboard "why"
explanation.

Components
----------
- `expected_return`  : the model's predicted forward return.
- `sharpe`           : `predicted_return / (raw_vol_21d + floor)` (risk-adjusted
                       return; uses RAW daily realized vol, not a z-score).
- `momentum`         : 63-day momentum (`mom_63d`; raw preferred, `_z` fallback).
- `low_volatility`   : negative of `vol_21d_z` (rewards stable names).
- `model_confidence` : `|predicted_return| / cross_sectional_std(predicted_return)`
                       (proxy: how far from the mean a name's prediction is).

Notes
-----
- Sharpe uses RAW `vol_21d` when available. The previous implementation divided
  predicted return by `vol_21d_z - vol_21d_z.min() + 1` which systematically
  amplified low-vol names by ~10x (z-score range ~5, raw vol range ~2x). That
  pulled defensive / pharma names into the Top-10 regardless of prediction
  magnitude. The raw-vol denominator still rewards low-vol names (that's what
  Sharpe *is*) but with a much more moderate ~2x spread.
- If raw `vol_21d` is missing (older features parquet), we fall back to a
  percentile-rank denominator in [0.5, 1.5] — moderate, not 10x.
- If a needed column is missing entirely, the component is treated as zero —
  composite degrades gracefully.
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


def _sharpe_denominator(
    features_day: pd.DataFrame, index: pd.Index
) -> tuple[pd.Series, str]:
    """
    Build the denominator used by the Sharpe component.

    Returns (denominator_series, mode_used) where mode_used is one of:
      - "raw_vol"       : raw `vol_21d` (daily realized vol, preferred).
      - "rank_pct"      : percentile rank of `vol_21d_z` mapped to [0.5, 1.5].
      - "zero_fallback" : no vol info available — denom is 1 everywhere.
    """
    if "vol_21d" in features_day.columns:
        raw = features_day["vol_21d"].reindex(index).astype(float)
        if raw.notna().any():
            # Floor the denominator at 25% of the cross-sectional median to
            # prevent a tiny-vol outlier from exploding Sharpe.
            floor = float(np.nanmedian(raw.to_numpy())) * 0.25
            if not np.isfinite(floor) or floor <= 0:
                floor = 0.005  # ~0.5% daily vol floor
            denom = raw.fillna(raw.median()).clip(lower=floor)
            return denom, "raw_vol"

    if "vol_21d_z" in features_day.columns:
        vol_z = features_day["vol_21d_z"].reindex(index).astype(float).fillna(0.0)
        # Convert z-score to a percentile rank in [0, 1], then map to [0.5, 1.5].
        # This keeps the denominator bounded so the low-vol boost is at most 3x,
        # not 10x like the legacy formula.
        pct = vol_z.rank(pct=True).fillna(0.5)
        denom = 0.5 + pct
        return denom, "rank_pct"

    return pd.Series(1.0, index=index), "zero_fallback"


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
        Features for the same as-of date, indexed by symbol. Reads:
          - `vol_21d` (raw, preferred) for Sharpe denominator; falls back to
             `vol_21d_z` rank if raw is absent.
          - `vol_21d_z` for the low-volatility tilt component.
          - `mom_63d` (raw) or `mom_63d_z` for the momentum component.
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

    if "mom_63d" in features_day.columns:
        mom_raw = features_day["mom_63d"].reindex(pred.index).astype(float).fillna(0.0)
        mom_z = _zscore(mom_raw)
    elif "mom_63d_z" in features_day.columns:
        mom_z = features_day["mom_63d_z"].reindex(pred.index).fillna(0.0).astype(float)
    else:
        mom_z = pd.Series(0.0, index=pred.index)

    # Sharpe: predicted return / realized vol (raw). Using raw vol gives a
    # moderate ~2x spread between low-vol and high-vol names instead of the
    # legacy ~10x amplification, which was the primary driver of defensive
    # names (pharma/telecom) dominating the Top-K list regardless of the
    # underlying prediction quality.
    denom, _mode = _sharpe_denominator(features_day, pred.index)
    sharpe = pred / denom
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
