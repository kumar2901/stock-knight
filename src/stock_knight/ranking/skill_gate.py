"""
Validation-skill gating for the ranking step.

The ranking layer calls `load_skill_summary(model_path)` to get two things:

1. ``recent_ic``   — mean Spearman IC of the last few walk-forward folds
2. ``recent_top10`` — mean Top-10 spread of the same folds

These are used by :func:`calibrated_confidence` (to scale a composite
percentile into a 0–95% confidence) and by :func:`regime_flag` (to decide
whether we suppress BUYs entirely because the model is in a no-skill regime).

The model metrics file is written by ``train_walk_forward`` next to each
``*.joblib`` as ``{stem}_metrics.json``. If it's missing (e.g. a custom
artifact), we fail open — treat the model as "unknown skill" and return a
neutral skill factor so that legacy callers still work.

Additionally, :func:`per_symbol_hit_rate` scans the most recent ranklist JSONs
under ``reports/`` and joins with the labels parquet to compute, per symbol,
how often that stock was in a BUY bucket AND actually outperformed the
cross-sectional mean over the horizon. This is consumed by the per-stock
suppression rule in ``daily.py`` (Phase 3c).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# Threshold below which we consider the model to have no out-of-sample edge.
# Chosen from the observed distribution of walk-forward ICs in
# artifacts/models/lgbm_*_metrics.json — median IC ~ 0 for 7D/1M, so anything
# <= 0.005 is noise.
LOW_SKILL_IC_THRESHOLD: float = 0.005

# How many recent folds to average when computing recent_ic / recent_top10.
RECENT_FOLDS: int = 4

# Target IC used to rescale the calibrated confidence. If recent_ic is at
# least this value, the skill factor saturates at 1.0.
TARGET_IC_FOR_FULL_CONFIDENCE: float = 0.03

# Hit-rate thresholds for per-stock suppression (Phase 3c).
MIN_PICKS_FOR_HIT_RATE: int = 20
MIN_HIT_RATE: float = 0.45


@dataclass(frozen=True)
class SkillSummary:
    """Validation-set summary of a horizon-specific model.

    Attributes
    ----------
    recent_ic
        Mean Spearman IC over the last ``n_folds`` walk-forward folds.
    recent_top10
        Mean Top-10 spread over the same folds (forward returns of the
        predicted-top-10 minus universe mean).
    n_folds
        How many folds were averaged (may be < RECENT_FOLDS for a freshly
        trained model with few folds).
    source
        Which metrics file was read (for debugging / UI display).
    is_unknown
        True when we couldn't find a metrics file — callers should treat this
        as neutral/unknown skill, not zero skill.
    """

    recent_ic: float
    recent_top10: float
    n_folds: int
    source: str | None
    is_unknown: bool

    @property
    def is_low_skill(self) -> bool:
        """True if recent validation IC is at/below the no-skill threshold."""
        if self.is_unknown:
            return False
        return self.recent_ic <= LOW_SKILL_IC_THRESHOLD

    def skill_factor(self) -> float:
        """Return a multiplier in [0, 1] proportional to how skilled the model is.

        - Unknown metrics → 0.5 (neutral).
        - recent_ic at/below LOW_SKILL_IC_THRESHOLD → 0.0 (no edge).
        - recent_ic >= TARGET_IC_FOR_FULL_CONFIDENCE → 1.0.
        - Negative recent_top10 halves the factor (model's own picks
          underperform on average → we should be less confident in them).
        """
        if self.is_unknown:
            return 0.5
        if self.is_low_skill:
            return 0.0
        ic_factor = max(0.0, self.recent_ic) / TARGET_IC_FOR_FULL_CONFIDENCE
        ic_factor = min(1.0, ic_factor)
        top_factor = 1.0 if self.recent_top10 > 0 else 0.5
        return float(ic_factor * top_factor)


def _metrics_path_for(model_path: Path) -> Path:
    """Given `foo/bar/lgbm_7d.joblib`, return `foo/bar/lgbm_7d_metrics.json`."""
    return model_path.parent / f"{model_path.stem}_metrics.json"


def load_skill_summary(model_path: Path) -> SkillSummary:
    """Read ``{model}_metrics.json`` and summarise the last few walk-forward folds.

    Returns a :class:`SkillSummary`. If the metrics file is missing or
    malformed, ``is_unknown`` is True and all numeric fields are zero — the
    caller should treat this as "cannot assess skill" (we fail open).
    """
    mpath = _metrics_path_for(model_path)
    if not mpath.exists():
        warnings.warn(
            f"No walk-forward metrics at {mpath} — skill gate will fail open "
            "(neutral confidence, no regime suppression).",
            UserWarning,
            stacklevel=2,
        )
        return SkillSummary(
            recent_ic=0.0,
            recent_top10=0.0,
            n_folds=0,
            source=None,
            is_unknown=True,
        )

    try:
        payload = json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(
            f"Failed to read {mpath}: {e}. Skill gate will fail open.",
            UserWarning,
            stacklevel=2,
        )
        return SkillSummary(
            recent_ic=0.0, recent_top10=0.0, n_folds=0, source=str(mpath), is_unknown=True
        )

    folds = payload.get("folds") or []
    if not folds:
        return SkillSummary(
            recent_ic=0.0, recent_top10=0.0, n_folds=0, source=str(mpath), is_unknown=True
        )

    recent = folds[-RECENT_FOLDS:]
    ics = [f.get("ic_spearman") for f in recent if f.get("ic_spearman") is not None]
    t10 = [f.get("top10_spread") for f in recent if f.get("top10_spread") is not None]
    if not ics:
        return SkillSummary(
            recent_ic=0.0, recent_top10=0.0, n_folds=0, source=str(mpath), is_unknown=True
        )

    return SkillSummary(
        recent_ic=float(np.mean(ics)),
        recent_top10=float(np.mean(t10)) if t10 else 0.0,
        n_folds=len(recent),
        source=str(mpath),
        is_unknown=False,
    )


def calibrated_confidence(
    scores: pd.Series,
    skill: SkillSummary,
    cap: float = 95.0,
) -> pd.Series:
    """Turn a per-symbol composite score into a calibrated confidence %.

    The baseline is each stock's cross-sectional percentile on the composite
    (so the top-ranked name gets 100, the bottom-ranked name gets 0). We then
    multiply by the model's skill factor. In a "no-skill" regime the top pick
    cannot exceed ``cap * skill_factor * 0.5`` ~ 24%, which is the explicit
    UX signal we want.

    Parameters
    ----------
    scores
        Per-symbol composite score (higher = more bullish). NaN is allowed.
    skill
        From :func:`load_skill_summary`.
    cap
        Upper bound. Kept at 95 by default so the UI never shows 100%.
    """
    s = scores.astype(float)
    if len(s) == 0:
        return s

    # Percentile rank in [0, 1], centre-at-0.5 for tied/degenerate inputs.
    pct = s.rank(pct=True, method="average").fillna(0.5)

    factor = skill.skill_factor()
    confidence = pct * factor * 100.0
    return confidence.clip(lower=0.0, upper=cap).round(1)


def regime_flag(skill: SkillSummary) -> Literal["ok", "low_skill", "unknown"]:
    """Classify the horizon's current model regime for UI display."""
    if skill.is_unknown:
        return "unknown"
    if skill.is_low_skill:
        return "low_skill"
    return "ok"


# ── Per-stock hit-rate (Phase 3c) ────────────────────────────────────────────


def per_symbol_hit_rate(
    reports_dir: Path,
    labels_parquet: Path,
    horizon: str,
    lookback_files: int = 100,
) -> pd.Series:
    """Historical hit-rate per symbol, computed from recent ranklist JSONs.

    For each ranklist JSON under ``reports_dir`` matching ``ranks_{horizon}_*.json``,
    we load the rows, join with the labels parquet by (date, symbol) to look
    up the realised forward return for that horizon, and compute a "hit" if
    the row's recommendation was BUY and the realised return exceeded the
    universe-mean realised return that day.

    Returns a ``pd.Series`` indexed by symbol with values in [0, 1]. Symbols
    with fewer than :data:`MIN_PICKS_FOR_HIT_RATE` BUY appearances are
    omitted so callers can use ``Series.get(symbol, default=None)``.
    """
    pattern = f"ranks_{horizon}_*.json"
    files = sorted(reports_dir.glob(pattern))
    if not files:
        return pd.Series(dtype=float)

    files = files[-lookback_files:]

    try:
        labels = pd.read_parquet(labels_parquet)
    except Exception as e:
        warnings.warn(
            f"Could not read labels parquet at {labels_parquet}: {e}. "
            "Skipping per-stock hit-rate.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series(dtype=float)

    col = f"r_fwd_{horizon}"
    if col not in labels.columns:
        return pd.Series(dtype=float)

    records: list[tuple[str, float, float]] = []  # (symbol, realised, universe_mean)
    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        asof = pd.to_datetime(payload.get("asof")).tz_localize(None) if payload.get("asof") else None
        if asof is None:
            continue

        rows = payload.get("rows") or []
        buys = [r.get("symbol") for r in rows if r.get("recommendation") == "BUY"]
        if not buys:
            continue

        # Universe mean realised return for this date (use every symbol we have labels for)
        try:
            day_labels = labels.xs(asof.normalize(), level=0)
        except KeyError:
            continue

        if col not in day_labels.columns:
            continue
        universe_mean = float(day_labels[col].mean())
        if not np.isfinite(universe_mean):
            continue

        for sym in buys:
            if sym not in day_labels.index:
                continue
            realised = float(day_labels.loc[sym, col])
            if not np.isfinite(realised):
                continue
            records.append((sym, realised, universe_mean))

    if not records:
        return pd.Series(dtype=float)

    hit_df = pd.DataFrame(records, columns=["symbol", "realised", "universe_mean"])
    hit_df["hit"] = (hit_df["realised"] > hit_df["universe_mean"]).astype(float)

    agg = hit_df.groupby("symbol")["hit"].agg(["mean", "count"])
    kept = agg[agg["count"] >= MIN_PICKS_FOR_HIT_RATE]
    return kept["mean"].astype(float)
