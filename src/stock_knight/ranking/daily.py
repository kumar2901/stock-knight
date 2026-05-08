from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_knight.config import CompositeWeights
from stock_knight.logger import get_logger
from stock_knight.ranking.composite import compute_composite, _zscore as _mom_cs_zscore
from stock_knight.ranking.skill_gate import (
    MIN_HIT_RATE,
    MIN_PICKS_FOR_HIT_RATE,
    calibrated_confidence,
    load_skill_summary,
    per_symbol_hit_rate,
    regime_flag,
)
from stock_knight.data.sectors import load_sector_map
from stock_knight.features.fundamentals_yfinance import enrich_day_slice_with_yfinance
from stock_knight.storage import read_df_parquet, write_json
from stock_knight.train.models import TrainedModel, load_model

_log = get_logger(__name__)


@dataclass(frozen=True)
class RankLists:
    asof: pd.Timestamp
    horizon: str
    training_target: str  # raw | excess (metadata)
    benchmark_symbol: str | None  # metadata (when training_target=excess)
    ranked: list[str]  # full ranking, best -> worst
    rows: list[dict]  # per-symbol enriched rows (symbol, score, current_price, target_price, predicted_return, composite, recommendation, confidence_pct, reason)
    top10: list[str]
    top20: list[str]
    top30: list[str]
    bottom10: list[str]
    bottom20: list[str]
    bottom30: list[str]
    live_ltp_fetched_at: str | None = None  # UTC ISO when live quotes were fetched
    live_ltp_broker: str | None = None  # e.g. yahoo | zerodha
    rank_method: str = "composite"  # composite | score_only
    composite_weights: dict | None = None  # weights used (composite mode only)
    regime_flag: str = "unknown"  # ok | low_skill | unknown (from skill gate)
    recent_ic: float | None = None  # mean Spearman IC of last few walk-forward folds
    recent_top10_spread: float | None = None  # mean Top-10 spread of same folds
    skill_source: str | None = None  # path to *_metrics.json consulted
    prediction_is_fractional_return: bool = True  # False for lgbm_rank / ensemble


def rank_for_date(
    features_parquet: Path,
    model_path: Path,
    ohlcv_parquet: Path | None,
    asof: pd.Timestamp,
    horizon: str,
    training_target: str = "raw",
    benchmark_symbol: str | None = None,
    min_non_na: bool = True,
    fetch_ltp: Callable[[list[str]], dict[str, float]] | None = None,
    live_ltp_broker: str | None = None,
    universe: list[str] | None = None,
    rank_method: str = "composite",
    composite_weights: CompositeWeights | None = None,
    reports_dir: Path | None = None,
    labels_parquet: Path | None = None,
    enable_momentum_gate: bool = True,
    enable_regime_gate: bool = True,
    enable_hit_rate_gate: bool = True,
    yfinance_fundamentals: bool = False,
    yfinance_delay_s: float = 0.2,
    sectors_csv: Path | None = None,
) -> RankLists:
    feats = read_df_parquet(features_parquet)
    asof = pd.to_datetime(asof).tz_localize(None)
    if universe is not None and len(universe) > 0:
        symbols_set = set(universe)
        feats = feats[feats.index.get_level_values(1).isin(symbols_set)]
        if len(feats) == 0:
            raise ValueError(
                "Features parquet has no overlap with the requested universe "
                f"({len(symbols_set)} symbols). Re-run ingest with the same universe."
            )

    # Find nearest available date on or before asof
    available_dates = pd.DatetimeIndex(
        feats.index.get_level_values(0).unique()
    ).normalize().sort_values()
    candidates = available_dates[available_dates <= asof]
    if len(candidates) == 0:
        raise ValueError(
            f"No feature data on or before {asof.date()}. "
            f"Earliest available date is {available_dates.min().date()}."
        )
    effective_asof = candidates[-1]
    if effective_asof != asof:
        warnings.warn(
            f"No features for {asof.date()} — using nearest available date {effective_asof.date()}.",
            stacklevel=2,
        )
    asof = effective_asof
    day = feats.xs(asof, level=0)
    # Defensive: drop duplicate symbol rows. They can appear when the features
    # parquet is built from overlapping ingests, or when a benchmark row leaks
    # into the symbol level. Downstream code (reindex, .loc lookups, row JSON)
    # assumes unique symbols.
    if not day.index.is_unique:
        n_dup = int(day.index.duplicated().sum())
        warnings.warn(
            f"{n_dup} duplicate symbol rows on {asof.date()} — keeping first occurrence.",
            UserWarning,
            stacklevel=2,
        )
        day = day[~day.index.duplicated(keep="first")]

    tm: TrainedModel = load_model(model_path)
    pred_is_fr = tm.predictions_are_fractional_returns()
    if min_non_na:
        day_full = day.dropna(subset=tm.feature_cols)
        if len(day_full) == 0:
            # Some long-lookback features (mom_252d, ma_dist_200) need 200-252 days.
            # If the parquet covers < 252 days, those are NaN. Fall back to features
            # that are well-populated on this specific date (≤ 20% NaN stocks).
            nan_rate_on_day = day[tm.feature_cols].isna().mean()
            good_feat_cols = nan_rate_on_day[nan_rate_on_day <= 0.20].index.tolist()
            missing = [c for c in tm.feature_cols if c not in good_feat_cols]

            if not good_feat_cols:
                raise ValueError(
                    f"No feature rows available for {asof.date()}. "
                    f"All {len(tm.feature_cols)} model features are NaN. "
                    f"Your features parquet likely covers too short a period (< 252 trading days). "
                    f"Use the full historical features parquet:\n"
                    f"  data/features/features_ohlcv_2019-01-01_*.parquet\n"
                    f"Or run: python -m stock_knight.cli features-labels "
                    f"--ohlcv-parquet data/ohlcv/ohlcv_2019-01-01_<TODAY>.parquet"
                )

            warnings.warn(
                f"Short-history parquet: {len(missing)} long-lookback features unavailable "
                f"({missing[:4]}{'...' if len(missing) > 4 else ''}). "
                f"Ranking with {len(good_feat_cols)}/{len(tm.feature_cols)} features. "
                f"For full accuracy use the 2019+ historical parquet.",
                UserWarning,
                stacklevel=2,
            )
            _log.warning(
                "Ranking with %d/%d features (missing: %s)",
                len(good_feat_cols), len(tm.feature_cols), missing[:6],
            )
            # Keep stocks that have all the good features, fill missing ones with 0
            day = day.dropna(subset=good_feat_cols)
            for c in missing:
                day[c] = 0.0
        else:
            day = day_full
    if len(day) == 0:
        raise ValueError(f"No feature rows available for {asof.date()}")

    if yfinance_fundamentals:
        smap = load_sector_map(sectors_csv)
        day = day.drop(columns=[c for c in day.columns if str(c).startswith("yf_")], errors="ignore")
        day = enrich_day_slice_with_yfinance(
            day,
            asof,
            sector_map=smap if not smap.empty else None,
            delay_s=yfinance_delay_s,
        )

    pred = tm.predict(day)
    pred_series = pd.Series(pred, index=day.index, name="predicted_return")

    if rank_method not in ("composite", "score_only"):
        raise ValueError(
            f"unknown rank_method={rank_method!r}; expected 'composite' or 'score_only'"
        )

    weights_used: CompositeWeights | None = None
    if rank_method == "composite":
        weights_used = composite_weights or CompositeWeights()
        components = compute_composite(
            predicted_return=pred_series,
            features_day=day,
            weights=weights_used,
            include_sharpe_component=pred_is_fr,
        )
        sort_key = components.composite
    else:
        components = None
        sort_key = pred_series

    scores = sort_key.sort_values(ascending=False)

    # ── Validation skill summary (Phase 1c/1d) ────────────────────────────
    # Drives both calibrated confidence and the regime gate. Failing open
    # (no metrics file) is treated as "unknown" -> neutral 0.5 skill factor.
    skill = load_skill_summary(model_path)
    regime = regime_flag(skill)

    # Recommendation buckets (initial assignment by quintile).
    n = len(scores)
    top_cut = max(1, int(round(n * 0.20)))
    bot_cut = max(1, int(round(n * 0.20)))
    recos: dict[str, str] = {}
    reasons: dict[str, str | None] = {}
    for i, sym in enumerate(scores.index):
        if i < top_cut:
            recos[sym] = "BUY"
        elif i >= n - bot_cut:
            recos[sym] = "SELL"
        else:
            recos[sym] = "HOLD"
        reasons[sym] = None

    # ── Phase 1b: momentum-agreement gate ─────────────────────────────────
    # Downgrade BUY -> HOLD only when BOTH: cross-sectional momentum z < 0
    # (lags peers) AND absolute 63d trend is negative. This matches the plan
    # (momentum_z + mom_63d) and avoids nuking a BUY that is merely
    # middle-of-pack on momentum while still in a positive absolute trend.
    mom_raw = day["mom_63d"] if "mom_63d" in day.columns else None
    mom_zsc = day["mom_63d_z"] if "mom_63d_z" in day.columns else None
    mom_cs_z: pd.Series | None = None
    if mom_raw is not None:
        # Same cross-sectional z as `compute_composite` uses for momentum_z.
        mom_cs_z = _mom_cs_zscore(mom_raw.astype(float))
    elif mom_zsc is not None:
        mom_cs_z = mom_zsc.astype(float)

    if enable_momentum_gate and mom_cs_z is not None:
        for sym in list(recos.keys()):
            if recos[sym] != "BUY":
                continue
            if sym not in mom_cs_z.index:
                continue
            if float(mom_cs_z.loc[sym]) >= 0:
                continue
            negative_abs_trend = False
            if mom_raw is not None and sym in mom_raw.index:
                v = mom_raw.loc[sym]
                negative_abs_trend = pd.notna(v) and float(v) < 0
            elif mom_zsc is not None and sym in mom_zsc.index:
                v = mom_zsc.loc[sym]
                negative_abs_trend = pd.notna(v) and float(v) < 0
            if negative_abs_trend:
                recos[sym] = "HOLD"
                reasons[sym] = "momentum_disagreement"

    # ── Phase 1d: regime gate ─────────────────────────────────────────────
    # In a no-skill regime (recent walk-forward IC at/below threshold),
    # downgrade ALL remaining BUYs to HOLD. The composite ranking is still
    # produced (for research / dashboard), but we refuse to call anything a
    # BUY when validation says we have no edge.
    if enable_regime_gate and skill.is_low_skill:
        for sym in list(recos.keys()):
            if recos[sym] == "BUY":
                recos[sym] = "HOLD"
                reasons[sym] = "low_skill_regime"

    # ── Phase 3c: per-stock hit-rate suppression ─────────────────────────
    # If we have a reports/ history and labels parquet, look up each BUY
    # candidate's historical hit-rate (was-BUY-and-outperformed) and
    # downgrade names with a sustained miss rate.
    hit_rate: pd.Series | None = None
    if (
        enable_hit_rate_gate
        and reports_dir is not None
        and labels_parquet is not None
        and reports_dir.exists()
        and labels_parquet.exists()
    ):
        try:
            hit_rate = per_symbol_hit_rate(
                reports_dir=reports_dir,
                labels_parquet=labels_parquet,
                horizon=horizon,
                lookback_files=100,
            )
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"per_symbol_hit_rate failed: {e}", UserWarning, stacklevel=2)
            hit_rate = None

    if hit_rate is not None and len(hit_rate) > 0:
        for sym in list(recos.keys()):
            if recos[sym] != "BUY":
                continue
            if sym in hit_rate.index and float(hit_rate.loc[sym]) < MIN_HIT_RATE:
                recos[sym] = "HOLD"
                reasons[sym] = f"low_hit_rate(<{int(MIN_HIT_RATE * 100)}%)"

    # ── Phase 1c: calibrated confidence ───────────────────────────────────
    # Replaces the legacy |sigma|/3 cap-at-100 scheme with a
    # percentile * skill_factor mapping (0..95). In a low-skill regime the
    # ceiling collapses (skill_factor ~ 0), so users see "I don't trust this".
    confidence_pct = calibrated_confidence(scores, skill, cap=95.0)

    # Price enrichment from OHLCV (optional but recommended)
    asof_close: pd.Series
    if ohlcv_parquet is not None:
        ohlcv = read_df_parquet(ohlcv_parquet)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None)
        px_hist = ohlcv[ohlcv["date"] <= asof][["date", "symbol", "close"]].dropna()
        # Latest available close on or before asof (defensive)
        px_hist = px_hist.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
        asof_close = px_hist.set_index("symbol")["close"]
    else:
        asof_close = pd.Series(dtype=float)

    live_ltp_fetched_at: str | None = None
    ltp_map: dict[str, float] = {}
    if fetch_ltp is not None:
        ltp_map = fetch_ltp(scores.index.tolist())
        live_ltp_fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    rows: list[dict] = []
    for sym, sort_score in scores.items():
        cp = float(asof_close.loc[sym]) if sym in asof_close.index else None
        # `predicted_return` is the raw model output (fractional return for ridge/lgbm/…,
        # or a rank score for lgbm_rank / ensemble). Target prices use it only when
        # it is in return units; see `prediction_is_fractional_return` in the payload.
        pr = float(pred_series.loc[sym]) if sym in pred_series.index else float(sort_score)
        ltp = float(ltp_map[sym]) if sym in ltp_map else None
        if pred_is_fr:
            tp = (cp * (1.0 + pr)) if cp is not None else None
            ltp_tgt = (ltp * (1.0 + pr)) if ltp is not None else None
        else:
            tp = None
            ltp_tgt = None
        row = {
            "symbol": sym,
            "predicted_return": pr,
            "score": float(sort_score),
            "asof_close": cp,
            "asof_target_price": tp,
            "live_ltp": ltp,
            "live_target_price": ltp_tgt,
            "recommendation": recos[sym],
            "confidence_pct": float(confidence_pct.loc[sym]),
            "reason": reasons.get(sym),
            "hit_rate": (
                float(hit_rate.loc[sym])
                if hit_rate is not None and sym in hit_rate.index
                else None
            ),
        }
        if components is not None:
            row["composite"] = float(components.composite.loc[sym])
            row["expected_return_z"] = float(components.expected_return_z.loc[sym])
            row["sharpe_z"] = float(components.sharpe_z.loc[sym])
            row["momentum_z"] = float(components.momentum_z.loc[sym])
            row["low_volatility_z"] = float(components.low_volatility_z.loc[sym])
            row["model_confidence_z"] = float(components.model_confidence_z.loc[sym])
        for col in day.columns:
            if not str(col).startswith("yf_"):
                continue
            try:
                v = day.loc[sym, col]
            except (KeyError, TypeError):
                continue
            if pd.isna(v):
                row[col] = None
            else:
                row[col] = float(v)
        rows.append(row)

    ranked = scores.index.tolist()
    top10 = scores.head(10).index.tolist()
    top20 = scores.head(20).index.tolist()
    top30 = scores.head(30).index.tolist()
    bottom10 = scores.tail(10).index.tolist()[::-1]
    bottom20 = scores.tail(20).index.tolist()[::-1]
    bottom30 = scores.tail(30).index.tolist()[::-1]

    return RankLists(
        asof=effective_asof,
        horizon=horizon,
        training_target=training_target,
        benchmark_symbol=benchmark_symbol,
        ranked=ranked,
        rows=rows,
        top10=top10,
        top20=top20,
        top30=top30,
        bottom10=bottom10,
        bottom20=bottom20,
        bottom30=bottom30,
        live_ltp_fetched_at=live_ltp_fetched_at,
        live_ltp_broker=live_ltp_broker if fetch_ltp is not None else None,
        rank_method=rank_method,
        composite_weights=(
            weights_used.model_dump() if weights_used is not None else None
        ),
        regime_flag=regime,
        recent_ic=None if skill.is_unknown else float(skill.recent_ic),
        recent_top10_spread=(
            None if skill.is_unknown else float(skill.recent_top10)
        ),
        skill_source=skill.source,
        prediction_is_fractional_return=pred_is_fr,
    )


def write_ranklists(out_path: Path, ranks: RankLists) -> None:
    write_json(
        out_path,
        {
            "asof": str(ranks.asof),
            "horizon": ranks.horizon,
            "training_target": ranks.training_target,
            "benchmark_symbol": ranks.benchmark_symbol,
            "live_ltp_fetched_at": ranks.live_ltp_fetched_at,
            "live_ltp_broker": ranks.live_ltp_broker,
            "rank_method": ranks.rank_method,
            "composite_weights": ranks.composite_weights,
            "regime_flag": ranks.regime_flag,
            "recent_ic": ranks.recent_ic,
            "recent_top10_spread": ranks.recent_top10_spread,
            "skill_source": ranks.skill_source,
            "prediction_is_fractional_return": ranks.prediction_is_fractional_return,
            "ranked": ranks.ranked,
            "rows": ranks.rows,
            "top10": ranks.top10,
            "top20": ranks.top20,
            "top30": ranks.top30,
            "bottom10": ranks.bottom10,
            "bottom20": ranks.bottom20,
            "bottom30": ranks.bottom30,
        },
    )

