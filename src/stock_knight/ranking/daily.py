from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_knight.logger import get_logger
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
    rows: list[dict]  # per-symbol enriched rows (symbol, score, current_price, target_price, predicted_return)
    top10: list[str]
    top20: list[str]
    top30: list[str]
    bottom10: list[str]
    bottom20: list[str]
    bottom30: list[str]
    live_ltp_fetched_at: str | None = None  # UTC ISO when live quotes were fetched
    live_ltp_broker: str | None = None  # e.g. yahoo | zerodha


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
) -> RankLists:
    feats = read_df_parquet(features_parquet)
    asof = pd.to_datetime(asof).tz_localize(None)

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

    tm: TrainedModel = load_model(model_path)
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

    pred = tm.predict(day)
    scores = pd.Series(pred, index=day.index, name="score").sort_values(ascending=False)

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
    for sym, score in scores.items():
        cp = float(asof_close.loc[sym]) if sym in asof_close.index else None
        pr = float(score)
        tp = (cp * (1.0 + pr)) if cp is not None else None
        ltp = float(ltp_map[sym]) if sym in ltp_map else None
        ltp_tgt = (ltp * (1.0 + pr)) if ltp is not None else None
        rows.append(
            {
                "symbol": sym,
                "predicted_return": pr,
                "score": pr,
                "asof_close": cp,
                "asof_target_price": tp,
                "live_ltp": ltp,
                "live_target_price": ltp_tgt,
            }
        )

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

