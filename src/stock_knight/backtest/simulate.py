from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_knight.backtest.costs import CostModel, estimate_turnover
from stock_knight.backtest.portfolio import PortfolioSpec, make_long_short_weights, topk_from_scores
from stock_knight.backtest.metrics import summarize_returns
from stock_knight.storage import read_df_parquet
from stock_knight.train.models import TrainedModel, load_model


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series  # indexed by date
    daily_returns: pd.Series
    turnover: pd.Series
    summary: dict


def _daily_pnl_from_weights(
    rets: pd.DataFrame,  # index: symbol, col: ret
    weights: dict[str, float],
) -> float:
    if not weights:
        return 0.0
    s = 0.0
    for sym, w in weights.items():
        if sym in rets.index:
            s += w * float(rets.loc[sym])
    return s


def run_backtest(
    features_parquet: Path,
    labels_parquet: Path,
    model_path: Path,
    horizon: str,
    spec: PortfolioSpec,
    costs: CostModel = CostModel(),
) -> BacktestResult:
    """
    Simple daily-rebalance backtest using next-day realized returns (1D) as PnL.

    Notes:
    - This is an MVP evaluator to enforce costs/turnover. For longer horizons,
      the portfolio construction should be aligned to holding periods and
      overlapping holdings. We'll implement that next as an upgrade.
    """
    feats = read_df_parquet(features_parquet)
    labels = read_df_parquet(labels_parquet)
    df = feats.join(labels, how="inner")

    tm: TrainedModel = load_model(model_path)

    # Use 1D realized return for PnL accounting (execution next close)
    if "r_fwd_1d" not in df.columns:
        raise ValueError("labels parquet must include r_fwd_1d for PnL accounting")

    # Iterate by date
    dates = pd.DatetimeIndex(df.index.get_level_values(0).unique()).sort_values()
    equity = 1.0
    eq = {}
    dr = {}
    to = {}

    prev_w: dict[str, float] = {}
    for d in dates:
        day = df.xs(d, level=0)
        if len(day) == 0:
            continue
        day = day.dropna(subset=tm.feature_cols + ["r_fwd_1d"])
        if len(day) == 0:
            continue

        pred = tm.predict(day)
        scores = {sym: float(s) for sym, s in zip(day.index.tolist(), pred, strict=False)}

        longs = topk_from_scores(scores, k=spec.long_k, ascending=False)
        shorts = topk_from_scores(scores, k=spec.short_k, ascending=True) if spec.long_short else []
        w = (
            make_long_short_weights(longs, shorts, max_weight=spec.max_weight)
            if spec.long_short
            else {s: v for s, v in make_long_short_weights(longs, [], max_weight=spec.max_weight).items() if v > 0}
        )

        turnover = estimate_turnover(prev_w, w)
        cost = turnover * 2 * costs.cost_fraction()  # round-trip approx

        rets = day["r_fwd_1d"]
        pnl = _daily_pnl_from_weights(rets=rets, weights=w)
        pnl_net = pnl - cost

        equity *= (1.0 + pnl_net)
        eq[pd.Timestamp(d)] = equity
        dr[pd.Timestamp(d)] = pnl_net
        to[pd.Timestamp(d)] = turnover

        prev_w = w

    eqs = pd.Series(eq).sort_index()
    drs = pd.Series(dr).sort_index()
    tos = pd.Series(to).sort_index()
    summary = summarize_returns(drs)
    return BacktestResult(equity_curve=eqs, daily_returns=drs, turnover=tos, summary=summary)

