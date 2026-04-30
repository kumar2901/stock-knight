from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_knight.storage import read_df_parquet, write_json


@dataclass(frozen=True)
class DriftReport:
    horizon: str
    window_days: int
    ic_mean: float
    ic_std: float
    top20_spread_mean: float
    turnover_mean: float | None


def _spearman_ic_by_date(scores: pd.Series, y: pd.Series) -> pd.Series:
    """
    scores and y indexed by (date, symbol)
    """
    df = pd.DataFrame({"score": scores, "y": y}).dropna()
    if len(df) == 0:
        return pd.Series(dtype=float)

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return float("nan")
        r1 = g["score"].rank()
        r2 = g["y"].rank()
        if r1.std(ddof=0) == 0 or r2.std(ddof=0) == 0:
            return float("nan")
        return float(np.corrcoef(r1, r2)[0, 1])

    return df.groupby(level=0).apply(_ic).sort_index()


def drift_report_from_scores(
    scores: pd.Series,
    labels: pd.Series,
    horizon: str,
    window_days: int = 252,
) -> DriftReport:
    ic = _spearman_ic_by_date(scores, labels)
    if len(ic) == 0:
        return DriftReport(
            horizon=horizon,
            window_days=window_days,
            ic_mean=float("nan"),
            ic_std=float("nan"),
            top20_spread_mean=float("nan"),
            turnover_mean=None,
        )

    ic_w = ic.tail(window_days)
    return DriftReport(
        horizon=horizon,
        window_days=window_days,
        ic_mean=float(ic_w.mean()),
        ic_std=float(ic_w.std(ddof=0)),
        top20_spread_mean=float("nan"),
        turnover_mean=None,
    )


def write_drift_report(path: Path, report: DriftReport) -> None:
    write_json(
        path,
        {
            "horizon": report.horizon,
            "window_days": report.window_days,
            "ic_mean": report.ic_mean,
            "ic_std": report.ic_std,
            "top20_spread_mean": report.top20_spread_mean,
            "turnover_mean": report.turnover_mean,
        },
    )

