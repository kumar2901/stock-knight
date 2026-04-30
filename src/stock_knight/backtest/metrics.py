from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_returns(daily_returns: pd.Series) -> dict:
    dr = daily_returns.dropna()
    if len(dr) == 0:
        return {"n": 0}
    mean = float(dr.mean())
    vol = float(dr.std(ddof=0))
    sharpe = float(mean / vol * np.sqrt(252)) if vol > 0 else float("nan")
    cum = float((1.0 + dr).prod() - 1.0)
    mdd = float(max_drawdown((1.0 + dr).cumprod()))
    return {"n": int(len(dr)), "mean": mean, "vol": vol, "sharpe": sharpe, "cum_return": cum, "max_dd": mdd}


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) == 0:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())

