from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def iter_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 5,
    valid_months: int = 6,
    step_months: int = 3,
    embargo_days: int = 5,
) -> Iterator[WalkForwardSplit]:
    """
    Generate walk-forward splits over a trading-day date index.

    Purging/embargo: after train_end, skip `embargo_days` before validation starts.
    """
    d = pd.DatetimeIndex(sorted(pd.to_datetime(dates).tz_localize(None).unique()))
    if len(d) == 0:
        return

    start = d.min()
    end = d.max()

    cursor = start + pd.DateOffset(years=train_years)
    while True:
        train_end = cursor
        valid_start = train_end + pd.Timedelta(days=embargo_days)
        valid_end = valid_start + pd.DateOffset(months=valid_months)
        if valid_end > end:
            break

        train_start = train_end - pd.DateOffset(years=train_years)

        yield WalkForwardSplit(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )

        cursor = cursor + pd.DateOffset(months=step_months)

