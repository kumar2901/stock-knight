from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_knight.storage import read_df_parquet, write_json
from stock_knight.train.models import TrainedModel, load_model


@dataclass(frozen=True)
class RankLists:
    asof: pd.Timestamp
    horizon: str
    top10: list[str]
    top20: list[str]
    top30: list[str]
    bottom10: list[str]
    bottom20: list[str]
    bottom30: list[str]


def rank_for_date(
    features_parquet: Path,
    model_path: Path,
    asof: pd.Timestamp,
    horizon: str,
    min_non_na: bool = True,
) -> RankLists:
    feats = read_df_parquet(features_parquet)
    asof = pd.to_datetime(asof).tz_localize(None)
    day = feats.xs(asof, level=0)

    tm: TrainedModel = load_model(model_path)
    if min_non_na:
        day = day.dropna(subset=tm.feature_cols)
    if len(day) == 0:
        raise ValueError(f"No feature rows available for {asof.date()}")

    pred = tm.predict(day)
    scores = pd.Series(pred, index=day.index, name="score").sort_values(ascending=False)

    top10 = scores.head(10).index.tolist()
    top20 = scores.head(20).index.tolist()
    top30 = scores.head(30).index.tolist()
    bottom10 = scores.tail(10).index.tolist()[::-1]
    bottom20 = scores.tail(20).index.tolist()[::-1]
    bottom30 = scores.tail(30).index.tolist()[::-1]

    return RankLists(
        asof=asof,
        horizon=horizon,
        top10=top10,
        top20=top20,
        top30=top30,
        bottom10=bottom10,
        bottom20=bottom20,
        bottom30=bottom30,
    )


def write_ranklists(out_path: Path, ranks: RankLists) -> None:
    write_json(
        out_path,
        {
            "asof": str(ranks.asof),
            "horizon": ranks.horizon,
            "top10": ranks.top10,
            "top20": ranks.top20,
            "top30": ranks.top30,
            "bottom10": ranks.bottom10,
            "bottom20": ranks.bottom20,
            "bottom30": ranks.bottom30,
        },
    )

