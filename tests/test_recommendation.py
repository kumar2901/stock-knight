"""End-to-end test that rank_for_date emits recommendation + confidence_pct."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_knight.ranking.daily import rank_for_date, write_ranklists
from stock_knight.storage import write_df_parquet
from stock_knight.train.models import save_model, train_ridge


def test_rank_for_date_emits_recommendation_and_confidence(
    features_labels, mock_ohlcv, tmp_path: Path
) -> None:
    feats, labels = features_labels

    df = feats.join(labels, how="inner").dropna(subset=["r_fwd_7d"])
    feature_cols = [c for c in feats.columns if c.endswith("_z")]
    usable = [c for c in feature_cols if df[c].notna().mean() >= 0.8]
    df = df[usable + ["r_fwd_7d"]].dropna()
    tm = train_ridge(df, df["r_fwd_7d"], horizon="7d")

    feats_parquet = tmp_path / "features.parquet"
    write_df_parquet(feats_parquet, feats)

    ohlcv_parquet = tmp_path / "ohlcv.parquet"
    write_df_parquet(ohlcv_parquet, mock_ohlcv)

    model_path = tmp_path / "ridge_7d.joblib"
    save_model(model_path, tm)

    asof = pd.DatetimeIndex(feats.index.get_level_values(0)).normalize().max()

    ranks = rank_for_date(
        features_parquet=feats_parquet,
        model_path=model_path,
        ohlcv_parquet=ohlcv_parquet,
        asof=asof,
        horizon="7d",
        rank_method="composite",
    )

    assert ranks.rank_method == "composite"
    assert ranks.composite_weights is not None
    assert len(ranks.rows) == len(ranks.ranked)

    recos = {r["recommendation"] for r in ranks.rows}
    assert recos.issubset({"BUY", "HOLD", "SELL"})
    assert "BUY" in recos
    assert "SELL" in recos

    for r in ranks.rows:
        assert 0.0 <= float(r["confidence_pct"]) <= 100.0
        assert "composite" in r
        assert "expected_return_z" in r

    out_path = tmp_path / "ranks.json"
    write_ranklists(out_path, ranks)
    payload = json.loads(out_path.read_text())
    assert payload["rank_method"] == "composite"
    assert payload["composite_weights"]["expected_return"] == 0.40
    assert payload["rows"][0]["recommendation"] == "BUY"


def test_rank_for_date_handles_duplicate_symbol_rows(
    features_labels, mock_ohlcv, tmp_path: Path
) -> None:
    """Regression test: features parquet with duplicate (date, symbol) rows
    must not crash rank_for_date — duplicates are deduped with a warning."""
    import warnings

    feats, labels = features_labels

    df = feats.join(labels, how="inner").dropna(subset=["r_fwd_7d"])
    feature_cols = [c for c in feats.columns if c.endswith("_z")]
    usable = [c for c in feature_cols if df[c].notna().mean() >= 0.8]
    df = df[usable + ["r_fwd_7d"]].dropna()
    tm = train_ridge(df, df["r_fwd_7d"], horizon="7d")

    # Inject a duplicate (date, symbol) row to simulate overlapping ingests.
    asof = pd.DatetimeIndex(feats.index.get_level_values(0)).normalize().max()
    one_row = feats.xs(asof, level=0).head(1).copy()
    sym = one_row.index[0]
    dup_row = one_row.copy()
    dup_row.index = pd.Index([sym], name="symbol")
    dup_row["__date_helper"] = asof
    dup_row = dup_row.set_index("__date_helper", append=True).swaplevel(0, 1)
    dup_row.index.names = feats.index.names
    feats_dup = pd.concat([feats, dup_row]).sort_index()
    assert not feats_dup.loc[asof].index.is_unique

    feats_parquet = tmp_path / "features_dup.parquet"
    write_df_parquet(feats_parquet, feats_dup)
    model_path = tmp_path / "ridge_7d.joblib"
    save_model(model_path, tm)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ranks = rank_for_date(
            features_parquet=feats_parquet,
            model_path=model_path,
            ohlcv_parquet=None,
            asof=asof,
            horizon="7d",
            rank_method="composite",
        )
    msgs = [str(x.message) for x in w]
    assert any("duplicate symbol rows" in m for m in msgs), msgs

    seen = [r["symbol"] for r in ranks.rows]
    assert len(seen) == len(set(seen)), "rows must have unique symbols after dedupe"


def test_rank_for_date_score_only_mode(
    features_labels, mock_ohlcv, tmp_path: Path
) -> None:
    feats, labels = features_labels

    df = feats.join(labels, how="inner").dropna(subset=["r_fwd_7d"])
    feature_cols = [c for c in feats.columns if c.endswith("_z")]
    usable = [c for c in feature_cols if df[c].notna().mean() >= 0.8]
    df = df[usable + ["r_fwd_7d"]].dropna()
    tm = train_ridge(df, df["r_fwd_7d"], horizon="7d")

    feats_parquet = tmp_path / "features.parquet"
    write_df_parquet(feats_parquet, feats)
    model_path = tmp_path / "ridge_7d.joblib"
    save_model(model_path, tm)

    asof = pd.DatetimeIndex(feats.index.get_level_values(0)).normalize().max()
    ranks = rank_for_date(
        features_parquet=feats_parquet,
        model_path=model_path,
        ohlcv_parquet=None,
        asof=asof,
        horizon="7d",
        rank_method="score_only",
    )
    assert ranks.rank_method == "score_only"
    assert ranks.composite_weights is None
    # The legacy composite_* fields should NOT appear in score_only mode.
    assert "composite" not in ranks.rows[0]
    # Recommendation/confidence still computed.
    assert ranks.rows[0]["recommendation"] in {"BUY", "HOLD", "SELL"}
