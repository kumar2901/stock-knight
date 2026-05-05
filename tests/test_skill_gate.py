"""Tests for skill_gate (Phase 1c/1d/3c)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_knight.ranking.skill_gate import (
    LOW_SKILL_IC_THRESHOLD,
    MIN_HIT_RATE,
    MIN_PICKS_FOR_HIT_RATE,
    calibrated_confidence,
    load_skill_summary,
    per_symbol_hit_rate,
    regime_flag,
)


def _write_metrics(path: Path, ics: list[float], top10s: list[float]) -> None:
    folds = [
        {"ic_spearman": ic, "top10_spread": t10}
        for ic, t10 in zip(ics, top10s, strict=True)
    ]
    path.write_text(json.dumps({"folds": folds}))


def test_skill_summary_high_skill_regime(tmp_path) -> None:
    model_path = tmp_path / "lgbm_7d.joblib"
    model_path.write_bytes(b"")  # empty placeholder
    _write_metrics(
        tmp_path / "lgbm_7d_metrics.json",
        ics=[0.05, 0.04, 0.06, 0.05],
        top10s=[0.02, 0.03, 0.025, 0.022],
    )
    s = load_skill_summary(model_path)
    assert not s.is_unknown
    assert not s.is_low_skill
    assert regime_flag(s) == "ok"
    assert s.skill_factor() > 0.8


def test_skill_summary_low_skill_regime(tmp_path) -> None:
    model_path = tmp_path / "lgbm_7d.joblib"
    model_path.write_bytes(b"")
    _write_metrics(
        tmp_path / "lgbm_7d_metrics.json",
        ics=[0.001, -0.002, 0.003, 0.0],   # all below threshold
        top10s=[-0.01, -0.005, -0.008, -0.002],
    )
    s = load_skill_summary(model_path)
    assert s.is_low_skill
    assert regime_flag(s) == "low_skill"
    assert s.skill_factor() == 0.0


def test_skill_summary_unknown_when_no_metrics(tmp_path) -> None:
    model_path = tmp_path / "lgbm_7d.joblib"
    s = load_skill_summary(model_path)
    assert s.is_unknown
    assert regime_flag(s) == "unknown"
    # Neutral 0.5 factor so we still produce non-zero confidences.
    assert s.skill_factor() == 0.5


def test_calibrated_confidence_shrinks_in_low_skill_regime(tmp_path) -> None:
    model_path = tmp_path / "lgbm_7d.joblib"
    model_path.write_bytes(b"")
    _write_metrics(
        tmp_path / "lgbm_7d_metrics.json",
        ics=[0.0, 0.0, 0.0, 0.0],
        top10s=[0.0, 0.0, 0.0, 0.0],
    )
    s_low = load_skill_summary(model_path)

    _write_metrics(
        tmp_path / "lgbm_7d_metrics.json",
        ics=[0.05, 0.05, 0.05, 0.05],
        top10s=[0.02, 0.02, 0.02, 0.02],
    )
    s_high = load_skill_summary(model_path)

    syms = [f"S{i}" for i in range(20)]
    scores = pd.Series(np.linspace(-1.0, 1.0, len(syms)), index=syms)
    conf_low = calibrated_confidence(scores, s_low, cap=95.0)
    conf_high = calibrated_confidence(scores, s_high, cap=95.0)

    assert conf_low.max() < conf_high.max()
    assert conf_high.max() <= 95.0
    # Low-skill ceiling should be substantially below high-skill ceiling.
    assert conf_low.max() < 30.0
    assert conf_high.max() > 80.0


def test_per_symbol_hit_rate_filters_low_count(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()

    # Build a labels parquet with two dates and three symbols.
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    syms = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    labels = pd.DataFrame(
        {"r_fwd_7d": [0.05, 0.01, -0.02, 0.04, -0.01, -0.03]}, index=idx
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path)

    # Build two ranklist JSONs that both pick A and B as BUY.
    for d in dates:
        payload = {
            "asof": str(d),
            "horizon": "7d",
            "rows": [
                {"symbol": "A", "recommendation": "BUY"},
                {"symbol": "B", "recommendation": "BUY"},
                {"symbol": "C", "recommendation": "HOLD"},
            ],
        }
        (reports / f"ranks_7d_{d.date()}.json").write_text(json.dumps(payload))

    # Without enough picks per symbol the result is empty.
    hr = per_symbol_hit_rate(reports, labels_path, horizon="7d", lookback_files=100)
    assert hr.empty  # only 2 picks per symbol; threshold is 20


def test_per_symbol_hit_rate_with_enough_picks(tmp_path, monkeypatch) -> None:
    """Drop the threshold so the toy fixture exercises the hit-rate logic."""
    monkeypatch.setattr(
        "stock_knight.ranking.skill_gate.MIN_PICKS_FOR_HIT_RATE", 2
    )

    reports = tmp_path / "reports"
    reports.mkdir()

    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    syms = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    # Cohort means: day1 = mean(0.05, 0.01, -0.02) = 0.0133; day2 = mean(0.04,-0.01,-0.03) = 0.0
    # A beats both means (always wins); B never beats them.
    labels = pd.DataFrame(
        {"r_fwd_7d": [0.05, 0.01, -0.02, 0.04, -0.01, -0.03]}, index=idx
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path)

    for d in dates:
        payload = {
            "asof": str(d),
            "horizon": "7d",
            "rows": [
                {"symbol": "A", "recommendation": "BUY"},
                {"symbol": "B", "recommendation": "BUY"},
                {"symbol": "C", "recommendation": "HOLD"},
            ],
        }
        (reports / f"ranks_7d_{d.date()}.json").write_text(json.dumps(payload))

    hr = per_symbol_hit_rate(reports, labels_path, horizon="7d", lookback_files=100)
    assert "A" in hr.index and "B" in hr.index
    assert hr.loc["A"] == pytest.approx(1.0)
    assert hr.loc["B"] == pytest.approx(0.0)
