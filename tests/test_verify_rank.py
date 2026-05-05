"""CLI verify-rank acceptance helper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from stock_knight.cli import cmd_verify_rank


def _payload(rows: list[dict]) -> dict:
    return {
        "asof": "2026-05-04",
        "horizon": "7d",
        "rows": rows,
        "ranked": [r["symbol"] for r in rows],
    }


def test_verify_rank_ok_when_must_not_buy_are_not_buy(tmp_path: Path) -> None:
    fp = tmp_path / "ranks.json"
    fp.write_text(
        json.dumps(
            _payload(
                [
                    {"symbol": "GLAXO", "recommendation": "HOLD", "confidence_pct": 40.0},
                    {"symbol": "OTHER", "recommendation": "BUY", "confidence_pct": 80.0},
                ]
            )
        )
    )
    cmd_verify_rank(
        argparse.Namespace(
            rank_file=str(fp),
            must_not_buy="GLAXO,BHARTIHEXA",
            max_confidence=95.0,
            skip_confidence_check=False,
        )
    )


def test_verify_rank_fails_on_forbidden_buy(tmp_path: Path) -> None:
    fp = tmp_path / "ranks.json"
    fp.write_text(
        json.dumps(
            _payload(
                [{"symbol": "GLAXO", "recommendation": "BUY", "confidence_pct": 50.0}]
            )
        )
    )
    with pytest.raises(SystemExit) as ei:
        cmd_verify_rank(
            argparse.Namespace(
                rank_file=str(fp),
                must_not_buy="GLAXO",
                max_confidence=95.0,
                skip_confidence_check=False,
            )
        )
    assert ei.value.code == 2


def test_verify_rank_fails_on_confidence_above_cap(tmp_path: Path) -> None:
    fp = tmp_path / "ranks.json"
    fp.write_text(
        json.dumps(
            _payload(
                [{"symbol": "X", "recommendation": "HOLD", "confidence_pct": 99.0}]
            )
        )
    )
    with pytest.raises(SystemExit) as ei:
        cmd_verify_rank(
            argparse.Namespace(
                rank_file=str(fp),
                must_not_buy="GLAXO",
                max_confidence=95.0,
                skip_confidence_check=False,
            )
        )
    assert ei.value.code == 3


def test_verify_rank_skips_confidence_when_requested(tmp_path: Path) -> None:
    fp = tmp_path / "ranks.json"
    fp.write_text(
        json.dumps(
            _payload(
                [{"symbol": "X", "recommendation": "HOLD", "confidence_pct": 99.0}]
            )
        )
    )
    cmd_verify_rank(
        argparse.Namespace(
            rank_file=str(fp),
            must_not_buy="GLAXO",
            max_confidence=95.0,
            skip_confidence_check=True,
        )
    )
