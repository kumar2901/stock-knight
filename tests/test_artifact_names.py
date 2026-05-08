"""Model artifact path conventions."""
from __future__ import annotations

from pathlib import Path

import pytest

from stock_knight.train.artifact_names import (
    model_artifact_path,
    model_artifact_stem,
    normalise_target_key,
)


def test_raw_stem_unchanged() -> None:
    assert model_artifact_stem("ridge", "1d", "raw") == "ridge_1d"
    assert model_artifact_stem("lgbm", "7d", "RAW") == "lgbm_7d"


def test_non_raw_suffix() -> None:
    assert model_artifact_stem("lgbm", "7d", "excess") == "lgbm_7d_excess"
    assert model_artifact_stem("lgbm", "1m", "sector_residual") == "lgbm_1m_sector_residual"


def test_model_artifact_path() -> None:
    root = Path("/tmp/m")
    assert model_artifact_path(root, "xgb", "12m", "excess") == root / "xgb_12m_excess.joblib"


def test_normalise_unknown() -> None:
    with pytest.raises(ValueError):
        normalise_target_key("nope")
