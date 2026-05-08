"""Stable model artifact names on disk (joblib + metrics JSON)."""
from __future__ import annotations

from pathlib import Path


def model_artifact_stem(model_kind: str, horizon: str, target: str) -> str:
    """Return filename stem like ``lgbm_7d`` or ``lgbm_7d_excess``.

    ``raw`` keeps the historical layout ``{kind}_{horizon}`` so existing
    ``artifacts/models/ridge_1d.joblib`` paths keep working. Other targets
    use a suffix so training one objective does not overwrite another.
    """
    t = normalise_target_key(target)
    if t == "raw":
        return f"{model_kind}_{horizon}"
    return f"{model_kind}_{horizon}_{t}"


def normalise_target_key(target: str) -> str:
    t = (target or "raw").strip().lower()
    if t in ("", "raw"):
        return "raw"
    if t in ("excess", "sector_residual"):
        return t
    raise ValueError(f"unknown training target={target!r}")


def model_artifact_path(out_dir: Path, model_kind: str, horizon: str, target: str) -> Path:
    return out_dir / f"{model_artifact_stem(model_kind, horizon, target)}.joblib"
