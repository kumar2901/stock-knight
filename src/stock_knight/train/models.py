from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    # Import can fail at runtime on macOS if libomp is missing; keep as optional.
    lgb = None  # type: ignore[assignment]

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None  # type: ignore[assignment]


ModelKind = Literal["ridge", "lgbm", "lgbm_rank", "xgb", "rf", "ensemble"]


@dataclass(frozen=True)
class TrainedModel:
    kind: ModelKind
    horizon: str
    feature_cols: list[str]
    model: object  # for "ensemble" this is dict[ModelKind, object]
    member_kinds: list[str] = field(default_factory=list)  # only for ensembles

    def predictions_are_fractional_returns(self) -> bool:
        """True if ``predict()`` is in forward-return units (e.g. 0.02 ⇒ +2%)."""
        return self.kind not in ("lgbm_rank", "ensemble")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "ridge":
            return self.model.predict(X[self.feature_cols].to_numpy())
        if self.kind in ("lgbm", "lgbm_rank"):
            return self.model.predict(
                X[self.feature_cols], num_iteration=self.model.best_iteration
            )
        if self.kind == "xgb":
            # xgb.Booster needs DMatrix; XGBRegressor sklearn API takes the DataFrame directly.
            if xgb is not None and isinstance(self.model, xgb.Booster):
                dmat = xgb.DMatrix(X[self.feature_cols].to_numpy())
                return self.model.predict(dmat)
            return self.model.predict(X[self.feature_cols].to_numpy())
        if self.kind == "rf":
            return self.model.predict(X[self.feature_cols].to_numpy())
        if self.kind == "ensemble":
            return _ensemble_predict(self, X)
        raise ValueError(f"unknown kind={self.kind}")


def _ensemble_predict(tm: "TrainedModel", X: pd.DataFrame) -> np.ndarray:
    """Average the rank-percentiles of each member model.

    Rank-averaging is robust to scale differences between members (LightGBM
    returns predicted return units, ridge returns the same, but RF often has
    a much smaller dynamic range). Working in rank space gives each member
    one equal vote on relative ordering, which is exactly what the composite
    cares about.
    """
    if not isinstance(tm.model, dict):
        raise ValueError("ensemble TrainedModel.model must be a dict of members")
    members: dict[str, object] = tm.model  # kind -> fitted model
    if not members:
        raise ValueError("ensemble has no member models")

    rank_sum: np.ndarray | None = None
    n_members = 0
    for kind, mdl in members.items():
        # Build a transient TrainedModel for each member to reuse predict logic
        member_tm = TrainedModel(
            kind=kind,  # type: ignore[arg-type]
            horizon=tm.horizon,
            feature_cols=tm.feature_cols,
            model=mdl,
        )
        try:
            preds = member_tm.predict(X)
        except Exception:
            # Skip a broken member rather than failing the whole ensemble.
            continue
        # Rank-percentile in [0, 1] preserves order, normalises scale.
        ranks = pd.Series(preds).rank(pct=True, method="average").to_numpy()
        rank_sum = ranks if rank_sum is None else rank_sum + ranks
        n_members += 1

    if rank_sum is None or n_members == 0:
        raise RuntimeError("ensemble could not produce predictions from any member")

    return rank_sum / n_members


def train_ridge(X: pd.DataFrame, y: pd.Series, horizon: str) -> TrainedModel:
    feature_cols = [c for c in X.columns if c.endswith("_z")]
    m = Ridge(alpha=5.0, random_state=7)
    m.fit(X[feature_cols].to_numpy(), y.to_numpy())
    return TrainedModel(kind="ridge", horizon=horizon, feature_cols=feature_cols, model=m)


def train_lgbm(X: pd.DataFrame, y: pd.Series, horizon: str) -> TrainedModel:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    feature_cols = [c for c in X.columns if c.endswith("_z")]

    # Split last 10% of training data as internal validation for early stopping
    cutoff = int(len(X) * 0.9)
    X_tr, X_val = X[feature_cols].iloc[:cutoff], X[feature_cols].iloc[cutoff:]
    y_tr, y_val = y.iloc[:cutoff], y.iloc[cutoff:]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        # Regression on returns, evaluated by correlation (IC) proxy
        "objective": "regression_l1",   # MAE loss — more robust to outlier returns
        "metric": "mae",
        "learning_rate": 0.03,          # Slower learning = more generalizable
        "num_leaves": 31,               # Reduced from 63 to prevent overfitting
        "max_depth": 6,                 # Limit tree depth
        "min_data_in_leaf": 150,        # Each leaf needs 150+ samples
        "feature_fraction": 0.7,        # Use 70% of features per tree
        "bagging_fraction": 0.7,        # Use 70% of rows per tree
        "bagging_freq": 1,
        "lambda_l1": 0.1,               # L1 regularization
        "lambda_l2": 0.5,               # L2 regularization
        "min_gain_to_split": 0.01,      # Only split if meaningful gain
        "seed": 7,
        "verbosity": -1,
        "n_jobs": -1,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return TrainedModel(kind="lgbm", horizon=horizon, feature_cols=feature_cols, model=booster)


def train_xgb(X: pd.DataFrame, y: pd.Series, horizon: str) -> TrainedModel:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    feature_cols = [c for c in X.columns if c.endswith("_z")]

    cutoff = int(len(X) * 0.9)
    X_tr, X_val = X[feature_cols].iloc[:cutoff], X[feature_cols].iloc[cutoff:]
    y_tr, y_val = y.iloc[:cutoff], y.iloc[cutoff:]

    dtrain = xgb.DMatrix(X_tr.to_numpy(), label=y_tr.to_numpy())
    dval = xgb.DMatrix(X_val.to_numpy(), label=y_val.to_numpy())

    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 150,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "gamma": 0.01,
        "seed": 7,
        "verbosity": 0,
        "nthread": -1,
        "tree_method": "hist",
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, "valid")],
        early_stopping_rounds=75,
        verbose_eval=False,
    )
    return TrainedModel(kind="xgb", horizon=horizon, feature_cols=feature_cols, model=booster)


def train_rf(X: pd.DataFrame, y: pd.Series, horizon: str) -> TrainedModel:
    feature_cols = [c for c in X.columns if c.endswith("_z")]
    m = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=150,
        max_features=0.7,
        bootstrap=True,
        n_jobs=-1,
        random_state=7,
    )
    m.fit(X[feature_cols].to_numpy(), y.to_numpy())
    return TrainedModel(kind="rf", horizon=horizon, feature_cols=feature_cols, model=m)


# ──────────────────────────────────────────────────────────────────────────
# LambdaRank: optimises Top-K precision rather than absolute return error.
# Best fit for 7D / 1M horizons where we care about ordering, not magnitude.
# ──────────────────────────────────────────────────────────────────────────


def _build_rank_groups(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sort `X` by date so LightGBM's group sizes correspond to per-day cohorts.

    Returns the sorted DataFrame, the integer label array (relative-rank within
    each day) and the per-group sizes vector that lgbm needs for ranking.
    """
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("LambdaRank requires a (date, symbol) MultiIndex on X")
    dates = X.index.get_level_values(0)
    order = np.argsort(dates.values, kind="stable")
    X_sorted = X.iloc[order]
    dates_sorted = X_sorted.index.get_level_values(0)
    group_sizes = (
        pd.Series(dates_sorted).groupby(dates_sorted, sort=False).size().to_numpy()
    )
    return X_sorted, order, group_sizes


def train_lgbm_rank(X: pd.DataFrame, y: pd.Series, horizon: str) -> TrainedModel:
    """LightGBM with `lambdarank` objective grouped by date.

    Targets are quantised into integer ranks per day (0..n-1) which LambdaRank
    expects. We split the trailing 10% of *dates* (not rows) for early
    stopping so validation respects the time ordering.
    """
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    feature_cols = [c for c in X.columns if c.endswith("_z")]
    if not feature_cols:
        raise ValueError("LambdaRank: no z-scored feature columns found")

    # Re-sort X+y together by date so groups line up.
    X_sorted, order, _ = _build_rank_groups(X)
    y_sorted = y.iloc[order]

    # Quantise y into integer ranks within each date (higher = better).
    dates = X_sorted.index.get_level_values(0)
    rank_label = (
        pd.Series(y_sorted.to_numpy(), index=dates)
        .groupby(level=0, sort=False)
        .rank(method="dense", ascending=True)
        .astype(int)
        - 1
    )

    # Time-respecting split: trailing 10% of unique dates -> validation.
    unique_dates = pd.Index(dates.unique())
    cutoff_idx = max(1, int(len(unique_dates) * 0.9))
    cutoff_date = unique_dates[cutoff_idx - 1]
    train_mask = dates <= cutoff_date
    valid_mask = ~train_mask

    if not valid_mask.any():
        train_mask = np.ones(len(dates), dtype=bool)
        valid_mask = np.zeros(len(dates), dtype=bool)

    X_tr = X_sorted.loc[train_mask, feature_cols]
    X_val = X_sorted.loc[valid_mask, feature_cols]
    y_tr = rank_label[train_mask].to_numpy()
    y_val = rank_label[valid_mask].to_numpy()
    grp_tr = (
        pd.Series(dates[train_mask]).groupby(dates[train_mask], sort=False).size().to_numpy()
    )

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=grp_tr)
    valid_sets = [dtrain]
    valid_names = ["train"]
    if valid_mask.any():
        grp_val = (
            pd.Series(dates[valid_mask])
            .groupby(dates[valid_mask], sort=False)
            .size()
            .to_numpy()
        )
        dval = lgb.Dataset(X_val, label=y_val, group=grp_val, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("valid")

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20, 30],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "min_gain_to_split": 0.01,
        "label_gain": list(range(2048)),  # accommodates very large universes
        "seed": 7,
        "verbosity": -1,
        "n_jobs": -1,
    }
    callbacks = [lgb.log_evaluation(period=-1)]
    if valid_mask.any():
        callbacks.append(lgb.early_stopping(stopping_rounds=75, verbose=False))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return TrainedModel(
        kind="lgbm_rank", horizon=horizon, feature_cols=feature_cols, model=booster
    )


# ──────────────────────────────────────────────────────────────────────────
# Ensemble: train all four members and store them in one TrainedModel.
# ──────────────────────────────────────────────────────────────────────────


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: str,
    members: tuple[str, ...] = ("lgbm", "xgb", "ridge", "rf"),
) -> TrainedModel:
    """Fit the requested member models on the same (X, y) and bundle them.

    A member is silently skipped if its underlying library isn't importable
    (e.g. xgboost without libomp on macOS) so the ensemble still trains.
    """
    feature_cols = [c for c in X.columns if c.endswith("_z")]
    fitted: dict[str, object] = {}
    for kind in members:
        try:
            if kind == "lgbm":
                tm = train_lgbm(X, y, horizon=horizon)
            elif kind == "lgbm_rank":
                tm = train_lgbm_rank(X, y, horizon=horizon)
            elif kind == "xgb":
                tm = train_xgb(X, y, horizon=horizon)
            elif kind == "ridge":
                tm = train_ridge(X, y, horizon=horizon)
            elif kind == "rf":
                tm = train_rf(X, y, horizon=horizon)
            else:
                continue
            fitted[kind] = tm.model
        except Exception:
            continue

    if not fitted:
        raise RuntimeError(
            f"Ensemble could not fit any of the requested members: {members}"
        )

    return TrainedModel(
        kind="ensemble",
        horizon=horizon,
        feature_cols=feature_cols,
        model=fitted,
        member_kinds=list(fitted.keys()),
    )


def winsorize_per_date(y: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip forward returns at the (lower, upper) percentiles within each date.

    Reduces COVID-day / earnings-shock contamination of the training signal
    without changing rank order materially. Operates on a Series indexed by
    (date, symbol).
    """
    if not isinstance(y.index, pd.MultiIndex):
        return y
    dates = y.index.get_level_values(0)
    grouped = y.groupby(dates, sort=False)
    # Compute per-day quantile bounds and broadcast back to row-level.
    lo = grouped.transform(lambda s: s.quantile(lower))
    hi = grouped.transform(lambda s: s.quantile(upper))
    return y.clip(lower=lo, upper=hi)


def save_model(path: Path, tm: TrainedModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": tm.kind,
        "horizon": tm.horizon,
        "feature_cols": tm.feature_cols,
        "model": tm.model,
        "member_kinds": list(tm.member_kinds),
    }
    joblib.dump(payload, path)


def load_model(path: Path) -> TrainedModel:
    payload = joblib.load(path)
    return TrainedModel(
        kind=payload["kind"],
        horizon=payload["horizon"],
        feature_cols=list(payload["feature_cols"]),
        model=payload["model"],
        member_kinds=list(payload.get("member_kinds", []) or []),
    )

