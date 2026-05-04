from __future__ import annotations

from dataclasses import dataclass
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


ModelKind = Literal["ridge", "lgbm", "xgb", "rf"]


@dataclass(frozen=True)
class TrainedModel:
    kind: ModelKind
    horizon: str
    feature_cols: list[str]
    model: object

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "ridge":
            return self.model.predict(X[self.feature_cols].to_numpy())
        if self.kind == "lgbm":
            return self.model.predict(X[self.feature_cols], num_iteration=self.model.best_iteration)
        if self.kind == "xgb":
            # xgb.Booster needs DMatrix; XGBRegressor sklearn API takes the DataFrame directly.
            if xgb is not None and isinstance(self.model, xgb.Booster):
                dmat = xgb.DMatrix(X[self.feature_cols].to_numpy())
                return self.model.predict(dmat)
            return self.model.predict(X[self.feature_cols].to_numpy())
        if self.kind == "rf":
            return self.model.predict(X[self.feature_cols].to_numpy())
        raise ValueError(f"unknown kind={self.kind}")


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


def save_model(path: Path, tm: TrainedModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": tm.kind,
        "horizon": tm.horizon,
        "feature_cols": tm.feature_cols,
        "model": tm.model,
    }
    joblib.dump(payload, path)


def load_model(path: Path) -> TrainedModel:
    payload = joblib.load(path)
    return TrainedModel(
        kind=payload["kind"],
        horizon=payload["horizon"],
        feature_cols=list(payload["feature_cols"]),
        model=payload["model"],
    )

