from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore[assignment]


ModelKind = Literal["ridge", "lgbm"]


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
    dtrain = lgb.Dataset(X[feature_cols], label=y)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 7,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return TrainedModel(kind="lgbm", horizon=horizon, feature_cols=feature_cols, model=booster)


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

