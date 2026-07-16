"""
Gradient-boosting signal model (LightGBM).

This is the primary model — most tabular alpha research in practice
uses GBMs over deep nets because tabular financial features are
low-dimensional, noisy, and non-stationary; trees handle that far more
data-efficiently than a network needs to learn feature interactions
from scratch. The LSTM variant (models/lstm_model.py) is the stretch
option to show range on sequence modeling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class LGBMConfig:
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 5
    num_leaves: int = 31
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    objective: str = "regression"
    early_stopping_rounds: int = 50


class GBMSignalModel:
    def __init__(self, cfg: LGBMConfig):
        self.cfg = cfg
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "GBMSignalModel":
        self.feature_names_ = list(X_train.columns)
        params = dict(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            num_leaves=self.cfg.num_leaves,
            min_child_samples=self.cfg.min_child_samples,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            objective=self.cfg.objective,
            verbosity=-1,
            random_state=42,
        )
        self.model = lgb.LGBMRegressor(**params)

        fit_kwargs = {}
        callbacks = []
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit yet")
        return self.model.predict(X)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fit yet")
        booster = self.model.booster_
        imp = booster.feature_importance(importance_type=importance_type)
        s = pd.Series(imp, index=self.feature_names_, name=f"importance_{importance_type}")
        return s.sort_values(ascending=False)
