from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_portfolio.features import feature_column_names


class MomentumScoringModel:
    def fit(self, features_train: pd.DataFrame) -> "MomentumScoringModel":
        return self

    def predict_scores(self, features_current: pd.DataFrame) -> pd.Series:
        columns = [column for column in ("momentum_252", "momentum_126", "momentum_63") if column in features_current]
        if not columns:
            raise ValueError("No momentum columns available for scoring.")
        scores = features_current[columns].mean(axis=1)
        return pd.Series(scores.to_numpy(), index=features_current["ticker"], name="score")


class VolatilityAdjustedMomentumModel:
    def fit(self, features_train: pd.DataFrame) -> "VolatilityAdjustedMomentumModel":
        return self

    def predict_scores(self, features_current: pd.DataFrame) -> pd.Series:
        denominator = features_current["volatility_60"].replace(0.0, np.nan)
        scores = features_current["momentum_126"] / denominator
        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
        return pd.Series(scores.to_numpy(), index=features_current["ticker"], name="score")


class SklearnReturnModel:
    def __init__(self, estimator, target_column: str = "forward_return_21") -> None:
        self.estimator = estimator
        self.target_column = target_column
        self.columns_: list[str] | None = None

    def fit(self, features_train: pd.DataFrame) -> "SklearnReturnModel":
        self.columns_ = feature_column_names(features_train, self.target_column)
        train = features_train.dropna(subset=[*self.columns_, self.target_column])
        if train.empty:
            raise ValueError("Cannot fit model with no training rows.")
        self.estimator.fit(train[self.columns_], train[self.target_column])
        return self

    def predict_scores(self, features_current: pd.DataFrame) -> pd.Series:
        if self.columns_ is None:
            raise ValueError("Model must be fit before predicting.")
        current = features_current.dropna(subset=self.columns_).copy()
        predictions = self.estimator.predict(current[self.columns_])
        return pd.Series(predictions, index=current["ticker"], name="score")


def make_ridge_model(random_seed: int = 42, target_column: str = "forward_return_21") -> SklearnReturnModel:
    del random_seed
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    return SklearnReturnModel(estimator, target_column=target_column)


def make_gradient_boosting_model(
    random_seed: int = 42,
    target_column: str = "forward_return_21",
) -> SklearnReturnModel:
    estimator = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=random_seed,
    )
    return SklearnReturnModel(estimator, target_column=target_column)


def make_model(name: str, random_seed: int = 42, target_column: str = "forward_return_21"):
    if name == "momentum":
        return MomentumScoringModel()
    if name == "vol_adjusted_momentum":
        return VolatilityAdjustedMomentumModel()
    if name == "ridge":
        return make_ridge_model(random_seed, target_column)
    if name == "gradient_boosting":
        return make_gradient_boosting_model(random_seed, target_column)
    raise ValueError(f"Unknown model: {name}")
