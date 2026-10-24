"""Forecasting models with a simple fit(X, y) / predict(X) interface.

- SeasonalNaive: same hour last week, the baseline
- RidgeForecaster: linear model on log demand with Fourier terms
- LGBMForecaster: LightGBM with a Poisson objective on a level-normalised target, with
  optional conformal prediction intervals

Demand grew 65% between 2011 and 2012 and tree models can't extrapolate, so LGBMForecaster
predicts demand divided by the last 7 days' level and scales the result back.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bikeforecast.config import RANDOM_STATE
from bikeforecast.features import (
    CALENDAR_FEATURES,
    FOURIER_FEATURES,
    LAG_HOURS,
    WEATHER_FEATURES,
)

LAG_COLUMNS = [f"lag_{h}" for h in LAG_HOURS] + ["same_hour_mean_7d"]


class SeasonalNaive:
    def __init__(self, lag_column: str = "lag_168"):
        self.lag_column = lag_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag_column].to_numpy(dtype=float)


class RidgeForecaster:
    """Ridge regression on log1p(demand) with one-hot hour x working-day effects."""

    def __init__(self, alpha: float = 1.0, use_weather: bool = True):
        self.alpha = alpha
        self.use_weather = use_weather

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        design = pd.DataFrame(index=X.index)
        design["hour_daytype"] = X["hour"].astype(str) + "_" + X["workingday"].astype(str)
        design["month"] = X["month"].astype(str)
        design["weather_cat"] = X["weather"].astype(str)
        for column in LAG_COLUMNS + ["level_7d"]:
            design[f"log_{column}"] = np.log1p(X[column])
        for column in FOURIER_FEATURES + ["holiday"]:
            design[column] = X[column]
        if self.use_weather:
            for column in ["temp_c", "feels_like_c", "humidity", "windspeed_kmh"]:
                design[column] = X[column]
            design["temp_sq"] = X["temp_c"] ** 2
        return design

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecaster:
        design = self._design(X)
        categorical = ["hour_daytype", "month"] + (["weather_cat"] if self.use_weather else [])
        design = design.drop(columns=[] if self.use_weather else ["weather_cat"])
        numeric = [c for c in design.columns if c not in categorical]
        self.model_ = make_pipeline(
            ColumnTransformer(
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ("num", StandardScaler(), numeric),
                ]
            ),
            Ridge(alpha=self.alpha),
        )
        self.model_.fit(design, np.log1p(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = self._design(X)
        if not self.use_weather:
            design = design.drop(columns=["weather_cat"])
        return np.clip(np.expm1(self.model_.predict(design)), 0, None)


class LGBMForecaster:
    """LightGBM on demand / 7-day level, with optional 80% prediction intervals.

    Plain quantile models were over-confident here (their 80% interval covered ~63% of the
    actuals). With intervals=True the quantile models are fitted without the last
    `calibration_days`, we measure how far the actuals fall outside the interval on those days,
    and widen the interval by that amount (split conformal / CQR, Romano et al. 2019).
    """

    def __init__(
        self,
        use_weather: bool = True,
        intervals: bool = False,
        quantiles: tuple[float, float] = (0.1, 0.9),
        calibration_days: int = 28,
        **params,
    ):
        self.use_weather = use_weather
        self.intervals = intervals
        self.quantiles = quantiles
        self.calibration_days = calibration_days
        self.params = {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,  # ~17k rows, more threads were slower here
            "verbose": -1,
            **params,
        }

    @property
    def feature_names(self) -> list[str]:
        features = CALENDAR_FEATURES + [f"{c}_rel" for c in LAG_COLUMNS] + ["level_7d"]
        return features + (WEATHER_FEATURES if self.use_weather else [])

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        design = X.copy()
        level = X["level_7d"].clip(lower=1)
        for column in LAG_COLUMNS:
            design[f"{column}_rel"] = X[column] / level
        return design[self.feature_names]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LGBMForecaster:
        design = self._design(X)
        ratio = y.to_numpy() / X["level_7d"].clip(lower=1).to_numpy()
        self.model_ = lgb.LGBMRegressor(objective="poisson", **self.params)
        self.model_.fit(design, ratio, categorical_feature=self._categorical(design))
        self.quantile_models_ = {}
        if self.intervals:
            self._fit_conformal_intervals(design, ratio)
        return self

    def _fit_conformal_intervals(self, design: pd.DataFrame, ratio: np.ndarray) -> None:
        cutoff = design.index.max() - pd.Timedelta(days=self.calibration_days)
        fit_rows = np.asarray(design.index <= cutoff)
        for q in self.quantiles:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            model.fit(
                design[fit_rows], ratio[fit_rows], categorical_feature=self._categorical(design)
            )
            self.quantile_models_[q] = model
        calibration = design[~fit_rows]
        low, high = (self.quantile_models_[q].predict(calibration) for q in self.quantiles)
        scores = np.maximum(low - ratio[~fit_rows], ratio[~fit_rows] - high)
        n, target = len(scores), self.quantiles[1] - self.quantiles[0]
        self.conformal_margin_ = float(np.quantile(scores, min(1.0, target * (n + 1) / n)))

    @staticmethod
    def _categorical(design: pd.DataFrame) -> list[str]:
        return [c for c in ("weather",) if c in design.columns]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        level = X["level_7d"].clip(lower=1).to_numpy()
        return np.clip(self.model_.predict(self._design(X)) * level, 0, None)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.quantile_models_:
            raise RuntimeError("Fit with intervals=True to get prediction intervals")
        level = X["level_7d"].clip(lower=1).to_numpy()
        design = self._design(X)
        low, high = (self.quantile_models_[q].predict(design) for q in self.quantiles)
        low, high = np.minimum(low, high), np.maximum(low, high)
        margin = self.conformal_margin_
        return np.clip((low - margin) * level, 0, None), np.clip((high + margin) * level, 0, None)

    def feature_importance(self) -> pd.Series:
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.feature_names).sort_values(ascending=False)


class RawLGBMForecaster(LGBMForecaster):
    """Same model trained on raw counts, kept for comparison."""

    @property
    def feature_names(self) -> list[str]:
        features = CALENDAR_FEATURES + LAG_COLUMNS + ["level_7d"]
        return features + (WEATHER_FEATURES if self.use_weather else [])

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_names]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RawLGBMForecaster:
        design = self._design(X)
        self.model_ = lgb.LGBMRegressor(objective="poisson", **self.params)
        self.model_.fit(design, y.to_numpy(), categorical_feature=self._categorical(design))
        self.quantile_models_ = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(self._design(X)), 0, None)
