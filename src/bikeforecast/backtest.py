"""Rolling-origin backtest with an expanding window.

At the start of each month every model is retrained on all earlier data and then predicts
that month. All features are lagged by at least 24h, so this gives the same numbers as making
a new day-ahead forecast every night.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from bikeforecast import metrics
from bikeforecast.config import TARGET

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive

    @property
    def name(self) -> str:
        return self.test_start.strftime("%Y-%m")


def monthly_folds(start: str, end: str) -> list[Fold]:
    """One fold per month in [start, end). Training data is everything before that month."""
    months = pd.date_range(start, end, freq="MS")
    return [Fold(m, m, m + pd.offsets.MonthBegin(1)) for m in months if m < pd.Timestamp(end)]


def run_backtest(
    data: pd.DataFrame,
    model_factories: dict[str, Callable[[], object]],
    folds: list[Fold],
    interval_model: str | None = None,
) -> pd.DataFrame:
    """One row per forecast hour with the actual value and each model's forecast."""
    frames = []
    for fold in folds:
        train = data[data.index < fold.train_end]
        test = data[(data.index >= fold.test_start) & (data.index < fold.test_end)]
        result = pd.DataFrame({"actual": test[TARGET], "fold": fold.name}, index=test.index)
        for name, factory in model_factories.items():
            model = factory().fit(train, train[TARGET])
            result[name] = model.predict(test)
            if name == interval_model:
                result["low"], result["high"] = model.predict_interval(test)
        logger.info(
            "Fold %s: trained on %d hours, forecast %d hours", fold.name, len(train), len(test)
        )
        frames.append(result)
    return pd.concat(frames)


def summarize(predictions: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = {
        name: metrics.all_metrics(predictions["actual"], predictions[name]) for name in model_names
    }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("mae")


def summarize_by(predictions: pd.DataFrame, model_names: list[str], key) -> pd.DataFrame:
    """MAE per model and group (e.g. month or hour of day)."""
    grouped = predictions.groupby(key)
    return pd.DataFrame(
        {
            name: grouped.apply(lambda g, n=name: metrics.mae(g["actual"], g[n]))
            for name in model_names
        }
    )
