import numpy as np
import pandas as pd
import pytest

from bikeforecast.backtest import monthly_folds, run_backtest, summarize
from bikeforecast.models import LGBMForecaster, RawLGBMForecaster, RidgeForecaster, SeasonalNaive

FAST = {"n_estimators": 150, "learning_rate": 0.1, "num_leaves": 15}


def test_monthly_folds_are_contiguous_and_ordered():
    folds = monthly_folds("2012-01-01", "2012-04-01")
    assert [f.name for f in folds] == ["2012-01", "2012-02", "2012-03"]
    for fold in folds:
        assert fold.train_end <= fold.test_start < fold.test_end
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end == b.test_start


def test_seasonal_naive_returns_lag(features):
    assert np.array_equal(SeasonalNaive("lag_168").predict(features), features["lag_168"])


@pytest.mark.parametrize(
    "model",
    [
        RidgeForecaster(),
        LGBMForecaster(**FAST),
        RawLGBMForecaster(**FAST),
        LGBMForecaster(use_weather=False, **FAST),
    ],
)
def test_models_fit_and_predict_non_negative(features, model):
    train, test = features.iloc[:-168], features.iloc[-168:]
    prediction = model.fit(train, train["cnt"]).predict(test)
    assert prediction.shape == (168,)
    assert (prediction >= 0).all()


def test_learned_models_beat_naive(features):
    train, test = features.iloc[:-336], features.iloc[-336:]
    naive = np.abs(SeasonalNaive().predict(test) - test["cnt"]).mean()
    model = LGBMForecaster(**FAST).fit(train, train["cnt"])
    assert np.abs(model.predict(test) - test["cnt"]).mean() < naive


def test_conformal_intervals(features):
    train, test = features.iloc[:-168], features.iloc[-168:]
    model = LGBMForecaster(intervals=True, calibration_days=7, **FAST).fit(train, train["cnt"])
    low, high = model.predict_interval(test)
    assert (low <= high).all() and (low >= 0).all()
    assert np.isfinite(model.conformal_margin_)


def test_intervals_require_flag(features):
    model = LGBMForecaster(**FAST).fit(features, features["cnt"])
    with pytest.raises(RuntimeError):
        model.predict_interval(features)


def test_backtest_never_trains_on_the_future(features):
    seen = []

    class Spy(SeasonalNaive):
        def fit(self, X, y):
            seen.append(X.index.max())
            return self

    folds = monthly_folds("2011-02-01", "2011-03-01")
    predictions = run_backtest(features, {"spy": Spy}, folds)
    assert seen[0] < pd.Timestamp("2011-02-01")
    assert predictions.index.min() >= pd.Timestamp("2011-02-01")
    assert set(summarize(predictions, ["spy"]).columns) == {"mae", "rmse", "wape", "bias"}
