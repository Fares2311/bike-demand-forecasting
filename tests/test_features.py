import numpy as np
import pandas as pd

from bikeforecast.features import DEMAND_FEATURES, assert_no_leakage, build_features


def test_lag_alignment(features, hourly):
    t = features.index[100]
    assert features.loc[t, "lag_24"] == hourly.loc[t - pd.Timedelta(hours=24), "cnt"]
    assert features.loc[t, "lag_168"] == hourly.loc[t - pd.Timedelta(hours=168), "cnt"]
    assert_no_leakage(features)


def test_daily_features_use_only_previous_days(features, hourly):
    t = features.index[200]
    day = t.normalize()
    prev_day = hourly.loc[day - pd.Timedelta(days=1) : day - pd.Timedelta(hours=1), "cnt"]
    assert features.loc[t, "prev_day_total"] == prev_day.sum()
    last_week = hourly.loc[day - pd.Timedelta(days=7) : day - pd.Timedelta(hours=1), "cnt"]
    assert np.isclose(features.loc[t, "level_7d"], last_week.mean())


def test_no_feature_sees_the_day_being_forecast(hourly):
    """Changing demand on day D or later must not change the features of day D."""
    day = hourly.index.normalize().unique()[40]
    perturbed = hourly.copy()
    perturbed.loc[perturbed.index >= day, "cnt"] *= 10
    original = build_features(hourly).loc[str(day.date()), DEMAND_FEATURES]
    changed = build_features(perturbed).loc[str(day.date()), DEMAND_FEATURES]
    pd.testing.assert_frame_equal(original, changed)


def test_warm_up_rows_are_dropped(features, hourly):
    assert features.index.min() == hourly.index.min() + pd.Timedelta(days=14)
    assert features[DEMAND_FEATURES].notna().all().all()
