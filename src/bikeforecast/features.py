"""Features for day-ahead forecasting.

The forecast for day D is made at midnight, so we can only use demand from before D 00:00:
lags of 24h or more, or totals of previous days. tests/test_features.py checks this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bikeforecast.config import HORIZON_HOURS, TARGET

CALENDAR_FEATURES = ["hour", "dayofweek", "month", "holiday", "workingday", "is_weekend"]
WEATHER_FEATURES = ["temp_c", "feels_like_c", "humidity", "windspeed_kmh", "weather"]
LAG_HOURS = [24, 48, 168, 336]
DEMAND_FEATURES = [
    *(f"lag_{h}" for h in LAG_HOURS),
    "same_hour_mean_7d",
    "prev_day_total",
    "level_7d",
]
FOURIER_FEATURES = [
    f"{name}_{fn}_{k}" for name in ("daily", "weekly") for k in (1, 2, 3) for fn in ("sin", "cos")
]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.day_of_week
    df["month"] = idx.month
    df["is_weekend"] = (idx.day_of_week >= 5).astype(int)
    hours_in_week = idx.day_of_week * 24 + idx.hour
    for k in (1, 2, 3):
        df[f"daily_sin_{k}"] = np.sin(2 * np.pi * k * idx.hour / 24)
        df[f"daily_cos_{k}"] = np.cos(2 * np.pi * k * idx.hour / 24)
        df[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * hours_in_week / 168)
        df[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * hours_in_week / 168)
    return df


def add_demand_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    y = df[target]
    for h in LAG_HOURS:
        df[f"lag_{h}"] = y.shift(h)
    df["same_hour_mean_7d"] = pd.concat([y.shift(24 * d) for d in range(1, 8)], axis=1).mean(
        axis=1, skipna=False
    )

    # daily totals are shifted by whole days, then mapped back to each hour
    day = df.index.normalize()
    daily_total = y.groupby(day).sum()
    complete_days = y.groupby(day).size() == 24
    daily_total = daily_total.where(complete_days)
    prev_day_total = daily_total.shift(1, freq="D")
    level = daily_total.rolling("7D", min_periods=7).mean().shift(1, freq="D") / 24
    df["prev_day_total"] = prev_day_total.reindex(day).to_numpy()
    df["level_7d"] = level.reindex(day).to_numpy()
    return df


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add all features and drop the first 14 days (not enough history for the lags)."""
    df = add_demand_features(add_calendar(hourly.copy()))
    return df.dropna(subset=DEMAND_FEATURES)


def assert_no_leakage(df: pd.DataFrame, target: str = TARGET) -> None:
    """Raise if a lag is shorter than the forecast horizon."""
    for h in LAG_HOURS:
        if h < HORIZON_HOURS:
            raise ValueError(f"lag_{h} is shorter than the {HORIZON_HOURS}h forecast horizon")
    lagged = df[target].shift(HORIZON_HOURS)
    mismatch = (df["lag_24"] != lagged) & lagged.notna()
    if mismatch.any():
        raise ValueError("lag_24 is not aligned with a 24-hour shift of the target")
