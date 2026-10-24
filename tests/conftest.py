"""Synthetic data in the raw UCI format, so the tests run offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bikeforecast.data import clean_hourly
from bikeforecast.features import build_features


def make_raw_hourly(days: int = 70, seed: int = 0, drop_rows: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2011-01-01", periods=days * 24, freq="h")
    hour, dow = ts.hour.to_numpy(), ts.day_of_week.to_numpy()
    workingday = (dow < 5).astype(int)
    commute = np.exp(-((hour - 8) ** 2) / 2) + np.exp(-((hour - 17.5) ** 2) / 3)
    leisure = np.exp(-((hour - 14) ** 2) / 10)
    temp = (
        0.4
        + 0.2 * np.sin(2 * np.pi * np.arange(len(ts)) / (24 * 365))
        + rng.normal(0, 0.03, len(ts))
    )
    trend = 1 + np.arange(len(ts)) / len(ts)  # demand doubles over the period
    mean = (
        trend * (10 + 300 * np.where(workingday, commute, 0.2 * commute + leisure)) * (0.5 + temp)
    )
    cnt = rng.poisson(mean) + 1
    casual = (cnt * 0.2).astype(int)
    raw = pd.DataFrame(
        {
            "instant": np.arange(1, len(ts) + 1),
            "dteday": ts.strftime("%Y-%m-%d"),
            "season": 1,
            "yr": 0,
            "mnth": ts.month,
            "hr": hour,
            "holiday": 0,
            "weekday": (dow + 1) % 7,
            "workingday": workingday,
            "weathersit": rng.choice([1, 2, 3], len(ts), p=[0.65, 0.28, 0.07]),
            "temp": temp,
            "atemp": temp * 0.9,
            "hum": rng.uniform(0.3, 0.9, len(ts)),
            "windspeed": rng.uniform(0, 0.4, len(ts)),
            "casual": casual,
            "registered": cnt - casual,
            "cnt": cnt,
        }
    )
    # like the real file, drop a few hours
    removed = rng.choice(np.arange(24 * 20, len(raw) - 24), size=drop_rows, replace=False)
    return raw.drop(index=removed).reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_hourly() -> pd.DataFrame:
    return make_raw_hourly()


@pytest.fixture(scope="session")
def hourly(raw_hourly) -> pd.DataFrame:
    return clean_hourly(raw_hourly)


@pytest.fixture(scope="session")
def features(hourly) -> pd.DataFrame:
    return build_features(hourly)
