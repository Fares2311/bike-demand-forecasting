import pandas as pd

from bikeforecast.data import clean_hourly


def test_clean_hourly_has_no_gaps(raw_hourly, hourly):
    expected = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    assert hourly.index.equals(expected)
    assert hourly.notna().all().all()
    assert hourly["is_filled"].sum() == len(expected) - len(raw_hourly)


def test_filled_hours_have_zero_rentals(hourly):
    filled = hourly[hourly["is_filled"] == 1]
    assert (filled["cnt"] == 0).all()
    assert (filled[["casual", "registered"]].sum(axis=1) == filled["cnt"]).all()


def test_units_are_converted():
    raw = pd.DataFrame(
        {
            "dteday": ["2011-01-01"],
            "hr": [5],
            "temp": [0.5],
            "atemp": [0.5],
            "hum": [0.5],
            "windspeed": [0.5],
            "weathersit": [1],
            "holiday": [0],
            "workingday": [0],
            "casual": [1],
            "registered": [2],
            "cnt": [3],
        }
    )
    row = clean_hourly(raw).iloc[0]
    assert row.name == pd.Timestamp("2011-01-01 05:00")
    assert row["temp_c"] == 20.5
    assert row["feels_like_c"] == 25
    assert row["humidity"] == 50
