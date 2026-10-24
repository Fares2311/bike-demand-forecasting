"""Download the UCI bike sharing data and build a clean hourly series."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from bikeforecast.config import DATA_URL, HOURLY_CSV

logger = logging.getLogger(__name__)

# weather is stored scaled to [0, 1]; the max values come from the dataset readme
TEMP_MAX_C, FEELS_LIKE_MAX_C, HUMIDITY_MAX, WIND_MAX_KMH = 41, 50, 100, 67
WEATHER_COLUMNS = ["temp_c", "feels_like_c", "humidity", "windspeed_kmh", "weather"]


def download_data(dest: Path = HOURLY_CSV, url: str = DATA_URL, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read("hour.csv"))
    return dest


def clean_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per hour, readable units, no gaps.

    165 hours are missing from the raw file. They are night hours with no rentals (the smallest
    count in the file is 1) or shutdowns like Hurricane Irene (Aug 2011) and Sandy (Oct 2012).
    Both mean zero rentals, so we fill them with 0 and flag them in `is_filled`. Weather is
    forward-filled.
    """
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.set_index("timestamp").sort_index()
    out = pd.DataFrame(
        {
            "cnt": df["cnt"],
            "casual": df["casual"],
            "registered": df["registered"],
            "temp_c": df["temp"] * TEMP_MAX_C,
            "feels_like_c": df["atemp"] * FEELS_LIKE_MAX_C,
            "humidity": df["hum"] * HUMIDITY_MAX,
            "windspeed_kmh": df["windspeed"] * WIND_MAX_KMH,
            "weather": df["weathersit"],  # 1 = clear ... 4 = heavy rain / snow
            "holiday": df["holiday"],
            "workingday": df["workingday"],
        }
    )
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="h", name="timestamp")
    out = out.reindex(full_index)
    out["is_filled"] = out["cnt"].isna().astype(int)
    out[["cnt", "casual", "registered"]] = out[["cnt", "casual", "registered"]].fillna(0)
    out[WEATHER_COLUMNS] = out[WEATHER_COLUMNS].ffill()

    # holiday / workingday are daily values, copy them to the filled hours
    day = out.index.normalize()
    for column in ["holiday", "workingday"]:
        out[column] = out[column].groupby(day).transform("max")
    int_columns = ["cnt", "casual", "registered", "weather", "holiday", "workingday"]
    out[int_columns] = out[int_columns].astype(int)
    return out


def load_hourly(path: Path = HOURLY_CSV) -> pd.DataFrame:
    if not path.exists():
        download_data(dest=path)
    return clean_hourly(pd.read_csv(path))
