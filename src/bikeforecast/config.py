"""Paths and constants."""

import os
from pathlib import Path


def _project_root() -> Path:
    """Use BIKEFORECAST_ROOT if set, otherwise walk up to the folder with pyproject.toml."""
    if os.getenv("BIKEFORECAST_ROOT"):
        return Path(os.environ["BIKEFORECAST_ROOT"])
    for folder in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    return Path.cwd()


PROJECT_ROOT = _project_root()
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HOURLY_CSV = RAW_DIR / "hour.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "cnt"
# forecasts are made at midnight for the next 24h, so features need lags of 24h or more
HORIZON_HOURS = 24
RANDOM_STATE = 42
