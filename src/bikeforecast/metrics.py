"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: total absolute error / total demand.

    Used instead of MAPE because some hours have zero rentals, and busy hours should count more.
    """
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def bias(y, yhat) -> float:
    """Total error relative to total demand (negative means under-forecasting)."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float((yhat - y).sum() / y.sum())


def coverage(y, low, high) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(low)) & (y <= np.asarray(high))))


def all_metrics(y, yhat) -> dict[str, float]:
    return {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "wape": wape(y, yhat),
        "bias": bias(y, yhat),
    }
