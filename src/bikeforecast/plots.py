"""Report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED = "#2563eb", "#f97316", "#16a34a", "#6b7280", "#dc2626"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def plot_daily_demand(hourly: pd.DataFrame, path: Path) -> None:
    daily = hourly["cnt"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily, color=BLUE, alpha=0.35, lw=0.8, label="Daily rentals")
    ax.plot(
        daily.index, daily.rolling(28, center=True).mean(), color=BLUE, lw=2, label="28-day average"
    )
    ax.set(title="Capital Bikeshare daily rentals, 2011-2012", ylabel="Rentals per day")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def plot_forecast_window(
    predictions: pd.DataFrame,
    model: str,
    baseline: str,
    start: str,
    end: str,
    path: Path,
    title: str,
) -> None:
    window = predictions.loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(
        window.index,
        window["low"],
        window["high"],
        color=BLUE,
        alpha=0.15,
        label="80% prediction interval",
    )
    ax.plot(window.index, window["actual"], color="black", lw=1.6, label="Actual")
    ax.plot(window.index, window[model], color=BLUE, lw=1.6, label="LightGBM forecast")
    ax.plot(
        window.index,
        window[baseline],
        color=ORANGE,
        lw=1,
        ls="--",
        label="Seasonal naive (last week)",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set(title=title, ylabel="Rentals per hour")
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_model_comparison(
    summary: pd.DataFrame, labels: dict[str, str], baseline: str, path: Path
) -> None:
    data = summary.sort_values("mae", ascending=False)
    base = summary.loc[baseline, "mae"]
    colors = [GREEN if name == data["mae"].idxmin() else GREY for name in data.index]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bars = ax.barh([labels[n] for n in data.index], data["mae"], color=colors)
    ax.bar_label(
        bars,
        labels=[
            f"{m:.1f}" + ("" if n == baseline else f"  ({m / base - 1:+.0%} vs naive)")
            for n, m in data["mae"].items()
        ],
        padding=3,
    )
    ax.set(
        title="Backtest MAE, Jan-Dec 2012 (lower is better)",
        xlabel="Mean absolute error (rentals per hour)",
    )
    ax.margins(x=0.35)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_error_by(
    table: pd.DataFrame, labels: dict[str, str], path: Path, title: str, xlabel: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [ORANGE, GREY, GREEN, BLUE]
    positions = range(len(table))
    for color, name in zip(colors, table.columns, strict=False):
        ax.plot(positions, table[name], marker="o", ms=3.5, color=color, label=labels[name])
    ax.set_xticks(positions, labels=table.index.astype(str))
    ax.set(title=title, xlabel=xlabel, ylabel="MAE (rentals per hour)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, path: Path) -> None:
    share = (importance / importance.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(share.index, share.values, color=BLUE)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.set(title="LightGBM feature importance (share of total gain)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_day_forecast(forecast: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    hours = forecast.index.hour
    ax.fill_between(
        hours,
        forecast["low"],
        forecast["high"],
        color=BLUE,
        alpha=0.15,
        label="80% prediction interval",
    )
    ax.plot(hours, forecast["forecast"], color=BLUE, marker="o", ms=3, label="Forecast")
    if "actual" in forecast:
        ax.plot(hours, forecast["actual"], color="black", marker="o", ms=3, label="Actual")
    ax.set(title=title, xlabel="Hour of day", ylabel="Rentals per hour", xticks=range(0, 24, 2))
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def plot_weather_effect(hourly: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    by_temp = hourly.groupby(hourly["temp_c"].round(0))["cnt"].mean()
    axes[0].plot(by_temp.index, by_temp.values, color=ORANGE, marker="o", ms=3)
    axes[0].set(
        title="Average hourly rentals by temperature",
        xlabel="Temperature (°C)",
        ylabel="Rentals per hour",
    )
    labels = {1: "Clear", 2: "Mist / cloudy", 3: "Light rain / snow", 4: "Heavy rain"}
    by_weather = hourly.groupby("weather")["cnt"].mean()
    axes[1].bar(
        [labels[w] for w in by_weather.index],
        by_weather.values,
        color=[GREEN, BLUE, ORANGE, RED][: len(by_weather)],
    )
    axes[1].set(title="Average hourly rentals by weather", ylabel="Rentals per hour")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
