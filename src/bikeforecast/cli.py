"""Command line entry point.

bikeforecast backtest                    # 2012 backtest -> reports/metrics.json + figures
bikeforecast forecast --date 2012-12-24  # train on earlier data, forecast one day
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from bikeforecast import metrics, plots
from bikeforecast.backtest import monthly_folds, run_backtest, summarize, summarize_by
from bikeforecast.config import FIGURES_DIR, HOURLY_CSV, REPORTS_DIR, TARGET
from bikeforecast.data import load_hourly
from bikeforecast.features import build_features
from bikeforecast.models import LGBMForecaster, RawLGBMForecaster, RidgeForecaster, SeasonalNaive

logger = logging.getLogger(__name__)

BEST = "lgbm"
BASELINE = "naive_week"
MODELS = {
    "naive_week": (lambda: SeasonalNaive("lag_168"), "Seasonal naive (same hour last week)"),
    "naive_day": (lambda: SeasonalNaive("lag_24"), "Seasonal naive (same hour yesterday)"),
    "ridge_noweather": (lambda: RidgeForecaster(use_weather=False), "Ridge, no weather"),
    "ridge": (lambda: RidgeForecaster(), "Ridge (log-linear)"),
    "lgbm_noweather": (lambda: LGBMForecaster(use_weather=False), "LightGBM, no weather"),
    "lgbm_raw": (lambda: RawLGBMForecaster(), "LightGBM, raw counts"),
    BEST: (lambda: LGBMForecaster(intervals=True), "LightGBM, level-normalised"),
}
LABELS = {name: label for name, (_, label) in MODELS.items()}


def backtest(start: str, end: str, data_path: Path, reports_dir: Path, figures_dir: Path) -> dict:
    hourly = load_hourly(data_path)
    data = build_features(hourly)
    folds = monthly_folds(start, end)
    factories = {name: factory for name, (factory, _) in MODELS.items()}
    predictions = run_backtest(data, factories, folds, interval_model=BEST)

    names = list(MODELS)
    summary = summarize(predictions, names)
    by_month = summarize_by(predictions, names, "fold")
    by_hour = summarize_by(predictions, names, predictions.index.hour)
    daily = predictions[["actual", *names]].resample("D").sum()

    report = {
        "backtest": {"start": start, "end": end, "folds": len(folds), "hours": len(predictions)},
        "hourly_metrics": summary.to_dict(orient="index"),
        "daily_total_wape": {n: metrics.wape(daily["actual"], daily[n]) for n in names},
        "improvement_vs_naive_mae": 1 - summary.loc[BEST, "mae"] / summary.loc[BASELINE, "mae"],
        "interval_80": {
            "coverage": metrics.coverage(
                predictions["actual"], predictions["low"], predictions["high"]
            ),
            "mean_width": float((predictions["high"] - predictions["low"]).mean()),
        },
        "mae_by_month": by_month.to_dict(),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    predictions.to_csv(reports_dir / "backtest_predictions.csv.gz", float_format="%.2f")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plots.plot_daily_demand(hourly, figures_dir / "daily_demand.png")
    plots.plot_weather_effect(hourly, figures_dir / "weather_effect.png")
    plots.plot_model_comparison(summary, LABELS, BASELINE, figures_dir / "model_comparison.png")
    compare = [BASELINE, "ridge", "lgbm_raw", BEST]
    plots.plot_error_by(
        by_month[compare],
        LABELS,
        figures_dir / "mae_by_month.png",
        "Backtest MAE by month (model retrained at the start of each month)",
        "Month",
    )
    plots.plot_error_by(
        by_hour[compare],
        LABELS,
        figures_dir / "mae_by_hour.png",
        "Backtest MAE by hour of day",
        "Hour of day",
    )
    first_test_day = predictions.index.min()
    week = (first_test_day + pd.Timedelta(days=156)).normalize()
    week_start = week - pd.Timedelta(days=week.day_of_week)
    plots.plot_forecast_window(
        predictions,
        BEST,
        BASELINE,
        str(week_start),
        str(week_start + pd.Timedelta(hours=167)),
        figures_dir / "forecast_week.png",
        f"Day-ahead forecasts, week of {week_start:%d %b %Y}",
    )
    final_model = MODELS[BEST][0]().fit(data, data[TARGET])
    plots.plot_feature_importance(
        final_model.feature_importance(), figures_dir / "feature_importance.png"
    )
    return report


def forecast_day(date: str, data_path: Path, figures_dir: Path) -> pd.DataFrame:
    data = build_features(load_hourly(data_path))
    day = pd.Timestamp(date).normalize()
    history = data[data.index < day]
    target_day = data[(data.index >= day) & (data.index < day + pd.Timedelta(days=1))]
    if target_day.empty:
        raise SystemExit(f"{date} is outside the dataset (2011-01-15 to 2012-12-31)")
    model = LGBMForecaster(intervals=True).fit(history, history[TARGET])
    low, high = model.predict_interval(target_day)
    forecast = pd.DataFrame(
        {
            "forecast": model.predict(target_day),
            "low": low,
            "high": high,
            "actual": target_day[TARGET],
        },
        index=target_day.index,
    ).round(0)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plots.plot_day_forecast(
        forecast,
        figures_dir / f"forecast_{day:%Y-%m-%d}.png",
        f"Day-ahead forecast for {day:%A %d %B %Y}",
    )
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(prog="bikeforecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bt = sub.add_parser("backtest", help="rolling-origin backtest of all models")
    bt.add_argument("--start", default="2012-01-01")
    bt.add_argument("--end", default="2013-01-01")
    fc = sub.add_parser("forecast", help="forecast the 24 hours of one day")
    fc.add_argument("--date", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "backtest":
        report = backtest(args.start, args.end, HOURLY_CSV, REPORTS_DIR, FIGURES_DIR)
        table = pd.DataFrame(report["hourly_metrics"]).T.rename(index=LABELS)
        print(table.round(3).to_string())
        print(f"\nImprovement vs naive: {report['improvement_vs_naive_mae']:.1%}")
        print(f"80% interval coverage: {report['interval_80']['coverage']:.1%}")
    else:
        forecast = forecast_day(args.date, HOURLY_CSV, FIGURES_DIR)
        print(forecast.to_string())
        print(
            f"\nForecast total {forecast['forecast'].sum():.0f}, "
            f"actual total {forecast['actual'].sum():.0f}"
        )


if __name__ == "__main__":
    main()
