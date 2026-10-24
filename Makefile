.PHONY: install test lint format backtest forecast notebook

install:  ## Install the package with dev dependencies
	pip install -e ".[dev]"

backtest:  ## Rolling-origin backtest of all models over 2012 (~1 min)
	bikeforecast backtest

forecast:  ## Example day-ahead forecast
	bikeforecast forecast --date 2012-10-17

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

notebook:  ## Re-run the EDA notebook
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_analysis.ipynb
