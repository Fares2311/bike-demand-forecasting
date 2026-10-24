import pytest

from bikeforecast import metrics


def test_metrics_by_hand():
    y, yhat = [10, 20, 30], [12, 18, 30]
    assert metrics.mae(y, yhat) == pytest.approx(4 / 3)
    assert metrics.rmse(y, yhat) == pytest.approx((8 / 3) ** 0.5)
    assert metrics.wape(y, yhat) == pytest.approx(4 / 60)
    assert metrics.bias(y, yhat) == pytest.approx(0)


def test_wape_handles_zero_hours():
    assert metrics.wape([0, 10], [1, 10]) == pytest.approx(0.1)


def test_coverage():
    assert metrics.coverage([1, 5, 10], [0, 6, 9], [2, 7, 11]) == pytest.approx(2 / 3)
