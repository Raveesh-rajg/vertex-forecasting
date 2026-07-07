import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from forecasting.data import load_series
from forecasting.models import SeasonalNaive, ETS
from forecasting.backtest import rolling_origin, smape, mape


@pytest.fixture(scope="session")
def y():
    return load_series()


class TestData:
    def test_series_shape(self, y):
        assert len(y) == 108
        assert y.index.freqstr in ("MS", "MS-JAN")
        assert (y > 0).all()


class TestMetrics:
    def test_perfect_forecast_zero_error(self):
        a = np.array([10.0, 20.0])
        assert smape(a, a) == 0 and mape(a, a) == 0

    def test_smape_symmetric_bounded(self):
        assert smape(np.array([100.0]), np.array([300.0])) == pytest.approx(100.0)
        assert smape(np.array([300.0]), np.array([100.0])) == pytest.approx(100.0)


class TestModels:
    def test_seasonal_naive_repeats_last_year(self, y):
        fc = SeasonalNaive().fit(y).forecast(12)
        assert np.array_equal(fc, y.iloc[-12:].to_numpy())

    def test_seasonal_naive_needs_full_season(self, y):
        with pytest.raises(ValueError):
            SeasonalNaive().fit(y.iloc[:6])

    def test_ets_forecast_plausible(self, y):
        fc = ETS().fit(y).forecast(6)
        assert len(fc) == 6
        assert (fc > 0).all()
        assert fc.max() < y.max() * 2  # sanity envelope


class TestBacktest:
    def test_folds_and_determinism(self, y):
        r1 = rolling_origin(y, SeasonalNaive)
        r2 = rolling_origin(y, SeasonalNaive)
        assert r1.n_folds == 19
        assert r1.smape == r2.smape

    def test_ets_beats_or_ties_naive(self, y):
        """The measured champion claim in the README, as a regression test."""
        naive = rolling_origin(y, SeasonalNaive)
        ets = rolling_origin(y, ETS)
        assert ets.smape <= naive.smape


class TestAPI:
    def test_endpoint(self):
        from fastapi.testclient import TestClient
        from forecasting.serve import app

        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "ok"
            r = client.get("/forecast", params={"h": 6})
            assert r.status_code == 200
            body = r.json()
            assert len(body["forecast"]) == 6
            assert body["model"] == "ets"
            assert client.get("/forecast", params={"h": 99}).status_code == 422
